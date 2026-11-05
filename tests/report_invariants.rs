//! Verifier identities: any correct implementation satisfies them, with
//! no per-test expectations.
//!
//! 1. Every Measurement's provenance index resolves to a real
//!    instruction.
//! 2. The blocks' instructions sum to the kernel's instruction count.
//! 3. Two-path consistency: with every parameter bound, the FLOPs the
//!    annotated PTX's `.entry` line sums per thread equal an
//!    independently computed sum (per-block flat tallies ×
//!    numerically evaluated trip chains), the check that catches two
//!    code paths disagreeing.
//! 4. The annotated PTX of every fixture, every bound case of 3 and every
//!    Nsight Compute record is the input, line for line, each line
//!    followed by nothing or by ` // ` and its annotation.

use ptxroof::analysis::affine::sym_expr::SymbolicConstant;
use ptxroof::analysis::affine::trip_counts::trip_counts;
use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::collect::collect;
use ptxroof::analysis::instruction_counts::measurement::MeasureKind;
use ptxroof::analysis::instruction_counts::measurement::Precision;
use ptxroof::ptx::cfg::build_cfg;
use ptxroof::ptx::ir::Stmt;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::report::annotate::annotate;
use ptxroof::report::measured::NcuRecord;
use ptxroof::report::{AnalyzeOptions, BindingSpec};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const FIXTURES: &[&str] = &[
    "k1/k1.sm_80.ptx",
    "k1/k1.sm_89.ptx",
    "k2/k2.sm_80.ptx",
    "k5/k5.sm_80.ptx",
    "k5/k5.sm_89.ptx",
    "k11/k11.sm_80.ptx",
    "k12/k12.sm_80.ptx",
    "k14/k14.sm_80.ptx",
    "mma_demo/mma_demo.sm_80.ptx",
    "micro/single_loop.ptx",
    "micro/branchy.ptx",
    "micro/irreducible.ptx",
    "micro/no_loc.ptx",
    "micro/data_dep.ptx",
];

fn read(fixture: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    fs::read_to_string(&path).expect("fixture readable")
}

#[test]
fn every_measurement_provenance_resolves_to_an_instruction() {
    for fixture in FIXTURES {
        let src = read(fixture);
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            for bm in collect(&m, k, &cfg, &f) {
                for meas in &bm.measurements {
                    assert!(
                        matches!(k.stmts.get(meas.provenance), Some(Stmt::Instr(_))),
                        "{fixture}: provenance {} is not an instruction",
                        meas.provenance
                    );
                }
            }
        }
    }
}

#[test]
fn block_instructions_sum_to_kernel_instruction_count() {
    for fixture in FIXTURES {
        let src = read(fixture);
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            let from_blocks: usize = collect(&m, k, &cfg, &f)
                .iter()
                .map(|b| b.instructions.len())
                .sum();
            let from_stmts = k
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::Instr(_)))
                .count();
            assert_eq!(from_blocks, from_stmts, "{fixture}");
        }
    }
}

/// Fixtures with fully resolvable trips and one symbolic parameter.
const BOUND: &[(&str, usize)] = &[
    ("k1/k1.sm_80.ptx", 2),
    ("k1/k1.sm_89.ptx", 2),
    ("k2/k2.sm_80.ptx", 2),
    ("k5/k5.sm_80.ptx", 2),
    ("k5/k5.sm_89.ptx", 2),
    ("k11/k11.sm_80.ptx", 2),
    ("k12/k12.sm_80.ptx", 2),
    ("k14/k14.sm_80.ptx", 2),
    ("micro/single_loop.ptx", 1),
    ("micro/branchy.ptx", 1),
];
const BOUND_VALUE: i64 = 4099; // deliberately not a multiple of the unrolls

fn bound(param: usize) -> AnalyzeOptions {
    AnalyzeOptions {
        bindings: vec![BindingSpec {
            index: Some(param),
            name: "N".into(),
            value: BOUND_VALUE,
        }],
        ..Default::default()
    }
}

#[test]
fn bound_flop_totals_agree_with_an_independent_evaluation() {
    for &(fixture, param) in BOUND {
        let src = read(fixture);
        let value = BOUND_VALUE;

        // Path 1: the annotated PTX's `.entry` line.
        let annotated = annotate(&src, &bound(param)).expect("annotates");
        let (reported, cuda_core, f32_only) = entry_flops(&annotated);

        // Path 2: flat per-block tallies × numerically evaluated chains.
        let m = parse(&src).expect("parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let info = trip_counts(&m, k, &cfg, &f, &Default::default());
        let bind_map: HashMap<SymbolicConstant, i64> =
            [(SymbolicConstant::KernelParam(param), value)]
                .into_iter()
                .collect();
        let trips_num: Vec<i64> = info
            .trips
            .iter()
            .map(|t| match t {
                Ok(e) => e.bind(&bind_map).as_const().expect("trips fully bound"),
                Err(_) => panic!("{fixture}: unexpected unknown trips"),
            })
            .collect();
        let blocks = collect(&m, k, &cfg, &f);
        let mut independent = 0i64;
        for bm in &blocks {
            let mut mult = 1i64;
            let mut cur = f.block_loop[bm.block.0 as usize];
            while let Some(l) = cur {
                mult *= trips_num[l.0 as usize];
                cur = f.get(l).parent;
            }
            let flat: u64 = bm
                .measurements
                .iter()
                .filter(|m| matches!(m.kind, MeasureKind::Flops { .. }))
                .map(|m| m.count)
                .sum();
            independent += flat as i64 * mult;
        }
        assert_eq!(reported, independent, "{fixture}: two paths disagree");
        // Sanity: cuda-core work in this corpus is all f32.
        assert_eq!(f32_only, cuda_core, "{fixture}");
    }
}

/// The FLOPs the `.entry` line of `annotated` sums per thread, a bound's
/// value: of every pipe and precision, of the cuda-core pipe, and of its
/// f32.
fn entry_flops(annotated: &str) -> (i64, i64, i64) {
    let entry = (annotated.lines())
        .find(|l| l.contains(".entry "))
        .expect("an .entry line");
    let sums = (entry.split(" | "))
        .find_map(|p| p.strip_prefix("per thread: "))
        .expect("per-thread sums");
    let (mut all, mut cuda_core, mut f32) = (0, 0, 0);
    for item in sums.split("; ") {
        let Some((pipe, counts)) = item.split_once("flops ") else {
            continue;
        };
        for count in counts.split(" + ") {
            let (precision, n) = count.split_once(' ').expect("a precision and a count");
            let n = n.trim_start_matches("<= ").replace(',', "");
            let n: i64 = n.parse().expect("a bound count");
            all += n;
            if pipe.is_empty() {
                cuda_core += n;
                f32 += if precision == Precision::F32.key() {
                    n
                } else {
                    0
                };
            }
        }
    }
    (all, cuda_core, f32)
}

fn files(dir: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("directory readable") {
        let path = entry.expect("entry readable").path();
        if path.is_dir() {
            files(&path, suffix, out);
        } else if path.to_string_lossy().ends_with(suffix) {
            out.push(path);
        }
    }
}

/// Every fixture unbound, the bound cases of 3, and every Nsight Compute
/// record with its PTX.
fn every_run() -> Vec<(PathBuf, AnalyzeOptions)> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut runs: Vec<(PathBuf, AnalyzeOptions)> = Vec::new();
    let mut ptx = Vec::new();
    files(&repo.join("tests/fixtures"), ".ptx", &mut ptx);
    runs.extend(ptx.into_iter().map(|p| (p, AnalyzeOptions::default())));
    runs.extend(
        (BOUND.iter()).map(|&(f, param)| (repo.join("tests/fixtures").join(f), bound(param))),
    );
    let mut records = Vec::new();
    files(&repo.join("tests/fixtures"), ".ncu.json", &mut records);
    files(&repo.join("tests/lit/Inputs"), ".ncu.json", &mut records);
    for path in records {
        let text = fs::read_to_string(&path).expect("record readable");
        let fields: serde_json::Value = serde_json::from_str(&text).expect("record is JSON");
        let record: NcuRecord = serde_json::from_str(&text).expect("record parses");
        let opts = AnalyzeOptions {
            bindings: record.bindings(),
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        runs.push((
            repo.join(fields["ptx"].as_str().expect("record names its ptx")),
            opts,
        ));
    }
    assert!(runs.len() > 40, "{} runs", runs.len());
    runs
}

#[test]
fn annotated_ptx_is_the_input_with_a_comment_on_some_lines() {
    for (path, opts) in &every_run() {
        let src = fs::read_to_string(path).expect("ptx readable");
        let out = annotate(&src, opts).expect("annotates");
        let (input, output): (Vec<&str>, Vec<&str>) =
            (src.split('\n').collect(), out.split('\n').collect());
        assert_eq!(input.len(), output.len(), "{}", path.display());
        for (i, o) in input.iter().zip(&output) {
            let rest = o.strip_prefix(i).expect("the input line");
            assert!(
                rest.is_empty() || rest.starts_with(" // "),
                "{}: {o}",
                path.display()
            );
        }
    }
}
