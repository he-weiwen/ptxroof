//! Verifier identities: any correct implementation satisfies them, with
//! no per-test expectations.
//!
//! 1. Every Measurement's provenance index resolves to a real
//!    instruction.
//! 2. Per-block class tallies sum to the kernel's instruction count.
//! 3. The block table is the CFG: block instruction counts sum to the
//!    kernel's instruction count, every successor and every loop
//!    label names a listed block, and each loop's header and latches
//!    are marked on exactly the blocks the loop forest says.
//! 4. Two-path consistency: with every parameter bound, the report's
//!    kernel flop total equals an independently-computed sum
//!    (per-block flat tallies × numerically-evaluated trip chains) —
//!    the check that catches two code paths disagreeing.
//! 5. The JSON report's own identities, on every fixture, every bound
//!    case of 4 and every Nsight Compute record: instruction classes
//!    sum to the total and unparsed statements are listed as unknowns;
//!    instruction kinds sum to the total and opcodes to their kind;
//!    AI(global) is flops over global bytes wherever both are numbers;
//!    the coverage fractions agree with the class counts and the loop
//!    tree.

use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::classify::Precision;
use ptxroof::analysis::instruction_counts::collect::collect;
use ptxroof::analysis::instruction_counts::measurement::MeasureKind;
use ptxroof::analysis::loop_names::loop_names;
use ptxroof::analysis::scalar::trip_counts::trip_counts;
use ptxroof::ptx::cfg::build_cfg;
use ptxroof::ptx::ir::Stmt;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::report::measured::NcuRecord;
use ptxroof::report::schema::{Aggregates, Count, LoopNode};
use ptxroof::report::{AnalyzeOptions, BindingSpec, Report, analyze};
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
fn block_class_tallies_sum_to_kernel_instruction_count() {
    for fixture in FIXTURES {
        let src = read(fixture);
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            let from_blocks: u32 = collect(&m, k, &cfg, &f)
                .iter()
                .map(|b| b.class_counts.total)
                .sum();
            let from_stmts = k
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::Instr(_)))
                .count() as u32;
            assert_eq!(from_blocks, from_stmts, "{fixture}");
        }
    }
}

#[test]
fn block_table_matches_the_cfg() {
    for fixture in FIXTURES {
        let src = read(fixture);
        let report = analyze(&src, fixture, &AnalyzeOptions::default()).expect("analyzes");
        for k in &report.kernels {
            let names: Vec<&str> = k.blocks.iter().map(|b| b.name.as_str()).collect();
            let total: u64 = k.blocks.iter().map(|b| b.instructions).sum();
            assert_eq!(total, k.instruction_classes.total, "{fixture} {}", k.name);
            for b in &k.blocks {
                for s in &b.successors {
                    assert!(names.contains(&s.as_str()), "{fixture}: {} -> {s}", b.name);
                }
            }
            fn walk<'a>(
                nodes: &'a [ptxroof::report::schema::LoopNode],
                out: &mut Vec<&'a ptxroof::report::schema::LoopNode>,
            ) {
                for n in nodes {
                    out.push(n);
                    walk(&n.loops, out);
                }
            }
            let mut loops = Vec::new();
            walk(&k.loops, &mut loops);
            for l in loops {
                let header: Vec<&str> = k
                    .blocks
                    .iter()
                    .filter(|b| {
                        b.r#loop
                            .as_ref()
                            .is_some_and(|bl| bl.name == l.name && bl.header)
                    })
                    .map(|b| b.name.as_str())
                    .collect();
                assert_eq!(
                    header,
                    [l.label.as_str()],
                    "{fixture}: header of {}",
                    l.name
                );
                assert!(
                    k.blocks
                        .iter()
                        .any(|b| b.r#loop.as_ref().is_some_and(|bl| bl.latch)
                            && b.successors.contains(&l.label)),
                    "{fixture}: no latch branches to {}",
                    l.label
                );
            }
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

        // Path 1: the report.
        let report = analyze(&src, fixture, &bound(param)).expect("analyzes");
        let totals = &report.kernels[0].totals;
        let table_total =
            |t: &std::collections::BTreeMap<String, ptxroof::report::schema::Count>| -> i64 {
                t["total"].expr.parse().expect("bound total is numeric")
            };
        let reported = table_total(&totals.flops)
            + table_total(&totals.tensor_flops)
            + table_total(&totals.sfu_flops)
            + table_total(&totals.atomic_flops);

        // Path 2: flat per-block tallies × numerically evaluated chains.
        let m = parse(&src).expect("parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names);
        let bind_map: HashMap<String, i64> =
            [(format!("param_{param}"), value)].into_iter().collect();
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
        let f32_only: i64 = totals.flops[Precision::F32.key()]
            .expr
            .parse()
            .expect("numeric");
        assert_eq!(f32_only, table_total(&totals.flops), "{fixture}");
    }
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

fn number(count: &Count) -> Option<i64> {
    count.expr.parse().ok()
}

fn aggregate_violations(agg: &Aggregates, at: &str, out: &mut Vec<String>) {
    let ins = &agg.instructions;
    let kinds: Option<Vec<(&String, i64, i64)>> = (ins.by_kind.iter())
        .map(|(kind, k)| {
            let opcodes = k.opcodes.values().map(number).sum::<Option<i64>>()?;
            Some((kind, number(&k.total)?, opcodes))
        })
        .collect();
    if let (Some(total), Some(kinds)) = (number(&ins.total), kinds) {
        let parts: i64 = kinds.iter().map(|k| k.1).sum();
        if parts != total {
            out.push(format!(
                "instruction-kinds-sum: {at}: kinds sum to {parts}, total is {total}"
            ));
        }
        for (kind, total, opcodes) in kinds {
            if total != opcodes {
                out.push(format!(
                    "opcodes-sum: {at}: {kind} opcodes sum to {opcodes}, kind is {total}"
                ));
            }
        }
    }
    let pipes = [
        &agg.flops,
        &agg.tensor_flops,
        &agg.sfu_flops,
        &agg.atomic_flops,
    ];
    let flops = pipes
        .iter()
        .map(|t| number(&t["total"]))
        .sum::<Option<i64>>();
    let global = &agg.bytes["global"];
    let bytes = number(&global.load)
        .zip(number(&global.store))
        .map(|(l, s)| l + s);
    if let (Some(ai), Some(flops), Some(bytes)) = (&agg.ai_global, flops, bytes)
        && bytes > 0
        && (ai.value - flops as f64 / bytes as f64).abs() > 1e-9
    {
        out.push(format!(
            "ai-consistency: {at}: ai_global {} != {flops}/{bytes}",
            ai.value
        ));
    }
}

fn report_violations(report: &Report) -> Vec<String> {
    let mut out = Vec::new();
    let (mut total, mut unknown, mut loops, mut resolved) = (0, 0, 0, 0);
    for k in &report.kernels {
        let c = &k.instruction_classes;
        let classes = c.flop + c.non_flop_arith + c.memory + c.sync;
        let classes = classes + c.communication + c.control + c.ignore + c.unknown;
        if classes != c.total {
            out.push(format!(
                "instruction-accounting: {}: classes sum to {classes}, total is {}",
                k.name, c.total
            ));
        }
        let listed: u64 = (k.unknowns.iter())
            .filter(|u| u.what == "unparsed statement")
            .filter_map(|u| u.count)
            .sum();
        if listed != c.unparsed {
            out.push(format!(
                "unparsed-surfaced: {}: {} unparsed, {listed} listed",
                k.name, c.unparsed
            ));
        }
        (total, unknown) = (total + c.total, unknown + c.unknown);
        aggregate_violations(&k.totals, &format!("{}/totals", k.name), &mut out);
        let mut stack: Vec<(&LoopNode, String)> = Vec::new();
        stack.extend(
            k.loops
                .iter()
                .map(|l| (l, format!("{}/{}", k.name, l.name))),
        );
        while let Some((l, at)) = stack.pop() {
            (loops, resolved) = (loops + 1, resolved + u64::from(l.trips.expr.is_some()));
            aggregate_violations(&l.per_iteration, &at, &mut out);
            stack.extend(l.loops.iter().map(|n| (n, format!("{at}/{}", n.name))));
        }
    }
    let fraction = |metric: &str| report.coverage.get(metric).map(|f| (f.num, f.den));
    if fraction("instructions_classified") != Some((total - unknown, total)) {
        out.push(format!(
            "classification-coverage: not {}/{total}",
            total - unknown
        ));
    }
    if fraction("loop_trips_resolved") != Some((resolved, loops)) {
        out.push(format!("trip-coverage: not {resolved}/{loops}"));
    }
    out
}

#[test]
fn json_report_identities_hold() {
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
        };
        runs.push((
            repo.join(fields["ptx"].as_str().expect("record names its ptx")),
            opts,
        ));
    }
    assert!(runs.len() > 40, "{} runs", runs.len());
    for (path, opts) in &runs {
        let src = fs::read_to_string(path).expect("ptx readable");
        let name = path.display().to_string();
        let report = analyze(&src, &name, opts).expect("analyzes");
        assert_eq!(report_violations(&report), Vec::<String>::new(), "{name}");
    }
}

#[test]
fn each_json_report_identity_names_its_violation() {
    let fresh =
        || analyze(&read("k5/k5.sm_80.ptx"), "k5", &AnalyzeOptions::default()).expect("analyzes");
    assert_eq!(report_violations(&fresh()), Vec::<String>::new());
    let named = |identity: &str, doctor: &dyn Fn(&mut Report)| {
        let mut report = fresh();
        doctor(&mut report);
        let violations = report_violations(&report);
        assert!(
            violations.iter().any(|v| v.starts_with(identity)),
            "{identity}: {violations:?}"
        );
    };
    named("instruction-accounting", &|r| {
        r.kernels[0].instruction_classes.flop += 1
    });
    named("unparsed-surfaced", &|r| {
        r.kernels[0].instruction_classes.unparsed += 1
    });
    named("classification-coverage", &|r| {
        r.coverage.get_mut("instructions_classified").unwrap().num -= 1
    });
    named("trip-coverage", &|r| {
        r.kernels[0].loops[0].trips.expr = None
    });
    named("ai-consistency", &|r| {
        r.kernels[0].loops[0]
            .per_iteration
            .ai_global
            .as_mut()
            .unwrap()
            .value += 1.0
    });
    named("instruction-kinds-sum", &|r| {
        r.kernels[0].loops[0].per_iteration.instructions.total.expr += "0"
    });
    named("opcodes-sum", &|r| {
        let kind = r.kernels[0].loops[0]
            .per_iteration
            .instructions
            .by_kind
            .values_mut()
            .next()
            .unwrap();
        kind.opcodes.values_mut().next().unwrap().expr += "0";
    });
}
