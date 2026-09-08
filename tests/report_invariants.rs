//! IR-level verifier identities  that need data the JSON
//! deliberately omits — the runner's JSON checks cover the rest.
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

use ptxroof::cfg::{build_cfg, loop_forest, loop_names};
use ptxroof::classify::Precision;
use ptxroof::core::Stmt;
use ptxroof::parse::parser::parse;
use ptxroof::report::{AnalyzeOptions, BindingSpec, Stats, analyze, collect};
use ptxroof::trips::trip_counts;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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
                nodes: &'a [ptxroof::report::tree::LoopNode],
                out: &mut Vec<&'a ptxroof::report::tree::LoopNode>,
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

#[test]
fn bound_flop_totals_agree_with_an_independent_evaluation() {
    // Fixtures with fully resolvable trips and one symbolic parameter.
    let cases: &[(&str, usize)] = &[
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
    for &(fixture, param) in cases {
        let src = read(fixture);
        let value = 4099; // deliberately not a multiple of the unrolls

        // Path 1: the report.
        let opts = AnalyzeOptions {
            bindings: vec![BindingSpec {
                index: Some(param),
                name: "N".into(),
                value,
            }],
            ..Default::default()
        };
        let report = analyze(&src, fixture, &opts).expect("analyzes");
        let totals = &report.kernels[0].totals;
        let table_total =
            |t: &std::collections::BTreeMap<String, ptxroof::report::tree::Count>| -> i64 {
                t["total"].expr.parse().expect("bound total is numeric")
            };
        let reported = table_total(&totals.flops)
            + table_total(&totals.tensor_flops)
            + table_total(&totals.sfu_flops);

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
        let stats = Stats::new(&blocks);
        let mut independent = 0i64;
        for bm in &blocks {
            let mut mult = 1i64;
            let mut cur = f.block_loop[bm.block.0 as usize];
            while let Some(l) = cur {
                mult *= trips_num[l.0 as usize];
                cur = f.get(l).parent;
            }
            let flat = stats.flops(&[bm.block], None).value as i64;
            independent += flat * mult;
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
