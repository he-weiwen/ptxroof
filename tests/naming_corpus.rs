//! Pinned loop display names on the corpus: every
//! loop in the committed corpus gets a human-readable name — source line
//! where debug info exists, label where it doesn't.

use ptxroof::analysis::control_flow::{build_cfg, loop_forest};
use ptxroof::analysis::loop_names::loop_names;
use ptxroof::ptx::parse::parser::parse;
use std::fs;
use std::path::PathBuf;

fn names_of(fixture: &str) -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let src = fs::read_to_string(&path).expect("fixture readable");
    let m = parse(&src).expect("fixture parses");
    let k = &m.kernels[0];
    let cfg = build_cfg(&m, k);
    let f = loop_forest(&cfg);
    let mut names: Vec<String> = loop_names(&m, k, &cfg, &f)
        .into_iter()
        .map(|n| n.display)
        .collect();
    names.sort();
    names
}

#[test]
fn every_corpus_loop_has_a_human_readable_name() {
    let expected: &[(&str, &[&str])] = &[
        ("k1/k1.sm_80.ptx", &["1_naive.cuh:15", "1_naive.cuh:15"]),
        (
            "k2/k2.sm_80.ptx",
            &["2_coalesced.cuh:14", "2_coalesced.cuh:14"],
        ),
        (
            "k5/k5.sm_80.ptx",
            &["5_2d_blocktiling.cuh:39", "5_2d_blocktiling.cuh:53"],
        ),
        ("micro/single_loop.ptx", &["single_loop.cu:4"]),
        ("micro/branchy.ptx", &["branchy.cu:4"]),
        ("micro/no_loc.ptx", &["$L__BB0_2"]),
        ("micro/data_dep.ptx", &["data_dep.cu:4"]),
    ];
    for (fixture, want) in expected {
        assert_eq!(names_of(fixture), *want, "{fixture}");
    }
}
