//! Pinned trip counts for the corpus: the ladder's
//! real nvcc shapes and the micro fixtures' honest unknowns. Every
//! unknown must carry a reason string — these are pinned too, because
//! they are user-facing output.

use ptxroof::analysis::control_flow::{build_cfg, loop_forest};
use ptxroof::analysis::loop_names::loop_names;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::trips::{TripInfo, trip_counts};
use std::fs;
use std::path::PathBuf;

fn info_of(fixture: &str) -> (Vec<String>, TripInfo) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let src = fs::read_to_string(&path).expect("fixture readable");
    let m = parse(&src).expect("fixture parses");
    let k = &m.kernels[0];
    let cfg = build_cfg(&m, k);
    let f = loop_forest(&cfg);
    let names = loop_names(&m, k, &cfg, &f);
    let info = trip_counts(&m, k, &cfg, &f, &names);
    let rendered = info
        .trips
        .iter()
        .map(|t| match t {
            Ok(e) => format!("ok: {e}"),
            Err(r) => format!("unknown: {r}"),
        })
        .collect();
    (rendered, info)
}

#[test]
fn ladder_trip_counts_are_pinned() {
    // k2: K is param 2; main loop (K − K mod 4)/4, nounroll remainder
    // K mod 4, linked as one logical loop with factor 4.
    let (trips, info) = info_of("k2/k2.sm_80.ptx");
    assert_eq!(
        trips,
        ["ok: (param_2 - param_2 mod 4) / 4", "ok: param_2 mod 4"]
    );
    assert_eq!(info.unroll_pairs.len(), 1);
    assert_eq!(info.unroll_pairs[0].factor, 4);

    // k1 is the same kernel shape.
    let (trips, info) = info_of("k1/k1.sm_80.ptx");
    assert_eq!(
        trips,
        ["ok: (param_2 - param_2 mod 4) / 4", "ok: param_2 mod 4"]
    );
    assert_eq!(info.unroll_pairs.len(), 1);

    // k5: outer tile loop ceildiv(K, 8); inner dot loop a constant 8.
    let (trips, info) = info_of("k5/k5.sm_80.ptx");
    assert_eq!(trips, ["ok: ⌈param_2/8⌉", "ok: 8"]);
    assert!(info.unroll_pairs.is_empty(), "different lines, no pair");
}

#[test]
fn gluon_trip_counts_are_pinned() {
    // Both fp8 GEMMs: K tiles of 64, param 10 is K.
    let (trips, _) = info_of("gluon/fp8_gemm_kernel.c_fc.sm_89.ptx");
    assert_eq!(trips, ["ok: ⌈param_10/64⌉"]);
    let (trips, _) = info_of("gluon/fp8_gemm_kernel.lm_head_dx.sm_89.ptx");
    assert_eq!(trips, ["ok: ⌈param_10/64⌉"]);
    // Cross-entropy: V/BLOCK = 8 for both passes ...
    let (trips, _) = info_of("gluon/ce_chunk_kernel.sm_89.ptx");
    assert_eq!(trips, ["ok: 8", "ok: 8"]);
    // ... and at V = 2·BLOCK LLVM unrolls the first pass and turns the
    // second into a predicate-phi loop of two trips.
    let (trips, _) = info_of("gluon/ce_chunk_kernel.v8192.sm_89.ptx");
    assert_eq!(trips, ["ok: 2"]);
}

#[test]
fn two_register_counter_is_an_induction_variable() {
    // t = i + 1 in the header, `mov i, t` in the latch after the compare.
    let (trips, _) = info_of("micro/two_register_counter.ptx");
    assert_eq!(trips, ["ok: param_1"]);
}

#[test]
fn attention_trip_reasons_are_pinned() {
    // Causal bounds depend on the CTA index, behind `bfe`, `div` and `or`
    // the tracer does not read; the reason is the index.
    let (trips, _) = info_of("gluon/attn_bwd_kernel.sm_89.ptx");
    assert_eq!(
        trips,
        ["unknown: latch condition depends on special register %ctaid.x"]
    );
    for f in [
        "gluon/attn_bwd_pre_kernel.sm_89.ptx",
        "gluon/attn_bwd_post_kernel.sm_89.ptx",
    ] {
        assert!(info_of(f).0.is_empty(), "{f}: no loops");
    }
    let (trips, _) = info_of("gluon/attn_fwd_ws_kernel.sm_89.ptx");
    let mut reasons: Vec<&str> = trips.iter().map(String::as_str).collect();
    reasons.sort();
    reasons.dedup();
    assert_eq!(
        reasons,
        [
            "unknown: latch condition depends on special register %ctaid.x",
            "unknown: latch predicate is defined by `mbarrier.test_wait.parity.shared::cta.b64`, not a comparison",
            "unknown: loop exit is not at the latch",
            "unknown: loop has multiple latches",
        ]
    );
}

#[test]
fn a_triangular_nest_is_an_honest_unknown() {
    let (trips, _) = info_of("micro/triangular.ptx");
    let mut sorted = trips.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        [
            "ok: param_0",
            "unknown: latch condition depends on an enclosing loop's counter",
        ]
    );
}

#[test]
fn scoped_labels_are_separate_loops() {
    let (trips, _) = info_of("micro/scoped_labels.ptx");
    assert_eq!(trips, ["ok: param_0", "ok: param_1"]);
}

#[test]
fn micro_trip_counts_and_honest_unknowns() {
    let (trips, _) = info_of("micro/single_loop.ptx");
    assert_eq!(trips, ["ok: param_1"]);

    let (trips, _) = info_of("micro/branchy.ptx");
    assert_eq!(trips, ["ok: param_1"]);

    let (trips, _) = info_of("micro/no_loc.ptx");
    assert_eq!(trips, ["ok: param_1"]);

    // The honesty case: a pointer-chase latch has no static trip count
    // and the reason is named, never guessed (S9.1).
    let (trips, _) = info_of("micro/data_dep.ptx");
    assert_eq!(
        trips,
        ["unknown: latch condition depends on a value loaded inside the loop"]
    );
}
