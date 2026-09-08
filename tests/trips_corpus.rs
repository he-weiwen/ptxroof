//! Pinned trip counts for the corpus: the ladder's
//! real nvcc shapes and the micro fixtures' honest unknowns. Every
//! unknown must carry a reason string — these are pinned too, because
//! they are user-facing output.

use ptxroof::cfg::{build_cfg, loop_forest, loop_names};
use ptxroof::parse::parser::parse;
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
