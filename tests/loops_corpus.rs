//! Pinned loop forests for the corpus: headers by
//! label, nesting depths, and the honest irreducible flag for
//! micro/irreducible. The k5 tree is also an insta snapshot — the
//! human-reviewable form.

use ptxroof::analysis::control_flow::{Cfg, LoopForest, build_cfg, loop_forest};
use ptxroof::ptx::ir::Module;
use ptxroof::ptx::parse::parser::parse;
use std::fs;
use std::path::PathBuf;

fn forest_of(fixture: &str) -> (Module, Cfg, LoopForest) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let src = fs::read_to_string(&path).expect("fixture readable");
    let m = parse(&src).expect("fixture parses");
    let cfg = build_cfg(&m, &m.kernels[0]);
    let f = loop_forest(&cfg);
    (m, cfg, f)
}

/// (header label, depth) per loop, sorted by header block id.
fn shape(m: &Module, cfg: &Cfg, f: &LoopForest) -> Vec<(String, u32)> {
    let mut v: Vec<_> = f
        .loops
        .iter()
        .map(|l| {
            let label = cfg
                .block(l.header)
                .label
                .map(|s| m.interner.resolve(s).to_owned())
                .unwrap_or_else(|| format!("block{}", l.header.0));
            (label, l.depth)
        })
        .collect();
    v.sort();
    v
}

/// (fixture, [(header label, depth)...], irreducible edge count)
type ForestRow = (&'static str, &'static [(&'static str, u32)], usize);

#[test]
fn ladder_and_micro_loop_forests_are_pinned() {
    let expected: &[ForestRow] = &[
        ("k1/k1.sm_80.ptx", &[("$L__BB0_4", 1), ("$L__BB0_7", 1)], 0),
        ("k2/k2.sm_80.ptx", &[("$L__BB0_4", 1), ("$L__BB0_7", 1)], 0),
        ("k5/k5.sm_80.ptx", &[("$L__BB0_2", 1), ("$L__BB0_3", 2)], 0),
        ("micro/single_loop.ptx", &[("$L__BB0_2", 1)], 0),
        ("micro/branchy.ptx", &[("$L__BB0_2", 1)], 0),
        ("micro/irreducible.ptx", &[], 1),
        ("micro/no_loc.ptx", &[("$L__BB0_2", 1)], 0),
        ("micro/data_dep.ptx", &[("$L__BB0_1", 1)], 0),
    ];
    for (fixture, loops, irr) in expected {
        let (m, cfg, f) = forest_of(fixture);
        let got = shape(&m, &cfg, &f);
        let want: Vec<(String, u32)> = loops.iter().map(|(l, d)| (l.to_string(), *d)).collect();
        assert_eq!(got, want, "{fixture} loop forest");
        assert_eq!(f.irreducible_edges.len(), *irr, "{fixture} irreducibility");
    }
}

#[test]
fn k5_inner_loop_nests_inside_outer() {
    let (_, _, f) = forest_of("k5/k5.sm_80.ptx");
    let inner = f
        .loops
        .iter()
        .position(|l| l.depth == 2)
        .expect("inner loop");
    let outer = f.loops[inner].parent.expect("inner has parent");
    assert_eq!(f.get(outer).depth, 1);
    assert_eq!(f.get(outer).children.len(), 1);
    // The dot loop is a single block; the tile loop wraps it plus the
    // copy block and the latch block.
    assert_eq!(f.loops[inner].blocks.len(), 1);
    assert!(f.get(outer).blocks.len() > 2);
}

#[test]
fn k5_loop_tree_snapshot() {
    let (m, cfg, f) = forest_of("k5/k5.sm_80.ptx");
    let mut out = String::new();
    fn walk(
        m: &Module,
        cfg: &Cfg,
        f: &LoopForest,
        id: ptxroof::analysis::control_flow::LoopId,
        out: &mut String,
    ) {
        let l = f.get(id);
        let label = cfg
            .block(l.header)
            .label
            .map(|s| m.interner.resolve(s).to_owned())
            .unwrap_or_default();
        out.push_str(&format!(
            "{}loop {} blocks={} latches={}\n",
            "  ".repeat(l.depth as usize - 1),
            label,
            l.blocks.len(),
            l.latches.len()
        ));
        for c in f.children_of(id) {
            walk(m, cfg, f, c, out);
        }
    }
    for top in f.top_level() {
        walk(&m, &cfg, &f, top, &mut out);
    }
    insta::assert_snapshot!(out);
}
