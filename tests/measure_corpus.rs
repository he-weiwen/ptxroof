//! Pinned per-block flop/byte numbers for the ladder: every expected
//! value is hand-computed directly from the committed PTX and kept as a
//! comment beside its assertion.

use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::classify::{Direction, Precision, Space};
use ptxroof::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier, collect};
use ptxroof::analysis::instruction_counts::measurement::MeasureKind;
use ptxroof::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
use ptxroof::ptx::ir::Module;
use ptxroof::ptx::parse::parser::parse;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

struct Fixture {
    module: Module,
    cfg: ControlFlowGraph,
    blocks: Vec<BlockMeasurements>,
}

fn load(fixture: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let src = fs::read_to_string(&path).expect("fixture readable");
    let module = parse(&src).expect("fixture parses");
    let kernel = &module.kernels[0];
    let cfg = build_cfg(&module, kernel);
    let forest = loop_forest(&cfg);
    let blocks = collect(&module, kernel, &cfg, &forest);
    Fixture {
        module,
        cfg,
        blocks,
    }
}

fn block_by_label(f: &Fixture, label: &str) -> BlockId {
    let sym = f.module.interner.get(label).expect("label interned");
    (0..f.cfg.blocks.len() as u32)
        .map(BlockId)
        .find(|&b| f.cfg.block(b).label == Some(sym))
        .expect("label names a block")
}

struct Tally {
    value: u64,
    ops: u64,
}

fn tally(
    blocks: &[BlockMeasurements],
    ids: &[BlockId],
    select: impl Fn(&MeasureKind) -> bool,
) -> Tally {
    let mut t = Tally { value: 0, ops: 0 };
    let mut contributors = BTreeSet::new();
    for &id in ids {
        for m in &blocks[id.0 as usize].measurements {
            if select(&m.kind) {
                t.value += m.count;
                t.ops += u64::from(contributors.insert(m.provenance));
            }
        }
    }
    t
}

fn all_blocks(blocks: &[BlockMeasurements]) -> Vec<BlockId> {
    blocks.iter().map(|b| b.block).collect()
}

fn flops(blocks: &[BlockMeasurements], ids: &[BlockId], precision: Option<Precision>) -> Tally {
    tally(blocks, ids, |k| match k {
        MeasureKind::Flops { precision: p, .. } => precision.is_none_or(|want| *p == want),
        _ => false,
    })
}

fn bytes(
    blocks: &[BlockMeasurements],
    ids: &[BlockId],
    space: Option<Space>,
    direction: Option<Direction>,
) -> Tally {
    tally(blocks, ids, |k| match k {
        MeasureKind::Bytes {
            space: s,
            direction: d,
        } => space.is_none_or(|want| *s == want) && direction.is_none_or(|want| *d == want),
        _ => false,
    })
}

fn conversions(blocks: &[BlockMeasurements], ids: &[BlockId]) -> Tally {
    tally(blocks, ids, |k| matches!(k, MeasureKind::Conversions))
}

fn sync_ops(blocks: &[BlockMeasurements], ids: &[BlockId]) -> Tally {
    tally(blocks, ids, |k| matches!(k, MeasureKind::SyncOps))
}

fn unquantified_memory_ops(blocks: &[BlockMeasurements], ids: &[BlockId]) -> Tally {
    tally(blocks, ids, |k| {
        matches!(k, MeasureKind::UnquantifiedBytes { .. })
    })
}

fn unknown_ops(blocks: &[BlockMeasurements], ids: &[BlockId]) -> Tally {
    tally(blocks, ids, |k| matches!(k, MeasureKind::UnknownOps { .. }))
}

#[test]
fn k2_per_block_numbers_match_hand_computation() {
    let f = load("k2/k2.sm_80.ptx");

    // Main loop $L__BB0_4 (x4-unrolled): 4 fma.rn.f32 = 8 flops;
    // 8 ld.global.u16 = 16 B; 8 inline-asm cvt.f32.f16.
    let main = [block_by_label(&f, "$L__BB0_4")];
    let mflops = flops(&f.blocks, &main, Some(Precision::F32));
    assert_eq!((mflops.value, mflops.ops), (8, 4));
    let loads = bytes(&f.blocks, &main, Some(Space::Global), Some(Direction::Load));
    assert_eq!((loads.value, loads.ops), (16, 8));
    assert_eq!(conversions(&f.blocks, &main).ops, 8);
    assert_eq!(
        f.blocks[main[0].0 as usize].qualifier,
        CountQualifier::Exact,
        "loop spine is exact"
    );
    // No f16 compute anywhere in this kernel (S8's claim, block level).
    assert_eq!(flops(&f.blocks, &main, Some(Precision::F16)).value, 0);

    // Remainder loop $L__BB0_7: 1 fma = 2 flops; 2 loads = 4 B; 2 cvt.
    let rem = [block_by_label(&f, "$L__BB0_7")];
    assert_eq!(flops(&f.blocks, &rem, Some(Precision::F32)).value, 2);
    assert_eq!(
        bytes(&f.blocks, &rem, Some(Space::Global), Some(Direction::Load)).value,
        4
    );
    assert_eq!(conversions(&f.blocks, &rem).ops, 2);

    // Epilogue $L__BB0_8: mul.f32 (1) + fma (2) = 3 flops; C readback
    // 2 B load + 2 B store; 2 cvt. Guarded by the bounds check, so ≤.
    let epi = [block_by_label(&f, "$L__BB0_8")];
    let eflops = flops(&f.blocks, &epi, Some(Precision::F32));
    assert_eq!((eflops.value, eflops.ops), (3, 2));
    assert_eq!(
        f.blocks[epi[0].0 as usize].qualifier,
        CountQualifier::AtMost,
        "bounds-guarded epilogue"
    );
    assert_eq!(
        bytes(&f.blocks, &epi, Some(Space::Global), Some(Direction::Load)).value,
        2
    );
    assert_eq!(
        bytes(&f.blocks, &epi, Some(Space::Global), Some(Direction::Store)).value,
        2
    );
    assert_eq!(conversions(&f.blocks, &epi).ops, 2);

    // Entry block: 8 ld.param = 3x4 (u32) + 2x4 (f32) + 3x8 (u64) = 44 B.
    let entry = [BlockId(0)];
    let params = bytes(&f.blocks, &entry, Some(Space::Param), Some(Direction::Load));
    assert_eq!((params.value, params.ops), (44, 8));

    // Kernel-wide flat totals (un-multiplied by trips):
    // flops 8+2+3 = 13; global loads 16+4+2 = 22 B; stores 2 B.
    let all = all_blocks(&f.blocks);
    assert_eq!(flops(&f.blocks, &all, None).value, 13);
    assert_eq!(
        bytes(&f.blocks, &all, Some(Space::Global), Some(Direction::Load)).value,
        22
    );
    assert_eq!(
        bytes(&f.blocks, &all, Some(Space::Global), Some(Direction::Store)).value,
        2
    );
    assert_eq!(unquantified_memory_ops(&f.blocks, &all).ops, 0);
    assert_eq!(unknown_ops(&f.blocks, &all).ops, 0);
}

#[test]
fn k1_matches_k2_shape() {
    // Same kernel structure, different block geometry only.
    let f = load("k1/k1.sm_80.ptx");
    let main = [block_by_label(&f, "$L__BB0_4")];
    assert_eq!(flops(&f.blocks, &main, Some(Precision::F32)).value, 8);
    assert_eq!(
        bytes(&f.blocks, &main, Some(Space::Global), Some(Direction::Load)).value,
        16
    );
    assert_eq!(flops(&f.blocks, &all_blocks(&f.blocks), None).value, 13);
}

#[test]
fn k5_dot_loop_and_copy_block() {
    let f = load("k5/k5.sm_80.ptx");

    // Inner dot loop $L__BB0_3: 64 fma = 128 flops; 16 ld.shared.u16
    // = 32 B; 16 cvt.
    let dot = [block_by_label(&f, "$L__BB0_3")];
    let dflops = flops(&f.blocks, &dot, Some(Precision::F32));
    assert_eq!((dflops.value, dflops.ops), (128, 64));
    assert_eq!(
        bytes(&f.blocks, &dot, Some(Space::Shared), Some(Direction::Load)).value,
        32
    );
    assert_eq!(conversions(&f.blocks, &dot).ops, 16);

    // Tile-copy block $L__BB0_2: 16 ld.global.u16 = 32 B in,
    // 16 st.shared.u16 = 32 B staged, one bar.sync.
    let copy = [block_by_label(&f, "$L__BB0_2")];
    assert_eq!(
        bytes(&f.blocks, &copy, Some(Space::Global), Some(Direction::Load)).value,
        32
    );
    assert_eq!(
        bytes(
            &f.blocks,
            &copy,
            Some(Space::Shared),
            Some(Direction::Store)
        )
        .value,
        32
    );
    assert_eq!(sync_ops(&f.blocks, &copy).ops, 1);

    // Whole kernel: zero unknown instructions, zero unquantified ops.
    let all = all_blocks(&f.blocks);
    assert_eq!(unknown_ops(&f.blocks, &all).ops, 0);
    assert_eq!(unquantified_memory_ops(&f.blocks, &all).ops, 0);
}
