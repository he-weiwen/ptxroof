//! Pinned per-block flop/byte numbers for the ladder: every expected
//! value is hand-computed directly from the committed PTX and kept as a
//! comment beside its assertion.

use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::classify::{Direction, Precision, Space};
use ptxroof::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier, collect};
use ptxroof::analysis::instruction_counts::stats;
use ptxroof::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
use ptxroof::ptx::ir::Module;
use ptxroof::ptx::parse::parser::parse;
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

#[test]
fn k2_per_block_numbers_match_hand_computation() {
    let f = load("k2/k2.sm_80.ptx");

    // Main loop $L__BB0_4 (x4-unrolled): 4 fma.rn.f32 = 8 flops;
    // 8 ld.global.u16 = 16 B; 8 inline-asm cvt.f32.f16.
    let main = [block_by_label(&f, "$L__BB0_4")];
    let flops = stats::flops(&f.blocks, &main, Some(Precision::F32));
    assert_eq!((flops.value, flops.ops), (8, 4));
    let loads = stats::bytes(&f.blocks, &main, Some(Space::Global), Some(Direction::Load));
    assert_eq!((loads.value, loads.ops), (16, 8));
    assert_eq!(stats::conversions(&f.blocks, &main).ops, 8);
    assert_eq!(
        flops.qualifier,
        CountQualifier::Exact,
        "loop spine is exact"
    );
    // No f16 compute anywhere in this kernel (S8's claim, block level).
    assert_eq!(
        stats::flops(&f.blocks, &main, Some(Precision::F16)).value,
        0
    );

    // Remainder loop $L__BB0_7: 1 fma = 2 flops; 2 loads = 4 B; 2 cvt.
    let rem = [block_by_label(&f, "$L__BB0_7")];
    assert_eq!(stats::flops(&f.blocks, &rem, Some(Precision::F32)).value, 2);
    assert_eq!(
        stats::bytes(&f.blocks, &rem, Some(Space::Global), Some(Direction::Load)).value,
        4
    );
    assert_eq!(stats::conversions(&f.blocks, &rem).ops, 2);

    // Epilogue $L__BB0_8: mul.f32 (1) + fma (2) = 3 flops; C readback
    // 2 B load + 2 B store; 2 cvt. Guarded by the bounds check, so ≤.
    let epi = [block_by_label(&f, "$L__BB0_8")];
    let eflops = stats::flops(&f.blocks, &epi, Some(Precision::F32));
    assert_eq!((eflops.value, eflops.ops), (3, 2));
    assert_eq!(
        eflops.qualifier,
        CountQualifier::AtMost,
        "bounds-guarded epilogue"
    );
    assert_eq!(
        stats::bytes(&f.blocks, &epi, Some(Space::Global), Some(Direction::Load)).value,
        2
    );
    assert_eq!(
        stats::bytes(&f.blocks, &epi, Some(Space::Global), Some(Direction::Store)).value,
        2
    );
    assert_eq!(stats::conversions(&f.blocks, &epi).ops, 2);

    // Entry block: 8 ld.param = 3x4 (u32) + 2x4 (f32) + 3x8 (u64) = 44 B.
    let entry = [BlockId(0)];
    let params = stats::bytes(&f.blocks, &entry, Some(Space::Param), Some(Direction::Load));
    assert_eq!((params.value, params.ops), (44, 8));

    // Kernel-wide flat totals (un-multiplied by trips):
    // flops 8+2+3 = 13; global loads 16+4+2 = 22 B; stores 2 B.
    let all = stats::all_blocks(&f.blocks);
    assert_eq!(stats::flops(&f.blocks, &all, None).value, 13);
    assert_eq!(
        stats::bytes(&f.blocks, &all, Some(Space::Global), Some(Direction::Load)).value,
        22
    );
    assert_eq!(
        stats::bytes(&f.blocks, &all, Some(Space::Global), Some(Direction::Store)).value,
        2
    );
    assert_eq!(stats::unquantified_memory_ops(&f.blocks, &all).ops, 0);
    assert!(stats::unknown_ops(&f.blocks, &all).is_empty());
}

#[test]
fn k1_matches_k2_shape() {
    // Same kernel structure, different block geometry only.
    let f = load("k1/k1.sm_80.ptx");
    let main = [block_by_label(&f, "$L__BB0_4")];
    assert_eq!(
        stats::flops(&f.blocks, &main, Some(Precision::F32)).value,
        8
    );
    assert_eq!(
        stats::bytes(&f.blocks, &main, Some(Space::Global), Some(Direction::Load)).value,
        16
    );
    assert_eq!(
        stats::flops(&f.blocks, &stats::all_blocks(&f.blocks), None).value,
        13
    );
}

#[test]
fn k5_dot_loop_and_copy_block() {
    let f = load("k5/k5.sm_80.ptx");

    // Inner dot loop $L__BB0_3: 64 fma = 128 flops; 16 ld.shared.u16
    // = 32 B; 16 cvt.
    let dot = [block_by_label(&f, "$L__BB0_3")];
    let flops = stats::flops(&f.blocks, &dot, Some(Precision::F32));
    assert_eq!((flops.value, flops.ops), (128, 64));
    assert_eq!(
        stats::bytes(&f.blocks, &dot, Some(Space::Shared), Some(Direction::Load)).value,
        32
    );
    assert_eq!(stats::conversions(&f.blocks, &dot).ops, 16);

    // Tile-copy block $L__BB0_2: 16 ld.global.u16 = 32 B in,
    // 16 st.shared.u16 = 32 B staged, one bar.sync.
    let copy = [block_by_label(&f, "$L__BB0_2")];
    assert_eq!(
        stats::bytes(&f.blocks, &copy, Some(Space::Global), Some(Direction::Load)).value,
        32
    );
    assert_eq!(
        stats::bytes(
            &f.blocks,
            &copy,
            Some(Space::Shared),
            Some(Direction::Store)
        )
        .value,
        32
    );
    assert_eq!(stats::sync_ops(&f.blocks, &copy).ops, 1);

    // Whole kernel: zero unknown instructions, zero unquantified ops.
    let all = stats::all_blocks(&f.blocks);
    assert!(stats::unknown_ops(&f.blocks, &all).is_empty());
    assert_eq!(stats::unquantified_memory_ops(&f.blocks, &all).ops, 0);
}
