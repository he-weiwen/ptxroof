//! Per-block measurement collection.
//!
//! Walks every block's instructions through the classifier and emits
//! the measurement stream, the block's instructions and its execution
//! qualifier.
//!
//! Within its innermost scope — the containing loop, or the kernel
//! for top-level blocks — a block's
//! execution count is **exact** iff the block dominates every latch of
//! that loop (it runs every iteration), respectively every exit block
//! of the kernel (it runs every invocation). Everything else is
//! **at_most**: a bounds-guarded epilogue, a data-dependent
//! conditional inside a loop body, an unreachable block. This is
//! deliberately conservative — nvcc's loop guards mirror the
//! zero-trip case of their latch expressions, and recognizing that
//! would require guard-implication analysis (outside the scope in PLAN.md).
//! The qualifier retains the upper bound.

use crate::analysis::affine::affine_analysis::AffineAnalysis;
use crate::analysis::control_flow::loops::LoopForest;
use crate::analysis::instruction_counts::classify::classify;
use crate::analysis::instruction_counts::measurement::Measurement;
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::Stmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountQualifier {
    Exact,
    AtMost,
}

#[derive(Debug)]
pub struct BlockMeasurements {
    pub block: BlockId,
    pub qualifier: CountQualifier,
    /// The block's contributions, in statement order.
    pub measurements: Vec<Measurement>,
    /// Statement indices of the instructions issued per execution of the
    /// block, each once even when it contributes several measurements or
    /// none. A predicated instruction is issued whether or not its
    /// predicate holds.
    pub instructions: Vec<usize>,
}

/// Collect measurements for every block; indexed by `BlockId`.
pub(crate) fn collect(affine: &AffineAnalysis) -> Vec<BlockMeasurements> {
    let KernelAnalyses {
        kernel,
        cfg,
        forest,
        ..
    } = *affine.data;
    let exit_blocks: Vec<BlockId> = (0..cfg.blocks.len() as u32)
        .map(BlockId)
        .filter(|&b| cfg.block(b).succs.is_empty())
        .collect();

    (0..cfg.blocks.len() as u32)
        .map(BlockId)
        .map(|bid| {
            let qualifier = block_qualifier(forest, &exit_blocks, bid);
            let mut measurements = Vec::new();
            let mut instructions = Vec::new();
            let b = cfg.block(bid);
            for (si, stmt) in kernel.stmts[b.start..b.end].iter().enumerate() {
                let Stmt::Instr(_) = stmt else { continue };
                let provenance = b.start + si;
                let contributions = classify(affine, provenance).into_iter();
                measurements.extend(contributions.map(|c| Measurement {
                    kind: c.kind,
                    count: c.count,
                    provenance,
                    memory_operand: c.memory_operand,
                }));
                instructions.push(provenance);
            }
            BlockMeasurements {
                block: bid,
                qualifier,
                measurements,
                instructions,
            }
        })
        .collect()
}

fn block_qualifier(forest: &LoopForest, exit_blocks: &[BlockId], block: BlockId) -> CountQualifier {
    if !forest.doms.is_reachable(block) {
        return CountQualifier::AtMost; // executes zero times; ≤ is honest
    }
    let dominated_targets: &[BlockId] = match forest.block_loop[block.0 as usize] {
        Some(l) => &forest.get(l).latches,
        None => exit_blocks,
    };
    let all = !dominated_targets.is_empty()
        && dominated_targets
            .iter()
            .all(|&t| forest.doms.dominates(block, t));
    if all {
        CountQualifier::Exact
    } else {
        CountQualifier::AtMost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::analysis::instruction_counts::measurement::Direction;
    use crate::analysis::instruction_counts::measurement::MeasureKind;
    use crate::analysis::instruction_counts::measurement::{Pipe, Precision};
    use crate::analysis::uniformity_info::UniformityInfo;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::modifier::Space;
    use crate::ptx::parse::parser::parse;

    fn collect_body(body: &str) -> (Vec<BlockMeasurements>, Module) {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let m = parse(&src).expect("test body parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let data = KernelAnalyses::new(&m, k, &cfg, &forest);
        let uniformity = UniformityInfo::new(&data);
        let blocks = collect(&AffineAnalysis::new(&data, &uniformity, None));
        (blocks, m)
    }
    use crate::ptx::ir::Module;

    #[test]
    fn conditional_block_inside_loop_is_at_most_loop_spine_is_exact() {
        // The branchy shape: header(0:pre) loop{header=1, cond=2, latch=3} exit=4
        let (blocks, _) = collect_body(
            "mov.u32 %r1, 0;\n\
             $L__H:\nld.global.f32 %f1, [%rd1];\n\
             setp.lt.f32 %p2, %f1, 0f00000000;\n@%p2 bra $L__S;\n\
             fma.rn.f32 %f1, %f1, %f1, %f1;\n\
             $L__S:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__H;\n\
             ret;",
        );
        assert_eq!(blocks[1].qualifier, CountQualifier::Exact, "loop header");
        assert_eq!(
            blocks[2].qualifier,
            CountQualifier::AtMost,
            "conditional body"
        );
        assert_eq!(blocks[3].qualifier, CountQualifier::Exact, "latch");
    }

    #[test]
    fn loop_only_nesting_stays_exact() {
        let (blocks, _) = collect_body(
            "mov.u32 %r1, 0;\n\
             $L__OUT:\nmov.u32 %r2, 0;\n\
             $L__IN:\nfma.rn.f32 %f1, %f1, %f1, %f1;\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r4;\n@%p1 bra $L__IN;\n\
             add.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p2, %r1, %r5;\n@%p2 bra $L__OUT;\n\
             ret;",
        );
        for b in &blocks {
            assert_eq!(b.qualifier, CountQualifier::Exact, "block {:?}", b.block);
        }
    }

    #[test]
    fn guarded_epilogue_is_at_most_final_ret_is_exact() {
        // entry -> (skip | body) -> ret: the body must be ≤, the ret =.
        let (blocks, _) = collect_body(
            "setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__END;\n\
             st.global.f32 [%rd1], %f1;\n\
             $L__END:\nret;",
        );
        assert_eq!(blocks[0].qualifier, CountQualifier::Exact, "entry");
        assert_eq!(blocks[1].qualifier, CountQualifier::AtMost, "guarded store");
        assert_eq!(blocks[2].qualifier, CountQualifier::Exact, "ret block");
    }

    #[test]
    fn a_copy_is_one_memory_instruction_with_two_byte_records() {
        let (blocks, _) = collect_body("cp.async.cg.shared.global [%r1], [%rd1], 16, 16;\nret;");
        assert_eq!(blocks[0].instructions, [0, 1]);
        let bytes: Vec<_> = blocks[0]
            .measurements
            .iter()
            .filter_map(|m| match m.kind {
                MeasureKind::Bytes { space, direction } => Some((space, direction, m.count)),
                _ => None,
            })
            .collect();
        assert_eq!(
            bytes,
            [
                (Space::Global, Direction::Load, 16),
                (Space::Shared, Direction::Store, 16)
            ]
        );
    }

    #[test]
    fn unquantified_bytes_never_become_zero() {
        let (blocks, m) = collect_body("ld.global %r1, [%rd1];\nret;");
        let has_unquantified = blocks[0]
            .measurements
            .iter()
            .any(|x| matches!(x.kind, MeasureKind::UnquantifiedBytes { .. } if x.count == 1));
        assert!(has_unquantified, "{:?}", blocks[0].measurements);
        drop(m);
    }

    /// A predicated atom is one memory instruction: collection records its
    /// atomic FLOP and its 4 B loaded and stored, each predicated.
    #[test]
    fn a_predicated_atom_is_one_instruction_with_three_contributions() {
        let (blocks, _) = collect_body("@%p1 atom.global.add.f32 %f1, [output], %f2;\nret;");
        assert_eq!(blocks[0].instructions, [0, 1]);
        let contributions: Vec<_> = blocks[0]
            .measurements
            .iter()
            .filter(|m| m.provenance == 0)
            .collect();
        let mut kinds: Vec<_> = contributions.iter().map(|m| (m.kind, m.count)).collect();
        kinds.sort();
        assert_eq!(
            kinds,
            [
                (
                    MeasureKind::Flops {
                        pipe: Pipe::Atomic,
                        precision: Precision::F32
                    },
                    1
                ),
                (
                    MeasureKind::Bytes {
                        space: Space::Global,
                        direction: Direction::Load
                    },
                    4
                ),
                (
                    MeasureKind::Bytes {
                        space: Space::Global,
                        direction: Direction::Store
                    },
                    4
                ),
            ]
        );
    }
}

#[cfg(test)]
mod ladder {
    //! Pinned per-block flop/byte numbers for the ladder: every expected
    //! value is hand-computed directly from the committed PTX and kept as a
    //! comment beside its assertion.

    use crate::analysis::affine::affine_analysis::AffineAnalysis;
    use crate::analysis::control_flow::loop_forest;
    use crate::analysis::instruction_counts::collect::{
        BlockMeasurements, CountQualifier, collect,
    };
    use crate::analysis::instruction_counts::measurement::MeasureKind;
    use crate::analysis::instruction_counts::measurement::{Direction, Precision};
    use crate::analysis::kernel_analyses::KernelAnalyses;
    use crate::analysis::uniformity_info::UniformityInfo;
    use crate::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
    use crate::ptx::ir::Module;
    use crate::ptx::modifier::Space;
    use crate::ptx::parse::parser::parse;
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
        let data = KernelAnalyses::new(&module, kernel, &cfg, &forest);
        let uniformity = UniformityInfo::new(&data);
        let blocks = collect(&AffineAnalysis::new(&data, &uniformity, None));
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

    fn unquantified_memory_ops(blocks: &[BlockMeasurements], ids: &[BlockId]) -> Tally {
        tally(blocks, ids, |k| {
            matches!(k, MeasureKind::UnquantifiedBytes { .. })
        })
    }

    fn unknown_ops(blocks: &[BlockMeasurements], ids: &[BlockId]) -> Tally {
        tally(blocks, ids, |k| matches!(k, MeasureKind::UnknownOps))
    }

    #[test]
    fn k2_per_block_numbers_match_hand_computation() {
        let f = load("k2/k2.sm_80.ptx");

        // Main loop $L__BB0_4 (x4-unrolled): 4 fma.rn.f32 = 8 flops;
        // 8 ld.global.u16 = 16 B.
        let main = [block_by_label(&f, "$L__BB0_4")];
        let mflops = flops(&f.blocks, &main, Some(Precision::F32));
        assert_eq!((mflops.value, mflops.ops), (8, 4));
        let loads = bytes(&f.blocks, &main, Some(Space::Global), Some(Direction::Load));
        assert_eq!((loads.value, loads.ops), (16, 8));
        assert_eq!(
            f.blocks[main[0].0 as usize].qualifier,
            CountQualifier::Exact,
            "loop spine is exact"
        );
        // No f16 compute anywhere in this kernel (S8's claim, block level).
        assert_eq!(flops(&f.blocks, &main, Some(Precision::F16)).value, 0);

        // Remainder loop $L__BB0_7: 1 fma = 2 flops; 2 loads = 4 B.
        let rem = [block_by_label(&f, "$L__BB0_7")];
        assert_eq!(flops(&f.blocks, &rem, Some(Precision::F32)).value, 2);
        assert_eq!(
            bytes(&f.blocks, &rem, Some(Space::Global), Some(Direction::Load)).value,
            4
        );

        // Epilogue $L__BB0_8: mul.f32 (1) + fma (2) = 3 flops; C readback
        // 2 B load + 2 B store. Guarded by the bounds check, so ≤.
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
        // = 32 B.
        let dot = [block_by_label(&f, "$L__BB0_3")];
        let dflops = flops(&f.blocks, &dot, Some(Precision::F32));
        assert_eq!((dflops.value, dflops.ops), (128, 64));
        assert_eq!(
            bytes(&f.blocks, &dot, Some(Space::Shared), Some(Direction::Load)).value,
            32
        );

        // Tile-copy block $L__BB0_2: 16 ld.global.u16 = 32 B in,
        // 16 st.shared.u16 = 32 B staged.
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

        // Whole kernel: zero unknown instructions, zero unquantified ops.
        let all = all_blocks(&f.blocks);
        assert_eq!(unknown_ops(&f.blocks, &all).ops, 0);
        assert_eq!(unquantified_memory_ops(&f.blocks, &all).ops, 0);
    }
}
