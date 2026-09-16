//! Per-block measurement collection.
//!
//! Walks every block's instructions through the classifier and emits
//! the measurement stream, plus per-block class tallies (the accounting
//! the `classified + allowlisted-unknown = total` verifier identity
//! reads) and the block's execution qualifier.
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

use crate::analysis::control_flow::loops::LoopForest;
use crate::analysis::control_flow::{BlockId, ControlFlowGraph};
use crate::analysis::instruction_counts::classify::{Direction, OpClass, classify};
use crate::analysis::instruction_counts::measurement::{MeasureKind, Measurement};
use crate::ptx::ir::{Kernel, Module, Stmt};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountQualifier {
    Exact,
    AtMost,
}

impl CountQualifier {
    pub fn and(self, other: CountQualifier) -> CountQualifier {
        if self == CountQualifier::Exact && other == CountQualifier::Exact {
            CountQualifier::Exact
        } else {
            CountQualifier::AtMost
        }
    }
}

/// Instruction-class tallies for one block; the accounting identity is
/// `flop + non_flop_arith + memory + sync + communication + control +
/// ignore + unknown == total`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClassCounts {
    pub total: u32,
    pub flop: u32,
    pub non_flop_arith: u32,
    pub memory: u32,
    pub sync: u32,
    pub communication: u32,
    pub control: u32,
    pub ignore: u32,
    pub unknown: u32,
    /// Statements the parser could not read; outside `total` and the
    /// identity, reported alongside the unknowns.
    pub unparsed: u32,
}

#[derive(Debug)]
pub struct BlockMeasurements {
    pub block: BlockId,
    pub qualifier: CountQualifier,
    pub measurements: Vec<Measurement>,
    pub class_counts: ClassCounts,
    /// Instructions issued per execution of the block, by class and
    /// opcode. A predicated instruction is issued whether or not its
    /// predicate holds, so it counts here in full.
    pub instructions: BTreeMap<(OpClass, String), u32>,
}

/// Collect measurements for every block; indexed by `BlockId`.
pub fn collect(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
) -> Vec<BlockMeasurements> {
    let exit_blocks: Vec<BlockId> = (0..cfg.blocks.len() as u32)
        .map(BlockId)
        .filter(|&b| cfg.block(b).succs.is_empty())
        .collect();

    (0..cfg.blocks.len() as u32)
        .map(BlockId)
        .map(|bid| {
            let qualifier = block_qualifier(forest, &exit_blocks, bid);
            let mut measurements = Vec::new();
            let mut counts = ClassCounts::default();
            let mut instructions: BTreeMap<(OpClass, String), u32> = BTreeMap::new();
            let b = cfg.block(bid);
            for (si, stmt) in kernel.stmts[b.start..b.end].iter().enumerate() {
                let Stmt::Instr(instr) = stmt else {
                    counts.unparsed += u32::from(matches!(stmt, Stmt::Unparsed { .. }));
                    continue;
                };
                let provenance = b.start + si;
                counts.total += 1;
                let predicated = instr.predicate.is_some();
                let mut push = |kind, count| {
                    measurements.push(Measurement {
                        kind,
                        count,
                        predicated,
                        provenance,
                    });
                };
                let mut push_bytes = |space, direction, bytes: Option<u32>| match bytes {
                    Some(n) => push(MeasureKind::Bytes { space, direction }, n as u64),
                    None => push(MeasureKind::UnquantifiedBytes { space, direction }, 1),
                };
                let class = classify(module, instr);
                *instructions
                    .entry((class, module.opcode(instr)))
                    .or_default() += 1;
                match class {
                    OpClass::Flop {
                        pipe,
                        precision,
                        flops,
                    } => {
                        counts.flop += 1;
                        push(MeasureKind::Flops { pipe, precision }, flops as u64);
                    }
                    OpClass::NonFlopArith { kind } => {
                        counts.non_flop_arith += 1;
                        if kind
                            == crate::analysis::instruction_counts::classify::ArithKind::Conversion
                        {
                            push(MeasureKind::Conversions, 1);
                        } else {
                            push(MeasureKind::NonFlopOps { kind }, 1);
                        }
                    }
                    OpClass::Memory {
                        space,
                        direction,
                        bytes,
                    } => {
                        counts.memory += 1;
                        push_bytes(space, direction, bytes);
                    }
                    OpClass::Copy {
                        from,
                        to,
                        read_bytes,
                        written_bytes,
                    } => {
                        counts.memory += 1;
                        push_bytes(from, Direction::Load, read_bytes);
                        push_bytes(to, Direction::Store, written_bytes);
                    }
                    OpClass::Sync => {
                        counts.sync += 1;
                        push(MeasureKind::SyncOps, 1);
                    }
                    OpClass::Communication => {
                        counts.communication += 1;
                        push(MeasureKind::CommunicationOps, 1);
                    }
                    OpClass::Control => {
                        counts.control += 1;
                        push(MeasureKind::ControlOps, 1);
                    }
                    OpClass::Ignore => {
                        counts.ignore += 1;
                    }
                    OpClass::Unknown => {
                        counts.unknown += 1;
                        push(
                            MeasureKind::UnknownOps {
                                mnemonic: instr.mnemonic,
                            },
                            1,
                        );
                    }
                }
            }
            BlockMeasurements {
                block: bid,
                qualifier,
                measurements,
                class_counts: counts,
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
    use crate::analysis::control_flow::{build_cfg, loop_forest};
    use crate::analysis::instruction_counts::classify::Space;
    use crate::ptx::parse::parser::parse;

    fn collect_body(body: &str) -> (Vec<BlockMeasurements>, Module) {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let m = parse(&src).expect("test body parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let blocks = collect(&m, k, &cfg, &f);
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
    fn predicated_instruction_is_marked() {
        let (blocks, _) = collect_body("@%p1 st.global.f32 [%rd1], %f1;\nret;");
        let m = &blocks[0].measurements[0];
        assert!(m.predicated);
    }

    #[test]
    fn class_counts_account_for_every_instruction() {
        let (blocks, _) = collect_body(
            "ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, 0;\n\
             fma.rn.f32 %f1, %f1, %f1, %f1;\ncvt.f32.f16 %f2, %rs1;\n\
             bar.sync 0;\nmma.sync.m8n8k4 %f1, %f2, %f3, %f4;\nret;",
        );
        let c = blocks[0].class_counts;
        assert_eq!(c.total, 7);
        assert_eq!(
            c.flop
                + c.non_flop_arith
                + c.memory
                + c.sync
                + c.communication
                + c.control
                + c.ignore
                + c.unknown,
            c.total,
            "accounting identity"
        );
        assert_eq!(c.unknown, 1); // the mma
    }

    #[test]
    fn a_copy_is_one_memory_instruction_with_two_byte_records() {
        let (blocks, _) = collect_body("cp.async.cg.shared.global [%r1], [%rd1], 16, 16;\nret;");
        let c = blocks[0].class_counts;
        assert_eq!((c.total, c.memory), (2, 1));
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
    fn unparsed_statements_are_counted_outside_the_identity() {
        let (blocks, _) = collect_body("@@ not ptx;\nfma.rn.f32 %f1, %f1, %f1, %f1;\nret;");
        let c = blocks[0].class_counts;
        assert_eq!((c.total, c.flop, c.control, c.unparsed), (2, 1, 1, 1));
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
}
