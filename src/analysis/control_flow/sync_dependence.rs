//! Where the paths leaving a branch meet again: LLVM's sync dependence
//! (`GenericSyncDependenceAnalysis`,
//! llvm/include/llvm/ADT/GenericUniformityImpl.h:146) over the loop
//! forest. Each successor of the branch starts a label; a block that
//! one label reaches takes it, and one that two reach is a join and
//! starts its own. Labels follow back edges to a loop's header, and go
//! on from the header to the loop's exits only (`ModifiedPostOrder`,
//! :63, :701): a path around the loop again meets the others where it
//! leaves, and the paths into the loop from before the branch do not
//! count. Within one trip of a loop, the same labels show which
//! branches decide at which trip a thread leaves it.

use super::loops::{Loop, LoopForest};
use crate::ptx::cfg::{BlockId, ControlFlowGraph};

/// For each block, the joins of the paths from its successors. A path
/// back at the block stops, as its threads take the branch again. In a
/// kernel with a cycle of more than one entry, every block they reach.
pub fn join_blocks(cfg: &ControlFlowGraph, forest: &LoopForest) -> Vec<Vec<BlockId>> {
    let blocks = (0..cfg.blocks.len() as u32).map(BlockId);
    if !forest.irreducible_edges.is_empty() {
        let succs: Vec<_> = blocks.clone().map(|b| cfg.block(b).succs.clone()).collect();
        return blocks
            .map(|b| {
                let mut reached = order(&succs, &succs[b.0 as usize], None);
                reached.sort();
                reached
            })
            .collect();
    }
    let next = label_edges(cfg, forest);
    (blocks.map(|b| labels(&next, b, &cfg.block(b).succs).1)).collect()
}

/// For each loop, the branches in it that decide at which trip, or by
/// which exit, a thread leaves it: those whose successors' labels reach
/// its exits and its header, within one trip, not all alike. A path back
/// at the branch goes on, as the thread takes it again later in the same
/// trip. In a kernel with a cycle of more than one entry, every branch
/// in it.
pub fn exit_branches(cfg: &ControlFlowGraph, forest: &LoopForest) -> Vec<Vec<BlockId>> {
    let next = label_edges(cfg, forest);
    let branches = |l: &Loop| {
        let inside = |x: usize| {
            x != l.header.0 as usize && l.blocks.binary_search(&BlockId(x as u32)).is_ok()
        };
        let within: Vec<Vec<BlockId>> = (next.iter().enumerate())
            .map(|(x, n)| if inside(x) { n.clone() } else { Vec::new() })
            .collect();
        let decides = |b: BlockId| {
            let succs = &cfg.block(b).succs;
            if !forest.irreducible_edges.is_empty() {
                return succs.iter().any(|s| *s != succs[0]);
            }
            let label = labels(&within, l.header, succs).0;
            let mut left = (0..label.len())
                .filter(|&x| !inside(x))
                .filter_map(|x| label[x]);
            left.next().is_some_and(|first| left.any(|l| l != first))
        };
        (l.blocks.iter().copied()).filter(|&b| decides(b)).collect()
    };
    forest.loops.iter().map(branches).collect()
}

/// Each block's successors along which labels go: a loop header's are
/// the loop's exits.
fn label_edges(cfg: &ControlFlowGraph, forest: &LoopForest) -> Vec<Vec<BlockId>> {
    let blocks = (0..cfg.blocks.len() as u32).map(BlockId);
    let mut next: Vec<Vec<BlockId>> = blocks.map(|b| cfg.block(b).succs.clone()).collect();
    for l in &forest.loops {
        let mut exits: Vec<BlockId> = (l.blocks.iter())
            .flat_map(|&b| cfg.block(b).succs.iter().copied())
            .filter(|s| l.blocks.binary_search(s).is_err())
            .collect();
        exits.sort();
        exits.dedup();
        next[l.header.0 as usize] = exits;
    }
    next
}

/// The label of each block the paths from the successors `succs` reach
/// along `next` without passing `stop`, and their joins.
fn labels(
    next: &[Vec<BlockId>],
    stop: BlockId,
    succs: &[BlockId],
) -> (Vec<Option<BlockId>>, Vec<BlockId>) {
    let mut label: Vec<Option<BlockId>> = vec![None; next.len()];
    for &s in succs {
        label[s.0 as usize] = Some(s);
    }
    let mut joins = Vec::new();
    for x in order(next, succs, Some(stop)) {
        let pushed = label[x.0 as usize];
        for &s in &next[x.0 as usize] {
            let old = &mut label[s.0 as usize];
            if old.is_none() {
                *old = pushed;
            } else if *old != pushed {
                *old = Some(s);
                joins.push(s);
            }
        }
    }
    joins.sort();
    joins.dedup();
    (label, joins)
}

/// The blocks `roots` reach along `next` without passing `stop`, each
/// after every block that reaches it, when `next` has no cycle.
fn order(next: &[Vec<BlockId>], roots: &[BlockId], stop: Option<BlockId>) -> Vec<BlockId> {
    let mut seen = vec![false; next.len()];
    if let Some(b) = stop {
        seen[b.0 as usize] = true;
    }
    let mut post = Vec::new();
    for &r in roots {
        let mut stack = vec![(r, 0)];
        while let Some((b, i)) = stack.pop() {
            if i == 0 && std::mem::replace(&mut seen[b.0 as usize], true) {
                continue;
            }
            match next[b.0 as usize].get(i) {
                Some(&s) => stack.extend([(b, i + 1), (s, 0)]),
                None => post.push(b),
            }
        }
    }
    post.reverse();
    post
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    /// The joins of the branch ending the block labelled `branch`, by
    /// label; `%r1` holds the thread index.
    fn joins_of(body: &str, branch: &str) -> Vec<String> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\nmov.u32 %r1, %tid.x;\nbra.uni $B0;\n$B0:\n{body}\nret;\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let cfg = build_cfg(&module, &module.kernels[0]);
        let joins = join_blocks(&cfg, &loop_forest(&cfg));
        let at = (0..cfg.blocks.len() as u32)
            .map(BlockId)
            .find(|&b| cfg.block_name(&module, b) == branch)
            .expect("the branch's block");
        (joins[at.0 as usize].iter())
            .map(|&j| cfg.block_name(&module, j))
            .collect()
    }

    /// The exit branches of the loop headed by the block labelled
    /// `header`, by label.
    fn exit_branches_of(body: &str, header: &str) -> Vec<String> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\nmov.u32 %r1, %tid.x;\nbra.uni $B0;\n$B0:\n{body}\nret;\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let cfg = build_cfg(&module, &module.kernels[0]);
        let forest = loop_forest(&cfg);
        let l = (forest
            .loops
            .iter()
            .position(|l| cfg.block_name(&module, l.header) == header))
        .expect("the loop");
        (exit_branches(&cfg, &forest)[l].iter())
            .map(|&b| cfg.block_name(&module, b))
            .collect()
    }

    #[test]
    fn the_sides_of_an_if_meet_after_it() {
        let body = "setp.lt.u32 %p1, %r1, 16;\n@%p1 bra $A;\n$B:\nmov.u32 %r2, 64;\n\
                    bra.uni $J;\n$A:\nmov.u32 %r2, 0;\n$J:\nst.global.u32 [%rd1], %r2;";
        assert_eq!(joins_of(body, "$B0"), ["$J"]);
    }

    /// A spin-wait in a loop: every thread leaves it for the same block,
    /// and goes around the outer loop from there.
    #[test]
    fn a_spin_wait_meets_nothing() {
        let body = "$H:\nsetp.ge.u32 %p1, %r2, 4;\n@%p1 bra $X;\n\
                    $W:\nld.volatile.shared.u32 %r4, [%rd2];\nsetp.eq.u32 %p2, %r4, 0;\n\
                    @%p2 bra $W;\n$C:\nadd.s32 %r2, %r2, 1;\nbra.uni $H;\n$X:";
        assert_eq!(joins_of(body, "$W"), Vec::<String>::new());
    }

    /// Every thread leaves the spin-wait for the same block, so it
    /// decides no trip of the loop around it.
    #[test]
    fn a_spin_wait_is_no_exit_branch_of_the_loop_around_it() {
        let body = "$H:\nsetp.ge.u32 %p1, %r2, 4;\n@%p1 bra $X;\n\
                    $W:\nld.volatile.shared.u32 %r4, [%rd2];\nsetp.eq.u32 %p2, %r4, 0;\n\
                    @%p2 bra $W;\n$C:\nadd.s32 %r2, %r2, 1;\nbra.uni $H;\n$X:";
        assert_eq!(exit_branches_of(body, "$H"), ["$H"]);
        assert_eq!(exit_branches_of(body, "$W"), ["$W"]);
    }

    /// A header's branch that skips the only exit test: the threads it
    /// sends around leave at a later trip.
    #[test]
    fn a_header_branch_that_skips_the_exit_test_is_an_exit_branch() {
        let body = "$H:\nsetp.le.s32 %p1, %r2, %r1;\n@%p1 bra $L;\n\
                    $T:\nsetp.ge.s32 %p2, %r2, 3;\n@%p2 bra $X;\n\
                    $L:\nadd.s32 %r2, %r2, 1;\nbra.uni $H;\n$X:";
        assert_eq!(exit_branches_of(body, "$H"), ["$H", "$T"]);
    }

    /// An inner header's branch that sends some threads around the inner
    /// loop again, past the test that keeps them in the outer one: they
    /// leave the outer loop by the other exit.
    #[test]
    fn an_inner_header_branch_decides_the_outer_exit() {
        let body = "$H2:\nsetp.ge.u32 %p1, %r2, 3;\n@%p1 bra $M;\n$P:\nmov.u32 %r3, 0;\n\
                    $H1:\nsetp.le.s32 %p2, %r3, %r1;\n@%p2 bra $L1;\n\
                    $T:\nsetp.eq.s32 %p3, %r3, 4;\n@%p3 bra $E1;\n\
                    $T2:\nsetp.ge.s32 %p4, %r3, 8;\n@%p4 bra $M;\n\
                    $L1:\nadd.s32 %r3, %r3, 1;\nbra.uni $H1;\n\
                    $E1:\nadd.s32 %r2, %r2, 1;\nbra.uni $H2;\n$M:";
        assert_eq!(exit_branches_of(body, "$H2"), ["$H2", "$H1", "$T", "$T2"]);
    }

    /// Two latches: the threads on either side meet at the header.
    #[test]
    fn two_latches_meet_at_the_header() {
        let body = "$H:\nsetp.ge.u32 %p1, %r5, 2;\n@%p1 bra $X;\n\
                    $B:\nadd.s32 %r5, %r5, 1;\nsetp.lt.u32 %p2, %r1, 16;\n@%p2 bra $W;\n\
                    $I:\nbra.uni $H;\n$W:\nxor.b32 %r2, %r2, 1;\nbra.uni $H;\n$X:";
        assert_eq!(joins_of(body, "$B"), ["$H"]);
    }

    /// A latch's branch: the threads going around again leave through
    /// the other exit, and meet those leaving now after the loop.
    #[test]
    fn going_around_again_meets_after_the_loop() {
        let body = "$H:\nsetp.eq.s32 %p1, %r2, 2;\n@%p1 bra $A;\n\
                    $L:\nadd.s32 %r2, %r2, 1;\nsetp.ge.u32 %p2, %r1, 16;\n@%p2 bra $H;\n\
                    $F:\nmov.u32 %r3, 128;\nbra.uni $J;\n$A:\nmov.u32 %r3, 0;\n$J:";
        assert_eq!(joins_of(body, "$L"), ["$F", "$J"]);
    }

    /// A cycle entered at A or B is no loop of the forest: every block
    /// the branch reaches counts.
    #[test]
    fn with_a_cycle_of_two_entries_every_block_reached_is_a_join() {
        let body = "setp.lt.u32 %p1, %r1, 5;\n@%p1 bra $B;\n$A:\nadd.s32 %r2, %r2, 1;\n\
                    $B:\nadd.s32 %r2, %r2, 1;\nsetp.lt.u32 %p2, %r2, 9;\n@%p2 bra $A;\n$X:";
        assert_eq!(joins_of(body, "$B0"), ["$A", "$B", "$X"]);
    }
}
