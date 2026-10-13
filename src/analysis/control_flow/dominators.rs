//! Dominator tree via Cooper–Harvey–Kennedy ("A Simple, Fast Dominance
//! Algorithm"), hand-rolled over the index arena.
//!
//! Unreachable blocks have no RPO number and no idom; every consumer
//! treats them as outside the analysis (they execute zero times).

use crate::ptx::cfg::{BlockId, ControlFlowGraph};

#[derive(Debug)]
pub struct DominanceInfo {
    /// Immediate dominator per block; `None` for the entry block and
    /// for unreachable blocks.
    pub idom: Vec<Option<BlockId>>,
    /// Reverse postorder over reachable blocks.
    pub rpo: Vec<BlockId>,
    /// RPO position per block; `usize::MAX` marks unreachable.
    pub rpo_index: Vec<usize>,
}

impl DominanceInfo {
    pub fn is_reachable(&self, b: BlockId) -> bool {
        self.rpo_index[b.0 as usize] != usize::MAX
    }

    /// Does `a` dominate `b`? (Reflexive; false if either unreachable.)
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if !self.is_reachable(a) || !self.is_reachable(b) {
            return false;
        }
        let mut cur = b;
        loop {
            if cur == a {
                return true;
            }
            match self.idom[cur.0 as usize] {
                Some(next) => cur = next,
                None => return false,
            }
        }
    }

    /// Each block's dominance frontier: the blocks where its dominance
    /// ends (Cooper, Harvey and Kennedy, §4, Figure 5).
    pub fn frontiers(&self, cfg: &ControlFlowGraph) -> Vec<Vec<BlockId>> {
        let mut df: Vec<Vec<BlockId>> = vec![Vec::new(); cfg.blocks.len()];
        for &b in &self.rpo {
            let stop = self.idom[b.0 as usize];
            for &p in &cfg.block(b).preds {
                let mut runner = Some(p).filter(|&p| self.is_reachable(p));
                while let Some(r) = runner
                    && Some(r) != stop
                {
                    if df[r.0 as usize].last() != Some(&b) {
                        df[r.0 as usize].push(b);
                    }
                    runner = self.idom[r.0 as usize];
                }
            }
        }
        df
    }
}

/// The iterated dominance frontier of `blocks`, in block order: where
/// a value written in them meets another (Cytron, Ferrante, Rosen,
/// Wegman and Zadeck, "Efficiently Computing Static Single Assignment
/// Form and the Control Dependence Graph", TOPLAS 1991).
pub fn iterated_frontier(df: &[Vec<BlockId>], blocks: &[BlockId]) -> Vec<BlockId> {
    let mut queued = vec![false; df.len()];
    let mut placed = vec![false; df.len()];
    let mut work: Vec<BlockId> = Vec::new();
    for &b in blocks {
        if !std::mem::replace(&mut queued[b.0 as usize], true) {
            work.push(b);
        }
    }
    while let Some(x) = work.pop() {
        for &y in &df[x.0 as usize] {
            placed[y.0 as usize] = true;
            if !std::mem::replace(&mut queued[y.0 as usize], true) {
                work.push(y);
            }
        }
    }
    (0..df.len() as u32)
        .map(BlockId)
        .filter(|b| placed[b.0 as usize])
        .collect()
}

pub fn dominators(cfg: &ControlFlowGraph) -> DominanceInfo {
    let n = cfg.blocks.len();

    // Iterative DFS postorder from the entry, then reverse.
    let mut postorder = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    // Stack holds (block, next-successor-index).
    let mut stack: Vec<(BlockId, usize)> = vec![(ControlFlowGraph::ENTRY, 0)];
    visited[ControlFlowGraph::ENTRY.0 as usize] = true;
    while let Some(&mut (b, ref mut next)) = stack.last_mut() {
        let succs = &cfg.block(b).succs;
        if *next < succs.len() {
            let s = succs[*next];
            *next += 1;
            if !visited[s.0 as usize] {
                visited[s.0 as usize] = true;
                stack.push((s, 0));
            }
        } else {
            postorder.push(b);
            stack.pop();
        }
    }
    let rpo: Vec<BlockId> = postorder.into_iter().rev().collect();
    let mut rpo_index = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_index[b.0 as usize] = i;
    }

    // CHK iteration to fixpoint.
    let mut idom: Vec<Option<BlockId>> = vec![None; n];
    idom[ControlFlowGraph::ENTRY.0 as usize] = Some(ControlFlowGraph::ENTRY); // self, during iteration
    let intersect = |idom: &[Option<BlockId>], rpo_index: &[usize], a: BlockId, b: BlockId| {
        let (mut x, mut y) = (a, b);
        while x != y {
            while rpo_index[x.0 as usize] > rpo_index[y.0 as usize] {
                x = idom[x.0 as usize].expect("processed block has idom");
            }
            while rpo_index[y.0 as usize] > rpo_index[x.0 as usize] {
                y = idom[y.0 as usize].expect("processed block has idom");
            }
        }
        x
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new_idom: Option<BlockId> = None;
            for &p in &cfg.block(b).preds {
                if idom[p.0 as usize].is_none() {
                    continue; // unprocessed or unreachable
                }
                new_idom = Some(match new_idom {
                    None => p,
                    Some(cur) => intersect(&idom, &rpo_index, cur, p),
                });
            }
            if new_idom.is_some() && idom[b.0 as usize] != new_idom {
                idom[b.0 as usize] = new_idom;
                changed = true;
            }
        }
    }

    idom[ControlFlowGraph::ENTRY.0 as usize] = None; // entry has no idom in the result
    DominanceInfo {
        idom,
        rpo,
        rpo_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    fn doms_of(body: &str) -> (ControlFlowGraph, DominanceInfo) {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let cfg = build_cfg(&module, &module.kernels[0]);
        let d = dominators(&cfg);
        (cfg, d)
    }

    #[test]
    fn diamond_join_is_dominated_by_split_not_arms() {
        let (_, d) = doms_of(
            "setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__T;\n\
             add.f32 %f1, %f1, %f1;\nbra $L__J;\n\
             $L__T:\nmul.f32 %f1, %f1, %f1;\n\
             $L__J:\nret;",
        );
        // blocks: 0 split, 1 fall arm, 2 taken arm, 3 join
        assert_eq!(d.idom[3], Some(BlockId(0)));
        assert!(d.dominates(BlockId(0), BlockId(3)));
        assert!(!d.dominates(BlockId(1), BlockId(3)));
        assert!(!d.dominates(BlockId(2), BlockId(3)));
    }

    #[test]
    fn loop_header_dominates_latch_and_exit() {
        let (_, d) = doms_of(
            "mov.u32 %r1, 0;\n$L__L:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\nret;",
        );
        // blocks: 0 preheader, 1 loop (self-latch), 2 exit
        assert!(d.dominates(BlockId(1), BlockId(1)));
        assert_eq!(d.idom[1], Some(BlockId(0)));
        assert_eq!(d.idom[2], Some(BlockId(1)));
    }

    #[test]
    fn unreachable_block_has_no_idom_and_dominates_nothing() {
        let (_, d) = doms_of("bra $L__END;\nadd.f32 %f1, %f1, %f1;\n$L__END:\nret;");
        assert!(!d.is_reachable(BlockId(1)));
        assert_eq!(d.idom[1], None);
        assert!(!d.dominates(BlockId(1), BlockId(2)));
        assert!(d.dominates(BlockId(0), BlockId(2)));
    }

    #[test]
    fn irreducible_two_entry_cycle_no_cycle_member_dominates_the_other() {
        let (_, d) = doms_of(
            "setp.gt.s32 %p1, %r1, %r2;\n@%p1 bra $L__B;\n\
             $L__A:\nadd.s32 %r3, %r3, 1;\n\
             $L__B:\nadd.s32 %r3, %r3, 2;\n\
             setp.lt.s32 %p2, %r3, %r1;\n@%p2 bra $L__A;\nret;",
        );
        // blocks: 0 entry, 1 = A, 2 = B, 3 exit
        assert!(!d.dominates(BlockId(1), BlockId(2)));
        assert!(!d.dominates(BlockId(2), BlockId(1)));
        assert_eq!(d.idom[1], Some(BlockId(0))); // join of two paths is entry
        assert_eq!(d.idom[2], Some(BlockId(0)));
    }

    fn frontiers_of(body: &str) -> Vec<Vec<u32>> {
        let (cfg, d) = doms_of(body);
        (d.frontiers(&cfg).into_iter())
            .map(|f| f.into_iter().map(|b| b.0).collect())
            .collect()
    }

    #[test]
    fn diamond_arms_have_the_join_as_frontier() {
        let df = frontiers_of(
            "setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__T;\n\
             add.f32 %f1, %f1, %f1;\nbra $L__J;\n\
             $L__T:\nmul.f32 %f1, %f1, %f1;\n\
             $L__J:\nret;",
        );
        assert_eq!(df, [vec![], vec![3], vec![3], vec![]]);
    }

    #[test]
    fn loop_blocks_have_the_header_as_frontier() {
        // blocks: 0 preheader, 1 header, 2 body ending in the latch, 3 exit
        let df = frontiers_of(
            "mov.u32 %r1, 0;\n$L__H:\nsetp.ge.s32 %p1, %r1, %r2;\n@%p1 bra $L__X;\n\
             add.s32 %r1, %r1, 1;\nbra $L__H;\n$L__X:\nret;",
        );
        assert_eq!(df, [vec![], vec![1], vec![1], vec![]]);
    }

    #[test]
    fn nested_loop_body_has_both_headers_as_frontier() {
        // blocks: 0 preheader, 1 outer header, 2 inner loop, 3 outer latch, 4 exit
        let df = frontiers_of(
            "mov.u32 %r1, 0;\n$L__O:\nmov.u32 %r3, 0;\n\
             $L__I:\nadd.s32 %r3, %r3, 1;\nsetp.lt.s32 %p1, %r3, %r2;\n@%p1 bra $L__I;\n\
             add.s32 %r1, %r1, 1;\nsetp.lt.s32 %p2, %r1, %r2;\n@%p2 bra $L__O;\nret;",
        );
        assert_eq!(df, [vec![], vec![1], vec![1, 2], vec![1], vec![]]);
    }

    #[test]
    fn two_entry_cycle_members_have_each_other_as_frontier() {
        let df = frontiers_of(
            "setp.gt.s32 %p1, %r1, %r2;\n@%p1 bra $L__B;\n\
             $L__A:\nadd.s32 %r3, %r3, 1;\n\
             $L__B:\nadd.s32 %r3, %r3, 2;\n\
             setp.lt.s32 %p2, %r3, %r1;\n@%p2 bra $L__A;\nret;",
        );
        assert_eq!(df, [vec![], vec![2], vec![1], vec![]]);
    }

    #[test]
    fn entry_targeted_by_a_branch_is_its_own_frontier() {
        let df = frontiers_of(
            "$L__E:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__E;\nret;",
        );
        assert_eq!(df, [vec![0], vec![]]);
    }

    fn iterated_of(body: &str, blocks: &[u32]) -> Vec<u32> {
        let (cfg, d) = doms_of(body);
        let blocks: Vec<BlockId> = blocks.iter().map(|&b| BlockId(b)).collect();
        (iterated_frontier(&d.frontiers(&cfg), &blocks).into_iter())
            .map(|b| b.0)
            .collect()
    }

    #[test]
    fn iterated_frontier_of_a_diamond_arm_is_the_join() {
        let diamond = "setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__T;\n\
                       add.f32 %f1, %f1, %f1;\nbra $L__J;\n\
                       $L__T:\nmul.f32 %f1, %f1, %f1;\n\
                       $L__J:\nret;";
        assert_eq!(iterated_of(diamond, &[1]), [3]);
        assert_eq!(iterated_of(diamond, &[1, 2]), [3]);
        assert_eq!(iterated_of(diamond, &[0]), Vec::<u32>::new());
    }

    #[test]
    fn iterated_frontier_of_an_inner_loop_holds_both_headers() {
        // blocks: 0 preheader, 1 outer header, 2 inner loop, 3 outer latch, 4 exit
        let nested = "mov.u32 %r1, 0;\n$L__O:\nmov.u32 %r3, 0;\n\
                      $L__I:\nadd.s32 %r3, %r3, 1;\nsetp.lt.s32 %p1, %r3, %r2;\n@%p1 bra $L__I;\n\
                      add.s32 %r1, %r1, 1;\nsetp.lt.s32 %p2, %r1, %r2;\n@%p2 bra $L__O;\nret;";
        assert_eq!(iterated_of(nested, &[2]), [1, 2]);
        assert_eq!(iterated_of(nested, &[3]), [1]);
    }

    #[test]
    fn iterated_frontier_closes_over_a_two_entry_cycle() {
        let cycle = "setp.gt.s32 %p1, %r1, %r2;\n@%p1 bra $L__B;\n\
                     $L__A:\nadd.s32 %r3, %r3, 1;\n\
                     $L__B:\nadd.s32 %r3, %r3, 2;\n\
                     setp.lt.s32 %p2, %r3, %r1;\n@%p2 bra $L__A;\nret;";
        assert_eq!(iterated_of(cycle, &[1]), [1, 2]);
    }

    #[test]
    fn iterated_frontier_of_an_entry_targeted_by_a_branch_is_the_entry() {
        let looped =
            "$L__E:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__E;\nret;";
        assert_eq!(iterated_of(looped, &[0]), [0]);
    }

    #[test]
    fn unreachable_block_has_no_frontier_and_is_in_none() {
        let df = frontiers_of("bra $L__END;\nadd.f32 %f1, %f1, %f1;\n$L__END:\nret;");
        assert_eq!(df, [vec![], vec![], vec![]]);
    }
}
