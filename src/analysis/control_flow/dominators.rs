//! Dominator tree via Cooper–Harvey–Kennedy ("A Simple, Fast Dominance
//! Algorithm"), hand-rolled over the index arena.
//!
//! Unreachable blocks have no RPO number and no idom; every consumer
//! treats them as outside the analysis (they execute zero times).

use super::graph::{BlockId, Cfg};

#[derive(Debug)]
pub struct Dominators {
    /// Immediate dominator per block; `None` for the entry block and
    /// for unreachable blocks.
    pub idom: Vec<Option<BlockId>>,
    /// Reverse postorder over reachable blocks.
    pub rpo: Vec<BlockId>,
    /// RPO position per block; `usize::MAX` marks unreachable.
    pub rpo_index: Vec<usize>,
}

impl Dominators {
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
}

pub fn dominators(cfg: &Cfg) -> Dominators {
    let n = cfg.blocks.len();

    // Iterative DFS postorder from the entry, then reverse.
    let mut postorder = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    // Stack holds (block, next-successor-index).
    let mut stack: Vec<(BlockId, usize)> = vec![(Cfg::ENTRY, 0)];
    visited[Cfg::ENTRY.0 as usize] = true;
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
    idom[Cfg::ENTRY.0 as usize] = Some(Cfg::ENTRY); // self, during iteration
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

    idom[Cfg::ENTRY.0 as usize] = None; // entry has no idom in the result
    Dominators {
        idom,
        rpo,
        rpo_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::build_cfg;
    use crate::ptx::parse::parser::parse;

    fn doms_of(body: &str) -> (Cfg, Dominators) {
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
}
