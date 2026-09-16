//! Natural-loop forest + irreducible-region detection.
//!
//! Back edges are edges whose target dominates their source; each
//! header's natural loop is the union of reverse-reachable blocks from
//! its latches (multiple back edges to one header merge into one loop
//! — the `continue`-statement shape). Nesting comes from header
//! containment.
//!
//! A retreating edge that is *not* a back edge means an irreducible
//! region (a cycle with two entries, where no member dominates the
//! rest). Those are flagged `unknown-multiplicity` and never guessed
//! at: the edges are recorded here, and the report surfaces the blocks
//! as a named unknown.

use super::dominators::{Dominators, dominators};
use super::graph::{BlockId, Cfg};

/// Index into [`LoopForest::loops`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopId(pub u32);

#[derive(Debug)]
pub struct Loop {
    pub header: BlockId,
    /// Sources of the back edges into `header`.
    pub latches: Vec<BlockId>,
    /// All blocks of the loop, sorted by id; includes `header`.
    pub blocks: Vec<BlockId>,
    pub parent: Option<LoopId>,
    pub children: Vec<LoopId>,
    /// 1 = top-level.
    pub depth: u32,
}

#[derive(Debug)]
pub struct LoopForest {
    pub loops: Vec<Loop>,
    /// Innermost containing loop per block.
    pub block_loop: Vec<Option<LoopId>>,
    /// Retreating-but-not-back edges: each names an irreducible region.
    pub irreducible_edges: Vec<(BlockId, BlockId)>,
    pub doms: Dominators,
}

impl LoopForest {
    pub fn get(&self, id: LoopId) -> &Loop {
        &self.loops[id.0 as usize]
    }

    /// Top-level loops in program order (header block id).
    pub fn top_level(&self) -> Vec<LoopId> {
        let mut tops: Vec<LoopId> = (0..self.loops.len() as u32)
            .map(LoopId)
            .filter(|&l| self.get(l).parent.is_none())
            .collect();
        tops.sort_by_key(|&l| self.get(l).header);
        tops
    }

    /// Children of a loop in program order.
    pub fn children_of(&self, id: LoopId) -> Vec<LoopId> {
        let mut kids = self.get(id).children.clone();
        kids.sort_by_key(|&l| self.get(l).header);
        kids
    }
}

pub fn loop_forest(cfg: &Cfg) -> LoopForest {
    let doms = dominators(cfg);
    let n = cfg.blocks.len();

    // -- back edges, grouped by header ------------------------------------
    let mut header_latches: Vec<(BlockId, Vec<BlockId>)> = Vec::new();
    for b in (0..n as u32).map(BlockId) {
        if !doms.is_reachable(b) {
            continue;
        }
        for &s in &cfg.block(b).succs {
            if doms.dominates(s, b) {
                match header_latches.iter_mut().find(|(h, _)| *h == s) {
                    Some((_, latches)) => latches.push(b),
                    None => header_latches.push((s, vec![b])),
                }
            }
        }
    }
    header_latches.sort_by_key(|(h, _)| *h);

    // -- natural loop bodies ------------------------------------------------
    let mut loops: Vec<Loop> = Vec::new();
    for (header, latches) in header_latches {
        let mut in_loop = vec![false; n];
        in_loop[header.0 as usize] = true;
        let mut stack: Vec<BlockId> = Vec::new();
        for &l in &latches {
            if !in_loop[l.0 as usize] {
                in_loop[l.0 as usize] = true;
                stack.push(l);
            }
        }
        while let Some(b) = stack.pop() {
            for &p in &cfg.block(b).preds {
                if doms.is_reachable(p) && !in_loop[p.0 as usize] {
                    in_loop[p.0 as usize] = true;
                    stack.push(p);
                }
            }
        }
        let blocks: Vec<BlockId> = (0..n as u32)
            .map(BlockId)
            .filter(|b| in_loop[b.0 as usize])
            .collect();
        loops.push(Loop {
            header,
            latches,
            blocks,
            parent: None,
            children: Vec::new(),
            depth: 0,
        });
    }

    // -- nesting: parent = smallest strictly-larger loop containing the
    // header ------------------------------------------------------------
    let mut order: Vec<usize> = (0..loops.len()).collect();
    order.sort_by_key(|&i| loops[i].blocks.len());
    for (pos, &i) in order.iter().enumerate() {
        let header = loops[i].header;
        for &j in order.iter().skip(pos + 1) {
            if loops[j].blocks.binary_search(&header).is_ok() {
                loops[i].parent = Some(LoopId(j as u32));
                break;
            }
        }
    }
    for i in 0..loops.len() {
        if let Some(p) = loops[i].parent {
            loops[p.0 as usize].children.push(LoopId(i as u32));
        }
    }
    for i in 0..loops.len() {
        let mut depth = 1;
        let mut cur = loops[i].parent;
        while let Some(p) = cur {
            depth += 1;
            cur = loops[p.0 as usize].parent;
        }
        loops[i].depth = depth;
    }

    // -- innermost loop per block ------------------------------------------
    let mut block_loop: Vec<Option<LoopId>> = vec![None; n];
    for &i in &order {
        // ascending size: the first hit is the innermost
        for &b in &loops[i].blocks {
            if block_loop[b.0 as usize].is_none() {
                block_loop[b.0 as usize] = Some(LoopId(i as u32));
            }
        }
    }

    // -- irreducible regions: retreating edges that are not back edges ----
    // DFS with an explicit on-stack mark.
    let mut irreducible = Vec::new();
    let mut state = vec![0u8; n]; // 0 unvisited, 1 on stack, 2 done
    let mut stack: Vec<(BlockId, usize)> = vec![(Cfg::ENTRY, 0)];
    state[Cfg::ENTRY.0 as usize] = 1;
    while let Some(&mut (b, ref mut next)) = stack.last_mut() {
        let succs = &cfg.block(b).succs;
        if *next < succs.len() {
            let s = succs[*next];
            *next += 1;
            match state[s.0 as usize] {
                0 => {
                    state[s.0 as usize] = 1;
                    stack.push((s, 0));
                }
                // Retreating edge b -> s that is not a back edge.
                1 if !doms.dominates(s, b) => irreducible.push((b, s)),
                1 => {}
                _ => {}
            }
        } else {
            state[b.0 as usize] = 2;
            stack.pop();
        }
    }

    LoopForest {
        loops,
        block_loop,
        irreducible_edges: irreducible,
        doms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::build_cfg;
    use crate::ptx::parse::parser::parse;

    fn forest_of(body: &str) -> (Cfg, LoopForest) {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let cfg = build_cfg(&module, &module.kernels[0]);
        let f = loop_forest(&cfg);
        (cfg, f)
    }

    #[test]
    fn straight_line_has_no_loops() {
        let (_, f) = forest_of("add.f32 %f1, %f1, %f1;\nret;");
        assert!(f.loops.is_empty());
        assert!(f.irreducible_edges.is_empty());
    }

    #[test]
    fn single_self_loop() {
        let (_, f) = forest_of(
            "mov.u32 %r1, 0;\n$L__L:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(f.loops.len(), 1);
        let l = &f.loops[0];
        assert_eq!(l.header, BlockId(1));
        assert_eq!(l.latches, [BlockId(1)]);
        assert_eq!(l.blocks, [BlockId(1)]);
        assert_eq!(l.depth, 1);
    }

    #[test]
    fn two_latches_merge_into_one_loop() {
        // The `continue` shape: body branches back to the header from
        // two places.
        let (_, f) = forest_of(
            "mov.u32 %r1, 0;\n\
             $L__H:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__H;\n\
             setp.lt.s32 %p2, %r1, %r3;\n@%p2 bra $L__H;\n\
             ret;",
        );
        assert_eq!(f.loops.len(), 1);
        assert_eq!(f.loops[0].latches.len(), 2);
    }

    #[test]
    fn nested_loops_have_correct_depths() {
        let (_, f) = forest_of(
            "mov.u32 %r1, 0;\n\
             $L__OUT:\nmov.u32 %r2, 0;\n\
             $L__IN:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r4;\n@%p1 bra $L__IN;\n\
             add.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p2, %r1, %r5;\n@%p2 bra $L__OUT;\n\
             ret;",
        );
        assert_eq!(f.loops.len(), 2);
        let outer = f
            .loops
            .iter()
            .position(|l| l.depth == 1)
            .expect("outer loop");
        let inner = f
            .loops
            .iter()
            .position(|l| l.depth == 2)
            .expect("inner loop");
        assert_eq!(f.loops[inner].parent, Some(LoopId(outer as u32)));
        assert!(f.loops[outer].blocks.len() > f.loops[inner].blocks.len());
        // innermost-loop map: the inner header maps to the inner loop
        let ih = f.loops[inner].header;
        assert_eq!(f.block_loop[ih.0 as usize], Some(LoopId(inner as u32)));
    }

    #[test]
    fn classic_irreducible_two_entry_cycle_is_flagged_not_guessed() {
        let (_, f) = forest_of(
            "setp.gt.s32 %p1, %r1, %r2;\n@%p1 bra $L__B;\n\
             $L__A:\nadd.s32 %r3, %r3, 1;\n\
             $L__B:\nadd.s32 %r3, %r3, 2;\n\
             setp.lt.s32 %p2, %r3, %r1;\n@%p2 bra $L__A;\nret;",
        );
        assert!(f.loops.is_empty(), "no natural loop may be invented");
        assert_eq!(f.irreducible_edges.len(), 1);
        // Which retreating edge witnesses the region depends on DFS
        // order; either direction inside the {A, B} cycle is correct.
        let (src, dst) = f.irreducible_edges[0];
        let cycle = [BlockId(1), BlockId(2)];
        assert!(cycle.contains(&src) && cycle.contains(&dst) && src != dst);
    }

    #[test]
    fn reducible_loop_alongside_irreducible_region() {
        // An ordinary counted loop, then a two-entry cycle: the loop is
        // still found, the cycle is still flagged.
        let (_, f) = forest_of(
            "mov.u32 %r1, 0;\n$L__L:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\n\
             setp.gt.s32 %p3, %r1, %r2;\n@%p3 bra $L__B;\n\
             $L__A:\nadd.s32 %r3, %r3, 1;\n\
             $L__B:\nadd.s32 %r3, %r3, 2;\n\
             setp.lt.s32 %p2, %r3, %r1;\n@%p2 bra $L__A;\nret;",
        );
        assert_eq!(f.loops.len(), 1);
        assert_eq!(f.irreducible_edges.len(), 1);
    }
}
