//! Control-flow graph over index arenas.
//!
//! Blocks are an `IndexVec<BlockId, Block>`; edges are block ID lists.
//! Leaders are the kernel entry, branch-target labels, and statements
//! after branches/terminators; untargeted labels do not split blocks. Edges
//! come from `bra` (conditional via predicate, unconditional otherwise),
//! `brx.idx` through its `.branchtargets` table, and fallthrough.
//! `ret`/`exit` terminate; a *predicated* ret/exit falls through.
//! `call` does not end a block — the CFG is intra-procedural and call
//! sites are recorded so later stages can surface non-inlined callees
//! as a visible unknown (never silently ignored).
//!
//! Successor ordering convention (PR 11's latch analysis relies on it):
//! for a conditional branch `succs[0]` is the taken target and
//! `succs[1]` the fallthrough.

use crate::ptx::ir::{Instr, Kernel, Module, Operand, Stmt};
use crate::support::index::IndexVec;
use crate::support::index::newtype_idx;
use crate::support::intern::Symbol;
use std::collections::HashSet;

newtype_idx! {
    /// Index into [`Cfg::blocks`].
    pub struct BlockId;
}

#[derive(Debug)]
pub struct Block {
    /// Label starting this block, if any.
    pub label: Option<Symbol>,
    /// Statement index range into `Kernel::stmts`.
    pub start: usize,
    pub end: usize,
    pub succs: Vec<BlockId>,
    pub preds: Vec<BlockId>,
}

#[derive(Debug)]
pub struct Cfg {
    pub blocks: IndexVec<BlockId, Block>,
    /// Statement indices of `call` instructions — surfaced by the
    /// report as a visible unknown (non-inlined callee).
    pub call_sites: Vec<usize>,
    /// `bra` targets that matched no label captured in this kernel: the
    /// edge is dropped, so the CFG may be incomplete. Either the label
    /// is genuinely absent (malformed input) or the parser did not
    /// register it — and since compilers do not emit dangling branches,
    /// the latter is the likelier cause on real input. Surfaced as a
    /// report unknown; empty on the whole committed corpus.
    pub unresolved_branches: Vec<(BlockId, Symbol)>,
}

impl Cfg {
    pub const ENTRY: BlockId = BlockId(0);

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id]
    }

    /// Instructions of a block, in order.
    /// The block's label, or `<block N>` when it has none (the entry
    /// block and fallthrough blocks after a branch).
    pub fn block_name(&self, module: &Module, id: BlockId) -> String {
        self.block(id)
            .label
            .map(|s| module.interner.resolve(s).to_owned())
            .unwrap_or_else(|| format!("<block {}>", id.0))
    }

    pub fn instrs<'k>(&self, kernel: &'k Kernel, id: BlockId) -> impl Iterator<Item = &'k Instr> {
        let b = self.block(id);
        kernel.stmts[b.start..b.end].iter().filter_map(|s| match s {
            Stmt::Instr(i) => Some(i),
            _ => None,
        })
    }
}

pub fn build_cfg(module: &Module, kernel: &Kernel) -> Cfg {
    let sym = |text: &str| module.interner.get(text);
    let sym_bra = sym("bra");
    let sym_brx = sym("brx");
    let sym_ret = sym("ret");
    let sym_exit = sym("exit");
    let sym_call = sym("call");
    let is_branch = |i: &Instr| {
        let m = Some(i.mnemonic);
        m == sym_bra || m == sym_brx || m == sym_ret || m == sym_exit
    };

    // -- leaders ----------------------------------------------------------
    // A label starts a block only if a branch can reach it; LLVM's
    // `$L__tmpN` debug-range labels are not control flow.
    let mut targets: HashSet<Symbol> = HashSet::new();
    for stmt in &kernel.stmts {
        match stmt {
            Stmt::Instr(i) if Some(i.mnemonic) == sym_bra || Some(i.mnemonic) == sym_brx => {
                for &id in module.operand_ids(i.operands) {
                    if let Operand::SymbolRef(s) = module.operand(id) {
                        targets.insert(*s);
                    }
                }
            }
            Stmt::BranchTargets(ts) => targets.extend(ts.iter().copied()),
            _ => {}
        }
    }
    let n = kernel.stmts.len();
    let mut leader = vec![false; n + 1];
    if n > 0 {
        leader[0] = true;
    }
    for (i, stmt) in kernel.stmts.iter().enumerate() {
        match stmt {
            Stmt::Label(l) if targets.contains(l) => leader[i] = true,
            Stmt::Instr(instr) if is_branch(instr) => leader[i + 1] = true,
            _ => {}
        }
    }

    // -- blocks -------------------------------------------------------------
    let mut blocks: IndexVec<BlockId, Block> = IndexVec::new();
    let mut call_sites = Vec::new();
    let mut start = 0usize;
    for (i, &is_leader) in leader.iter().enumerate().skip(1) {
        if i == n || is_leader {
            // Labels after the final terminator are not a block.
            let has_instr = kernel.stmts[start..i]
                .iter()
                .any(|s| matches!(s, Stmt::Instr(_)));
            if i == n && !has_instr && !blocks.is_empty() {
                break;
            }
            let label = match kernel.stmts[start] {
                Stmt::Label(l) => Some(l),
                _ => None,
            };
            blocks.push(Block {
                label,
                start,
                end: i,
                succs: Vec::new(),
                preds: Vec::new(),
            });
            start = i;
        }
    }
    if blocks.is_empty() {
        blocks.push(Block {
            label: None,
            start: 0,
            end: 0,
            succs: Vec::new(),
            preds: Vec::new(),
        });
    }

    let label_to_block: std::collections::HashMap<Symbol, BlockId> = blocks
        .iter_enumerated()
        .filter_map(|(id, b)| b.label.map(|l| (l, id)))
        .collect();

    // -- edges --------------------------------------------------------------
    let mut unresolved = Vec::new();
    let mut edges: Vec<(BlockId, Vec<BlockId>)> = Vec::new();
    for (bid, block) in blocks.iter_enumerated() {
        let next = ((bid.0 as usize) + 1 < blocks.len()).then(|| BlockId(bid.0 + 1));
        let mut succs: Vec<BlockId> = Vec::new();

        // Record call sites while we're walking the statements.
        for (si, stmt) in kernel.stmts[block.start..block.end].iter().enumerate() {
            if let Stmt::Instr(instr) = stmt
                && Some(instr.mnemonic) == sym_call
            {
                call_sites.push(block.start + si);
            }
        }

        let last_instr = kernel.stmts[block.start..block.end]
            .iter()
            .rev()
            .find_map(|s| match s {
                Stmt::Instr(i) => Some(i),
                _ => None,
            });

        match last_instr {
            Some(i) if Some(i.mnemonic) == sym_bra => {
                let target = branch_target(module, kernel, i, &label_to_block);
                match target {
                    Ok(t) => succs.push(t),
                    Err(name) => unresolved.push((bid, name)),
                }
                if i.predicate.is_some()
                    && let Some(next) = next
                {
                    succs.push(next); // fallthrough after untaken branch
                }
            }
            Some(i) if Some(i.mnemonic) == sym_brx => {
                // `brx.idx %r, TBL;` — successors are the .branchtargets
                // list bound to label TBL.
                let mut resolved = false;
                if let Some(Operand::SymbolRef(tbl)) = module
                    .operand_ids(i.operands)
                    .last()
                    .map(|&id| module.operand(id))
                    && let Some(&tbl_block) = label_to_block.get(tbl)
                {
                    let tb = &blocks[tbl_block];
                    for stmt in &kernel.stmts[tb.start..tb.end] {
                        if let Stmt::BranchTargets(targets) = stmt {
                            for t in targets {
                                match label_to_block.get(t) {
                                    Some(&b) if !succs.contains(&b) => succs.push(b),
                                    Some(_) => {}
                                    None => unresolved.push((bid, *t)),
                                }
                            }
                            resolved = true;
                            break;
                        }
                    }
                }
                if !resolved && let Some(i) = last_instr {
                    // Table not found: honest hole, recorded.
                    unresolved.push((bid, i.mnemonic));
                }
                if let Some(i) = last_instr
                    && i.predicate.is_some()
                    && let Some(next) = next
                {
                    succs.push(next);
                }
            }
            Some(i) if Some(i.mnemonic) == sym_ret || Some(i.mnemonic) == sym_exit => {
                // Terminator. Predicated ret/exit falls through.
                if i.predicate.is_some()
                    && let Some(next) = next
                {
                    succs.push(next);
                }
            }
            _ => {
                if let Some(next) = next {
                    succs.push(next);
                }
            }
        }
        edges.push((bid, succs));
    }

    for (bid, succs) in edges {
        for s in &succs {
            blocks[*s].preds.push(bid);
        }
        blocks[bid].succs = succs;
    }

    Cfg {
        blocks,
        call_sites,
        unresolved_branches: unresolved,
    }
}

fn branch_target(
    module: &Module,
    _kernel: &Kernel,
    instr: &Instr,
    labels: &std::collections::HashMap<Symbol, BlockId>,
) -> Result<BlockId, Symbol> {
    for &id in module.operand_ids(instr.operands) {
        if let Operand::SymbolRef(name) = module.operand(id) {
            return labels.get(name).copied().ok_or(*name);
        }
    }
    Err(instr.mnemonic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::parse::parser::parse;

    fn cfg_of(body: &str) -> (Module, Cfg) {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let cfg = build_cfg(&module, &module.kernels[0]);
        (module, cfg)
    }

    fn shape(cfg: &Cfg) -> Vec<Vec<u32>> {
        cfg.blocks
            .iter()
            .map(|b| b.succs.iter().map(|s| s.0).collect())
            .collect()
    }

    #[test]
    fn block_names_are_labels_or_synthesized_from_the_id() {
        let (module, cfg) = cfg_of("@%p1 bra $L__A;\nadd.f32 %f1, %f1, %f1;\n$L__A:\nret;");
        let names: Vec<String> = (0..cfg.blocks.len() as u32)
            .map(|i| cfg.block_name(&module, BlockId(i)))
            .collect();
        assert_eq!(names, ["<block 0>", "<block 1>", "$L__A"]);
    }

    #[test]
    fn only_branch_targets_start_blocks() {
        // A label nobody branches to (LLVM's `$L__tmpN`) stays inside its
        // block, and labels after the final `ret` are not a block.
        let (module, cfg) = cfg_of(
            "add.f32 %f1, %f1, %f1;\n$L__tmp0:\nadd.f32 %f1, %f1, %f1;\n\
             @%p1 bra $L__A;\n$L__tmp1:\nmul.f32 %f1, %f1, %f1;\n$L__A:\nret;\n\
             $L__tmp2:\n$L__func_end0:",
        );
        assert_eq!(shape(&cfg), [vec![2, 1], vec![2], vec![]]);
        let names: Vec<String> = (0..cfg.blocks.len() as u32)
            .map(|i| cfg.block_name(&module, BlockId(i)))
            .collect();
        assert_eq!(names, ["<block 0>", "$L__tmp1", "$L__A"]);
    }

    #[test]
    fn conditional_branch_diamond() {
        // entry splits; both sides join at $L__J. succs[0] is the taken
        // target, succs[1] the fallthrough (PR 11 relies on this).
        let (_, cfg) = cfg_of(
            "setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__T;\n\
             add.f32 %f1, %f1, %f1;\nbra $L__J;\n\
             $L__T:\nmul.f32 %f1, %f1, %f1;\n\
             $L__J:\nret;",
        );
        assert_eq!(shape(&cfg), [vec![2, 1], vec![3], vec![3], vec![]]);
    }

    #[test]
    fn brx_table_edges() {
        let (_, cfg) = cfg_of(
            "$L_tbl: .branchtargets $L_a, $L_b;\nbrx.idx %r1, $L_tbl;\n\
             $L_a:\nret;\n$L_b:\nret;",
        );
        assert_eq!(shape(&cfg), [vec![1, 2], vec![], vec![]]);
        assert!(cfg.unresolved_branches.is_empty());
    }

    #[test]
    fn unreachable_block_is_kept() {
        let (_, cfg) = cfg_of("bra $L__END;\nadd.f32 %f1, %f1, %f1;\n$L__END:\nret;");
        assert_eq!(shape(&cfg), [vec![2], vec![2], vec![]]);
        assert!(
            cfg.block(BlockId(1)).preds.is_empty(),
            "unreachable: no preds"
        );
    }

    #[test]
    fn self_loop_back_edge() {
        let (_, cfg) = cfg_of(
            "mov.u32 %r1, 0;\n$L__L:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(shape(&cfg), [vec![1], vec![1, 2], vec![]]);
        assert_eq!(cfg.block(BlockId(1)).preds, [BlockId(0), BlockId(1)]);
    }

    #[test]
    fn predicated_ret_falls_through() {
        let (_, cfg) = cfg_of("@%p1 ret;\nadd.f32 %f1, %f1, %f1;\nret;");
        assert_eq!(shape(&cfg), [vec![1], vec![]]);
    }

    #[test]
    fn call_is_recorded_not_a_terminator() {
        let (_, cfg) = cfg_of("call foo;\nadd.f32 %f1, %f1, %f1;\nret;");
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.call_sites.len(), 1);
    }

    #[test]
    fn unresolved_target_is_recorded_never_panics() {
        let (_, cfg) = cfg_of("bra $L__MISSING;\nret;");
        assert_eq!(cfg.unresolved_branches.len(), 1);
    }
}
