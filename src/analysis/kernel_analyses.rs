//! Register definitions and control-flow queries for one kernel.
//! Builds SSA, reachability, the joins of each
//! branch and the exit branches of each loop; contains no scalar interpretation, uniformity solver or
//! report state.

use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::control_flow::sync_dependence::{exit_branches, join_blocks};
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Operand, Register, Stmt};
use crate::ptx::modifier::Form;
use crate::ptx::register_effects;
use crate::ssa::{RegisterSsa, Value};
use crate::support::intern::Symbol;
use std::collections::{HashMap, HashSet};

pub(crate) struct KernelAnalyses<'a> {
    pub(crate) module: &'a Module,
    pub(crate) kernel: &'a Kernel,
    pub(crate) cfg: &'a ControlFlowGraph,
    pub(crate) forest: &'a LoopForest,
    /// The statements writing each register, CC included, in order.
    pub(crate) writes: HashMap<Register, Vec<usize>>,
    /// The value each read of a register sees.
    pub(crate) ssa: RegisterSsa<'a>,
    /// `reach[a][b]`: a path of at least one edge from block a to b.
    pub(crate) reach: Vec<Vec<bool>>,
    /// `joins[b]`: the blocks where the paths from block b's successors
    /// meet again.
    pub(crate) joins: Vec<Vec<BlockId>>,
    /// `exit_branches[l]`: the branches that decide at which trip, or by
    /// which exit, a thread leaves loop l.
    pub(crate) exit_branches: Vec<Vec<BlockId>>,
}

impl<'a> KernelAnalyses<'a> {
    pub(crate) fn new(
        module: &'a Module,
        kernel: &'a Kernel,
        cfg: &'a ControlFlowGraph,
        forest: &'a LoopForest,
    ) -> Self {
        let mut writes: HashMap<Register, Vec<usize>> = HashMap::new();
        for (i, stmt) in kernel.stmts.iter().enumerate() {
            let in_cfg = (cfg.blocks.iter()).any(|b| (b.start..b.end).contains(&i));
            if !in_cfg {
                continue;
            }
            for reg in register_effects::writes(module, stmt) {
                writes.entry(reg).or_default().push(i);
            }
        }
        let n = cfg.blocks.len();
        let mut reach = vec![vec![false; n]; n];
        for (a, row) in reach.iter_mut().enumerate() {
            let mut stack: Vec<BlockId> = cfg.block(BlockId(a as u32)).succs.clone();
            while let Some(b) = stack.pop() {
                if !row[b.0 as usize] {
                    row[b.0 as usize] = true;
                    stack.extend(cfg.block(b).succs.iter().copied());
                }
            }
        }
        let ssa = RegisterSsa::new(cfg, &forest.doms, writes.clone());
        Self {
            module,
            kernel,
            cfg,
            forest,
            writes,
            ssa,
            reach,
            joins: join_blocks(cfg, forest),
            exit_branches: exit_branches(cfg, forest),
        }
    }

    pub(crate) fn last_instr(&self, b: BlockId) -> Option<(usize, &Instr)> {
        let blk = self.cfg.block(b);
        self.kernel.stmts[blk.start..blk.end]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, s)| match s {
                Stmt::Instr(instr) => Some((blk.start + i, instr)),
                _ => None,
            })
    }

    /// The last write of `reg` in block `b` before statement `before`.
    pub(crate) fn last_write_in(&self, b: BlockId, before: usize, reg: Symbol) -> Option<usize> {
        let blk = self.cfg.block(b);
        let defs = self.writes.get(&Register::Named(reg))?;
        (defs.iter().rev())
            .copied()
            .find(|&d| blk.start <= d && d < before)
    }

    /// The statement writing the value `reg` holds where statement `pos`
    /// reads it, and its instruction; `None` for a φ, a value no path
    /// writes or a statement the parser did not read.
    pub(crate) fn def_at(&self, reg: Symbol, pos: usize) -> Option<(usize, &'a Instr)> {
        let Some(Value::Result { op, .. }) = self.ssa.value_before(pos, Register::Named(reg))
        else {
            return None;
        };
        match &self.kernel.stmts[op] {
            Stmt::Instr(instr) => Some((op, instr)),
            _ => None,
        }
    }

    /// Which destination of `instr` register `reg` is: 0, or 1 for the
    /// second of `d|p`.
    pub(crate) fn destination(&self, instr: &Instr, reg: Symbol) -> Option<usize> {
        let first = *self.module.operand_ids(instr.operands).first()?;
        let places = match self.module.operand(first) {
            Operand::MultipleDestinations { children } => self.module.operand_ids(*children),
            _ => std::slice::from_ref(&first),
        };
        (places.iter()).position(|&id| *self.module.operand(id) == Operand::Register(reg))
    }

    /// The `setp` whose output `pred` holds where statement `pos` reads
    /// it, and whether `pred` is its second output: of a `setp` with no
    /// Boolean operator, the complement of the comparison (PTX ISA
    /// §9.7.6.2: "A related value computed using the complement of the
    /// compare result is written to the second destination operand").
    pub(crate) fn setp_at(&self, pred: Symbol, pos: usize) -> Option<(usize, &Instr, bool)> {
        let (op, instr) = self.def_at(pred, pos)?;
        let place = self.destination(instr, pred)?;
        (instr.mnemonic == Opcode::Setp).then_some((op, instr, place == 1))
    }

    /// Predicate value `g` holding makes `p` hold: `g` is `p`, or an
    /// unguarded `and.pred` or `mov.pred` of a predicate that does.
    pub(crate) fn implies(&self, g: Value, p: Value) -> bool {
        let Value::Result { op, .. } = g else {
            return g == p;
        };
        let Stmt::Instr(instr) = &self.kernel.stmts[op] else {
            return false;
        };
        let read = register_effects::split_operands(self.module, instr).1;
        g == p
            || instr.predicate.is_none()
                && matches!(instr.mnemonic, Opcode::And | Opcode::Mov)
                && Form::of(self.module, instr) == Form::Pred
                && read.iter().any(|&o| match *self.module.operand(o) {
                    Operand::Register(r) => (self.ssa.value_before(op, Register::Named(r)))
                        .is_some_and(|a| self.implies(a, p)),
                    _ => false,
                })
    }

    /// Every read of `v` runs where predicate `p` holds: under a guard that
    /// implies `p`, or by an unguarded pure instruction whose result is
    /// read so in turn; none through a φ. A read is of an operand, the
    /// guard or CC.
    pub(crate) fn read_only_under(&self, v: Value, p: Value) -> bool {
        let (mut work, mut seen) = (vec![(v, 0)], HashSet::new());
        while let Some((w @ Value::Result { reg, .. }, depth)) = work.pop() {
            if !seen.insert(w) {
                continue;
            }
            for pos in self.cfg.blocks.iter().flat_map(|b| b.start..b.end) {
                let Stmt::Instr(instr) = &self.kernel.stmts[pos] else {
                    continue;
                };
                let at = |r: Register| self.ssa.value_before(pos, r);
                let reads: Vec<Value> = (register_effects::split_operands(self.module, instr).1)
                    .iter()
                    .flat_map(|&o| register_effects::registers(self.module, o))
                    .map(Register::Named)
                    .chain(instr.predicate.map(|g| Register::Named(g.reg)))
                    .chain(register_effects::reads_cc(self.module, instr).then_some(Register::Cc))
                    .filter(|&r| r == reg)
                    .filter_map(at)
                    .collect();
                let phi = |x: &Value| {
                    matches!(x, Value::Argument { .. }) && self.ssa.sources(*x).contains(&w)
                };
                if reads.iter().any(phi) {
                    return false;
                }
                if !reads.contains(&w) {
                    continue;
                }
                match instr.predicate {
                    Some(g)
                        if !g.negated
                            && at(Register::Named(g.reg)).is_some_and(|g| self.implies(g, p)) => {}
                    None if register_effects::is_pure(instr.mnemonic) && depth < 16 => {
                        let written =
                            register_effects::writes(self.module, &self.kernel.stmts[pos]);
                        work.extend(
                            written
                                .into_iter()
                                .map(|reg| (Value::Result { op: pos, reg }, depth + 1)),
                        );
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    /// The block holding statement `stmt`.
    pub(crate) fn block_of(&self, stmt: usize) -> BlockId {
        (0..self.cfg.blocks.len() as u32)
            .map(BlockId)
            .find(|&b| {
                let blk = self.cfg.block(b);
                blk.start <= stmt && stmt < blk.end
            })
            .expect("statement belongs to a block")
    }

    /// A path from `from` to `to` of at least one edge that avoids `avoid`.
    pub(crate) fn path_avoiding(&self, from: BlockId, to: BlockId, avoid: BlockId) -> bool {
        let mut seen = vec![false; self.cfg.blocks.len()];
        let mut stack: Vec<BlockId> = self.cfg.block(from).succs.clone();
        while let Some(b) = stack.pop() {
            if b == avoid || seen[b.0 as usize] {
                continue;
            }
            if b == to {
                return true;
            }
            seen[b.0 as usize] = true;
            stack.extend(self.cfg.block(b).succs.iter().copied());
        }
        false
    }

    /// Value `v` is defined in loop `l`, and may differ between its trips.
    pub(crate) fn defined_in(&self, v: Value, l: LoopId) -> bool {
        let b = match v {
            Value::Result { op, .. } => self.block_of(op),
            Value::Argument { block, .. } => block,
            Value::Undef(_) => return false,
        };
        self.forest.get(l).blocks.binary_search(&b).is_ok()
    }

    /// Loop `inner` is `outer` or nested in it, and so runs again on each
    /// of `outer`'s trips.
    pub(crate) fn loop_within(&self, inner: LoopId, outer: LoopId) -> bool {
        std::iter::successors(Some(inner), |&l| self.forest.get(l).parent).any(|l| l == outer)
    }

    pub(crate) fn in_loop(&self, id: LoopId, stmt: usize) -> bool {
        self.forest.get(id).blocks.iter().any(|&b| {
            let blk = self.cfg.block(b);
            blk.start <= stmt && stmt < blk.end
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    /// %p6, a copy of %p3 ∧ (%p1 ∧ %p2), holds only where each of the three
    /// does; `!%p1 ∧ %p2`, `%p1 ∨ %p2` and a guarded `and.pred` do not make
    /// %p1 hold.
    #[test]
    fn an_and_pred_chain_implies_each_operand() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k()\n{\n\
                   mov.u32 %r1, %tid.x;\nsetp.lt.u32 %p1, %r1, 16;\nsetp.lt.u32 %p2, %r1, 8;\n\
                   setp.lt.u32 %p3, %r1, 4;\nand.pred %p4, %p1, %p2;\nand.pred %p5, %p3, %p4;\n\
                   mov.pred %p6, %p5;\nand.pred %p7, !%p1, %p2;\nor.pred %p8, %p1, %p2;\n\
                   @%p3 and.pred %p9, %p1, %p2;\nret;\n}\n";
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let data = KernelAnalyses::new(&m, k, &cfg, &forest);
        let end = k.stmts.len() - 1;
        let value = |name| {
            let reg = Register::Named(m.interner.get(name).expect("register"));
            data.ssa.value_before(end, reg).expect("reachable")
        };
        let implies = |g, p| data.implies(value(g), value(p));
        assert!(
            ["%p1", "%p2", "%p3", "%p6"]
                .iter()
                .all(|p| implies("%p6", p))
        );
        assert!(implies("%p4", "%p1") && !implies("%p4", "%p3"));
        assert!(!implies("%p7", "%p1") && implies("%p7", "%p2"));
        assert!(!implies("%p8", "%p1") && !implies("%p9", "%p1"));
    }
}
