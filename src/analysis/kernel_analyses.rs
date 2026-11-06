//! Register definitions and control-flow queries for one kernel.
//! Builds SSA, reachability, the joins of each
//! branch and the exit branches of each loop; contains no scalar interpretation, uniformity solver or
//! report state.

use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::control_flow::sync_dependence::{exit_branches, join_blocks};
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Operand, Register, Stmt};
use crate::ptx::register_effects;
use crate::ssa::{RegisterSsa, Value};
use crate::support::intern::Symbol;
use std::collections::HashMap;

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
