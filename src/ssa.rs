//! Register SSA over the PTX control-flow graph: the value a read of a
//! register, CC included, sees. Minimal SSA (Cytron et al., TOPLAS
//! 1991): a φ for a register at each block in the iterated dominance
//! frontier of the blocks writing it. Values name PTX statements and
//! blocks; the CFG is not changed.

use crate::analysis::control_flow::DominanceInfo;
use crate::analysis::control_flow::dominators::iterated_frontier;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::Register;
use std::collections::{HashMap, HashSet};

/// A value of a register: what a statement wrote, what a block merged,
/// or nothing (MLIR's `OpResult` and `BlockArgument`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Value {
    /// Statement `op`'s write of `reg`. Where a guarded write's guard is
    /// false, `reg` keeps the value a read of it at `op` sees.
    Result { op: usize, reg: Register },
    /// A φ of `reg` at the start of `block`.
    Argument { block: BlockId, reg: Register },
    /// No write of `reg` on some path from the kernel's start.
    Undef(Register),
}

pub struct Values<'a> {
    cfg: &'a ControlFlowGraph,
    doms: &'a DominanceInfo,
    writes: HashMap<Register, Vec<usize>>,
    phis: HashSet<(Register, BlockId)>,
}

impl<'a> Values<'a> {
    /// `writes`: the statements writing each register, in order, each
    /// in a block of `cfg`.
    pub fn new(
        cfg: &'a ControlFlowGraph,
        doms: &'a DominanceInfo,
        writes: HashMap<Register, Vec<usize>>,
    ) -> Self {
        let df = doms.frontiers(cfg);
        let phis = (writes.iter())
            .flat_map(|(&reg, ws)| {
                let blocks: Vec<BlockId> = ws.iter().map(|&w| block_of(cfg, w)).collect();
                iterated_frontier(&df, &blocks)
                    .into_iter()
                    .map(move |b| (reg, b))
            })
            .collect();
        Values {
            cfg,
            doms,
            writes,
            phis,
        }
    }

    /// The value of `reg` a read at statement `pos` sees; `None` in an
    /// unreachable block.
    pub fn at(&self, reg: Register, pos: usize) -> Option<Value> {
        self.before(reg, block_of(self.cfg, pos), pos)
    }

    /// The value of `reg` after block `b`'s last statement.
    pub fn at_end(&self, reg: Register, b: BlockId) -> Option<Value> {
        self.before(reg, b, self.cfg.block(b).end)
    }

    /// The value of `reg` before statement `end` of block `b`, or after
    /// it when `end` is the block's end.
    fn before(&self, reg: Register, mut b: BlockId, mut end: usize) -> Option<Value> {
        if !self.doms.is_reachable(b) {
            return None;
        }
        let ws = self.writes.get(&reg).map_or(&[][..], Vec::as_slice);
        loop {
            let i = ws.partition_point(|&w| w < end);
            if i > 0 && ws[i - 1] >= self.cfg.block(b).start {
                return Some(Value::Result { op: ws[i - 1], reg });
            }
            if self.phis.contains(&(reg, b)) {
                return Some(Value::Argument { block: b, reg });
            }
            // A block without a φ starts with what its immediate
            // dominator ends with.
            match self.doms.idom[b.0 as usize] {
                Some(d) => (b, end) = (d, self.cfg.block(d).end),
                None => return Some(Value::Undef(reg)),
            }
        }
    }
}

fn block_of(cfg: &ControlFlowGraph, stmt: usize) -> BlockId {
    (cfg.blocks.iter_enumerated())
        .find(|(_, b)| b.start <= stmt && stmt < b.end)
        .map(|(id, _)| id)
        .expect("statement belongs to a block")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::dominators;
    use crate::analysis::register_effects::{carry, stmt_written_registers};
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;

    /// Where a test reads: before a line of the body, or at a block's end.
    #[derive(Clone, Copy)]
    enum At {
        L(usize),
        End(u32),
    }
    use At::{End, L};

    /// The values `reads` see in kernel `body`, one statement per line;
    /// `%cc` is CC. Each value prints as `wN` (written at line N),
    /// `phiB`, `undef` or `none`.
    fn values(body: &str, reads: &[(&str, At)]) -> Vec<String> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let kernel = &module.kernels[0];
        assert_eq!(kernel.stmts.len(), body.lines().count());
        let cfg = build_cfg(&module, kernel);
        let doms = dominators(&cfg);
        let mut writes: HashMap<Register, Vec<usize>> = HashMap::new();
        for (i, stmt) in kernel.stmts.iter().enumerate() {
            for r in stmt_written_registers(&module, stmt) {
                writes.entry(Register::Named(r)).or_default().push(i);
            }
            if let Stmt::Instr(instr) = stmt
                && carry(&module, instr).0
            {
                writes.entry(Register::Cc).or_default().push(i);
            }
        }
        let v = Values::new(&cfg, &doms, writes);
        (reads.iter())
            .map(|&(name, at)| {
                let reg = match name {
                    "%cc" => Register::Cc,
                    _ => Register::Named(module.interner.get(name).expect("register")),
                };
                let value = match at {
                    L(line) => v.at(reg, line),
                    End(b) => v.at_end(reg, BlockId(b)),
                };
                match value {
                    Some(Value::Result { op, .. }) => format!("w{op}"),
                    Some(Value::Argument { block, .. }) => format!("phi{}", block.0),
                    Some(Value::Undef(_)) => "undef".to_owned(),
                    None => "none".to_owned(),
                }
            })
            .collect()
    }

    #[test]
    fn a_read_sees_the_last_write_before_it() {
        let body = "mov.u32 %r1, 1;\nadd.u32 %r1, %r1, 2;\nst.global.u32 [%rd9], %r1;\nret;";
        let reads = [("%r1", L(1)), ("%r1", L(2)), ("%r1", L(0)), ("%rd9", L(2))];
        assert_eq!(values(body, &reads), ["w0", "w1", "undef", "undef"]);
    }

    #[test]
    fn a_write_on_one_arm_meets_the_other_in_a_phi() {
        // blocks: 0 lines 0-2, 1 line 3, 2 lines 4-6
        let body = "mov.u32 %r1, 1;\nsetp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__J;\n\
                    mov.u32 %r1, 2;\n\
                    $L__J:\nst.global.u32 [%rd9], %r1;\nret;";
        let reads = [
            ("%r1", L(5)),
            ("%p1", L(5)),
            ("%r1", End(1)),
            ("%r1", End(0)),
        ];
        assert_eq!(values(body, &reads), ["phi2", "w1", "w3", "w0"]);
    }

    #[test]
    fn a_counter_is_a_phi_at_its_header_until_its_increment() {
        // blocks: 0 line 0, 1 lines 1-4, 2 lines 5-6
        let body = "mov.u32 %r1, 0;\n$L__L:\nadd.s32 %r1, %r1, 1;\n\
                    setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\n\
                    st.global.u32 [%rd9], %r1;\nret;";
        let reads = [
            ("%r1", L(2)),
            ("%r1", L(3)),
            ("%r1", L(5)),
            ("%r2", L(3)),
            ("%r1", End(1)),
        ];
        assert_eq!(values(body, &reads), ["phi1", "w2", "w2", "undef", "w2"]);
    }

    #[test]
    fn an_outer_counter_read_in_an_inner_loop_is_the_outer_phi() {
        // blocks: 0 line 0, 1 lines 1-2, 2 lines 3-6, 3 lines 7-9, 4 line 10
        let body = "mov.u32 %r1, 0;\n$L__O:\nmov.u32 %r3, 0;\n\
                    $L__I:\nadd.s32 %r3, %r3, 1;\nsetp.lt.s32 %p1, %r3, %r2;\n@%p1 bra $L__I;\n\
                    add.s32 %r1, %r1, 1;\nsetp.lt.s32 %p2, %r1, %r2;\n@%p2 bra $L__O;\nret;";
        let reads = [("%r3", L(4)), ("%r1", L(5)), ("%r1", L(7)), ("%r3", L(2))];
        assert_eq!(values(body, &reads), ["phi2", "phi1", "phi1", "phi1"]);
    }

    #[test]
    fn each_entry_of_a_two_entry_cycle_has_a_phi() {
        // blocks: 0 lines 0-1, 1 lines 2-3, 2 lines 4-7, 3 line 8
        let body = "setp.gt.s32 %p1, %r1, %r2;\n@%p1 bra $L__B;\n\
                    $L__A:\nadd.s32 %r3, %r3, 1;\n\
                    $L__B:\nadd.s32 %r3, %r3, 2;\n\
                    setp.lt.s32 %p2, %r3, %r1;\n@%p2 bra $L__A;\nret;";
        let reads = [("%r3", L(3)), ("%r3", L(5)), ("%r3", L(6)), ("%r3", L(8))];
        assert_eq!(values(body, &reads), ["phi1", "phi2", "w5", "w5"]);
    }

    #[test]
    fn an_entry_block_with_predecessors_has_a_phi() {
        let body = "$L__E:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, %r2;\n\
                    @%p1 bra $L__E;\nret;";
        let reads = [("%r1", L(1)), ("%r2", L(2)), ("%r1", L(4))];
        assert_eq!(values(body, &reads), ["phi0", "undef", "w1"]);
    }

    #[test]
    fn an_unreachable_block_reads_nothing_and_its_writes_reach_nothing() {
        // blocks: 0 lines 0-1, 1 lines 2-3 (unreachable), 2 lines 4-6
        let body = "mov.u32 %r1, 1;\nbra $L__END;\n\
                    mov.u32 %r1, 2;\nst.global.u32 [%rd9], %r1;\n\
                    $L__END:\nst.global.u32 [%rd9], %r1;\nret;";
        let reads = [("%r1", L(3)), ("%r1", End(1)), ("%r1", L(5))];
        assert_eq!(values(body, &reads), ["none", "none", "w0"]);
    }

    #[test]
    fn a_call_writes_the_carry_flag() {
        let body = "add.cc.u32 %r2, %r0, 1;\ncall (%r5), f, (%r0);\naddc.u32 %r3, 0, 0;\nret;";
        let reads = [("%cc", L(1)), ("%cc", L(2)), ("%cc", L(0))];
        assert_eq!(values(body, &reads), ["w0", "w1", "undef"]);
    }
}
