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

/// Register SSA of one kernel, over its unchanged statements and CFG,
/// as LLVM's `MemorySSA` is over memory.
pub struct RegisterSsa<'a> {
    cfg: &'a ControlFlowGraph,
    doms: &'a DominanceInfo,
    writes: HashMap<Register, Vec<usize>>,
    phis: HashSet<(Register, BlockId)>,
}

impl<'a> RegisterSsa<'a> {
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
        RegisterSsa {
            cfg,
            doms,
            writes,
            phis,
        }
    }

    /// The value a read of `reg` just before statement `stmt` sees
    /// (MLIR's `getProgramPointBefore`); `stmt` is any statement of a
    /// block, a label included. `None` in an unreachable block.
    pub fn value_before(&self, stmt: usize, reg: Register) -> Option<Value> {
        self.lookup(reg, block_of(self.cfg, stmt), stmt)
    }

    /// What φ `arg` takes on each edge into its block: the value at each
    /// reachable predecessor's end, and `Undef` from the kernel's start
    /// (`None`), an edge into the entry block. Empty for a value that is
    /// not a φ.
    pub fn incoming(&self, arg: Value) -> Vec<(Option<BlockId>, Value)> {
        let Value::Argument { block, reg } = arg else {
            return Vec::new();
        };
        let start = (block == ControlFlowGraph::ENTRY).then_some((None, Value::Undef(reg)));
        let preds = (self.cfg.block(block).preds.iter()).filter_map(|&p| {
            let end = self.cfg.block(p).end;
            self.lookup(reg, p, end).map(|v| (Some(p), v))
        });
        start.into_iter().chain(preds).collect()
    }

    /// The `Result`s `v` may be, through φs, in statement order, then
    /// `Undef` if some path writes nothing; each once. A guarded
    /// `Result` is not looked through.
    pub fn sources(&self, v: Value) -> Vec<Value> {
        let mut seen = HashSet::new();
        let mut work = vec![v];
        let mut out = Vec::new();
        while let Some(v) = work.pop() {
            if !seen.insert(v) {
                continue;
            }
            match v {
                Value::Argument { .. } => work.extend(self.incoming(v).into_iter().map(|(_, w)| w)),
                _ => out.push(v),
            }
        }
        out.sort_by_key(|v| match *v {
            Value::Result { op, .. } => op,
            _ => usize::MAX,
        });
        out
    }

    /// The value of `reg` before statement `end` of block `b`, or after
    /// the block when `end` is its end.
    fn lookup(&self, reg: Register, mut b: BlockId, mut end: usize) -> Option<Value> {
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
    use crate::analysis::register_effects::writes;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    /// Where a test reads: before a line of the body, or at a block's end.
    #[derive(Clone, Copy)]
    enum At {
        L(usize),
        End(u32),
    }
    use At::{End, L};

    /// Kernel `body`'s register SSA, one statement per line, handed to
    /// `f` with a lookup of registers by name, `%cc` for CC.
    fn with_ssa<R>(
        body: &str,
        f: impl FnOnce(&dyn Fn(&str) -> Register, &ControlFlowGraph, &RegisterSsa) -> R,
    ) -> R {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let kernel = &module.kernels[0];
        assert_eq!(kernel.stmts.len(), body.lines().count());
        let cfg = build_cfg(&module, kernel);
        let doms = dominators(&cfg);
        let mut table: HashMap<Register, Vec<usize>> = HashMap::new();
        for (i, stmt) in kernel.stmts.iter().enumerate() {
            for reg in writes(&module, stmt) {
                table.entry(reg).or_default().push(i);
            }
        }
        let reg = |name: &str| match name {
            "%cc" => Register::Cc,
            _ => Register::Named(module.interner.get(name).expect("register")),
        };
        f(&reg, &cfg, &RegisterSsa::new(&cfg, &doms, table))
    }

    /// `wN` (written at line N), `phiB`, `undef` or `none`.
    fn show(value: Option<Value>) -> String {
        match value {
            Some(Value::Result { op, .. }) => format!("w{op}"),
            Some(Value::Argument { block, .. }) => format!("phi{}", block.0),
            Some(Value::Undef(_)) => "undef".to_owned(),
            None => "none".to_owned(),
        }
    }

    /// The values `reads` see in kernel `body`.
    fn values(body: &str, reads: &[(&str, At)]) -> Vec<String> {
        with_ssa(body, |reg, cfg, ssa| {
            (reads.iter())
                .map(|&(name, at)| {
                    show(match at {
                        L(line) => ssa.value_before(line, reg(name)),
                        End(b) => ssa.lookup(reg(name), BlockId(b), cfg.block(BlockId(b)).end),
                    })
                })
                .collect()
        })
    }

    /// The `incoming` (`bN:` from block N, `start:` from the kernel's
    /// start) and, after `->`, the `sources` of the value `name` has
    /// before line `line`.
    fn merged(body: &str, name: &str, line: usize) -> String {
        with_ssa(body, |reg, _, ssa| {
            let v = ssa.value_before(line, reg(name)).expect("a reachable line");
            let incoming = (ssa.incoming(v).into_iter()).map(|(p, w)| match p {
                Some(p) => format!("b{}:{}", p.0, show(Some(w))),
                None => format!("start:{}", show(Some(w))),
            });
            let sources = (ssa.sources(v).into_iter()).map(|w| show(Some(w)));
            let incoming: Vec<String> = incoming.collect();
            let sources: Vec<String> = sources.collect();
            format!("{} -> {}", incoming.join(" "), sources.join(" "))
        })
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
        assert_eq!(merged(body, "%r1", 5), "b0:w0 b1:w3 -> w0 w3");
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
        assert_eq!(merged(body, "%r1", 2), "b0:w0 b1:w2 -> w0 w2");
    }

    #[test]
    fn an_outer_counter_read_in_an_inner_loop_is_the_outer_phi() {
        // blocks: 0 line 0, 1 lines 1-2, 2 lines 3-6, 3 lines 7-9, 4 line 10
        let body = "mov.u32 %r1, 0;\n$L__O:\nmov.u32 %r3, 0;\n\
                    $L__I:\nadd.s32 %r3, %r3, 1;\nsetp.lt.s32 %p1, %r3, %r2;\n@%p1 bra $L__I;\n\
                    add.s32 %r1, %r1, 1;\nsetp.lt.s32 %p2, %r1, %r2;\n@%p2 bra $L__O;\nret;";
        let reads = [("%r3", L(4)), ("%r1", L(5)), ("%r1", L(7)), ("%r3", L(2))];
        assert_eq!(values(body, &reads), ["phi2", "phi1", "phi1", "phi1"]);
        assert_eq!(merged(body, "%r1", 5), "b0:w0 b3:w7 -> w0 w7");
        // %r3's φ at the outer header holds nothing a read sees; its
        // value on entry was never written.
        assert_eq!(merged(body, "%r3", 2), "b0:undef b3:w4 -> w4 undef");
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
        assert_eq!(merged(body, "%r3", 3), "b0:undef b2:w5 -> w5 undef");
        assert_eq!(merged(body, "%r3", 5), "b0:undef b1:w3 -> w3 undef");
    }

    #[test]
    fn an_entry_block_with_predecessors_has_a_phi() {
        let body = "$L__E:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, %r2;\n\
                    @%p1 bra $L__E;\nret;";
        let reads = [("%r1", L(1)), ("%r2", L(2)), ("%r1", L(4))];
        assert_eq!(values(body, &reads), ["phi0", "undef", "w1"]);
        assert_eq!(merged(body, "%r1", 1), "start:undef b0:w1 -> w1 undef");
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
    fn sources_follow_a_cycle_of_phis_once() {
        // blocks: 0 line 0, 1 lines 1-3 (header), 2 line 4, 3 lines 5-7
        // (join and latch), 4 line 8
        let body = "mov.u32 %r1, 0;\n$L__H:\nsetp.lt.s32 %p1, %r2, 3;\n@%p1 bra $L__S;\n\
                    mov.u32 %r1, 7;\n\
                    $L__S:\nsetp.lt.s32 %p2, %r1, %r2;\n@%p2 bra $L__H;\nret;";
        assert_eq!(merged(body, "%r1", 6), "b1:phi1 b2:w4 -> w0 w4");
        assert_eq!(merged(body, "%r1", 2), "b0:w0 b3:phi3 -> w0 w4");
    }

    #[test]
    fn a_phi_takes_nothing_from_an_unreachable_predecessor() {
        // blocks: 0 lines 0-1, 1 line 2 (unreachable), 2 lines 3-6, 3 line 7
        let body = "mov.u32 %r1, 0;\nbra $L__H;\n\
                    mov.u32 %r1, 5;\n\
                    $L__H:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__H;\n\
                    ret;";
        assert_eq!(merged(body, "%r1", 4), "b0:w0 b2:w4 -> w0 w4");
    }

    #[test]
    fn a_call_writes_the_carry_flag() {
        let body = "add.cc.u32 %r2, %r0, 1;\ncall (%r5), f, (%r0);\naddc.u32 %r3, 0, 0;\nret;";
        let reads = [("%cc", L(1)), ("%cc", L(2)), ("%cc", L(0))];
        assert_eq!(values(body, &reads), ["w0", "w1", "undef"]);
    }
}
