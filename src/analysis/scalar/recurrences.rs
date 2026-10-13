//! Loop counters as recurrences on register SSA values: a φ at a loop's
//! header that every back edge feeds back plus the same step, as LLVM's
//! ScalarEvolution recognizes an add-recurrence on a header PHI
//! (`createAddRecFromPHI`). Recognition is structural; whether a
//! register step is invariant, and what the start is, are the reader's.

use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::register_effects::writes;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Operand, OperandId, Register, Stmt};
use crate::ptx::literal::parse_int;
use crate::ssa::{RegisterSsa, Value};
use crate::support::intern::Symbol;
use std::collections::BTreeSet;

/// A counter of a loop: `phi`, the header φ, is `start` on entry and
/// `phi + step` on every back edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recurrence {
    pub phi: Value,
    pub start: Value,
    pub step: Step,
}

/// A sum of constants and of registers, each read at the statement
/// that adds it, with its sign.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Step {
    pub constant: i64,
    pub registers: Vec<(Symbol, usize, i64)>,
}

/// The recurrences of loop `l`.
pub fn recurrences(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    ssa: &RegisterSsa,
    l: LoopId,
) -> Vec<Recurrence> {
    let lp = forest.get(l);
    let start = cfg.block(lp.header).start;
    let in_loop = |b: BlockId| lp.blocks.binary_search(&b).is_ok();
    let regs: BTreeSet<Symbol> = (lp.blocks.iter())
        .flat_map(|&b| cfg.block(b).start..cfg.block(b).end)
        .flat_map(|i| writes(module, &kernel.stmts[i]))
        .filter_map(|r| match r {
            Register::Named(r) => Some(r),
            Register::Cc => None,
        })
        .collect();
    let walk = Walk {
        module,
        kernel,
        cfg,
        forest,
        ssa,
        l,
    };
    let mut out = Vec::new();
    for reg in regs {
        let Some(phi @ Value::Argument { block, .. }) =
            ssa.value_before(start, Register::Named(reg))
        else {
            continue;
        };
        if block != lp.header {
            continue;
        }
        let (mut starts, mut steps) = (Vec::new(), Vec::new());
        for (p, v) in ssa.incoming(phi) {
            match p.filter(|&p| in_loop(p)) {
                Some(_) => steps.push(walk.offset(v, phi, 16)),
                None => starts.push(v),
            }
        }
        let [start, rest @ ..] = &starts[..] else {
            continue;
        };
        let Some(Some(step)) = steps.first() else {
            continue;
        };
        // Register steps match by the value they add, wherever it is added.
        let read = |&(r, at, sign): &(Symbol, usize, i64)| {
            (ssa.value_before(at, Register::Named(r)), sign)
        };
        let same = |s: &Step| {
            s.constant == step.constant
                && s.registers
                    .iter()
                    .map(read)
                    .eq(step.registers.iter().map(read))
        };
        if rest.iter().all(|v| v == start)
            && steps.iter().all(|s| s.as_ref().is_some_and(same))
            && *step != Step::default()
        {
            out.push(Recurrence {
                phi,
                start: *start,
                step: step.clone(),
            });
        }
    }
    out
}

struct Walk<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    cfg: &'a ControlFlowGraph,
    forest: &'a LoopForest,
    ssa: &'a RegisterSsa<'a>,
    l: LoopId,
}

impl Walk<'_> {
    /// The step from `phi` to `v` through unguarded `mov`, integer `add`
    /// and `sub` in the loop's own blocks, or `None`.
    fn offset(&self, v: Value, phi: Value, depth: u32) -> Option<Step> {
        if v == phi {
            return Some(Step::default());
        }
        let Value::Result { op, .. } = v else {
            return None;
        };
        let Stmt::Instr(instr) = &self.kernel.stmts[op] else {
            return None;
        };
        let own = (self.cfg.blocks.iter_enumerated())
            .find(|(_, b)| b.start <= op && op < b.end)
            .is_some_and(|(b, _)| self.forest.block_loop[b.0 as usize] == Some(self.l));
        if depth == 0 || instr.predicate.is_some() || !own {
            return None;
        }
        let ops = self.module.operand_ids(instr.operands);
        let reg_value = |o: OperandId| match self.module.operand(o) {
            Operand::Register(r) => self.ssa.value_before(op, Register::Named(*r)),
            _ => None,
        };
        let term = |o: OperandId, sign: i64| -> Option<Step> {
            match self.module.operand(o) {
                Operand::Immediate(t) => Some(Step {
                    constant: sign * parse_int(self.module.interner.resolve(*t))?,
                    registers: Vec::new(),
                }),
                Operand::Register(r) => Some(Step {
                    constant: 0,
                    registers: vec![(*r, op, sign)],
                }),
                _ => None,
            }
        };
        let then = |from: OperandId, plus: Option<Step>| -> Option<Step> {
            let mut s = self.offset(reg_value(from)?, phi, depth - 1)?;
            let plus = plus?;
            s.constant = s.constant.checked_add(plus.constant)?;
            s.registers.extend(plus.registers);
            Some(s)
        };
        match (instr.mnemonic, ops) {
            (Opcode::Mov, [_, src]) => then(*src, Some(Step::default())),
            (Opcode::Add, [_, a, b]) if self.integer(instr) => {
                then(*a, term(*b, 1)).or_else(|| then(*b, term(*a, 1)))
            }
            (Opcode::Sub, [_, a, b]) if self.integer(instr) => then(*a, term(*b, -1)),
            _ => None,
        }
    }

    /// A scalar integer `add` or `sub`, not saturating.
    fn integer(&self, instr: &Instr) -> bool {
        let mods: Vec<&str> = (self.module.modifiers(instr).iter())
            .map(|&m| self.module.interner.resolve(m))
            .collect();
        !mods.contains(&"sat")
            && mods
                .iter()
                .any(|m| matches!(*m, "s16" | "u16" | "s32" | "u32" | "s64" | "u64"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;
    use std::collections::HashMap;

    /// The recurrences of loop `loop_index` (by header order) in kernel
    /// `body`, one statement per line, as `reg: start → step`, the start
    /// `wN` (written at line N), `phiB` or `undef`, the step its constant
    /// and `±reg@N` per register.
    fn recurrences_of(body: &str, loop_index: usize) -> Vec<String> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body}\n}}\n"
        );
        let module = parse(&src).expect("test body parses");
        let kernel = &module.kernels[0];
        assert_eq!(kernel.stmts.len(), body.lines().count());
        let cfg = build_cfg(&module, kernel);
        let forest = loop_forest(&cfg);
        let mut table: HashMap<Register, Vec<usize>> = HashMap::new();
        for (i, stmt) in kernel.stmts.iter().enumerate() {
            for reg in writes(&module, stmt) {
                table.entry(reg).or_default().push(i);
            }
        }
        let ssa = RegisterSsa::new(&cfg, &forest.doms, table);
        let mut loops: Vec<LoopId> = (0..forest.loops.len() as u32).map(LoopId).collect();
        loops.sort_by_key(|&l| forest.get(l).header);
        let show = |v: Value| match v {
            Value::Result { op, .. } => format!("w{op}"),
            Value::Argument { block, .. } => format!("phi{}", block.0),
            Value::Undef(_) => "undef".to_owned(),
        };
        recurrences(&module, kernel, &cfg, &forest, &ssa, loops[loop_index])
            .into_iter()
            .map(|r| {
                let Value::Argument {
                    reg: Register::Named(reg),
                    ..
                } = r.phi
                else {
                    unreachable!("a recurrence is a named register's header φ")
                };
                let mut step = format!("{:+}", r.step.constant);
                for (s, at, sign) in &r.step.registers {
                    let sign = if *sign > 0 { "+" } else { "-" };
                    step += &format!(" {sign}{}@{at}", module.interner.resolve(*s));
                }
                format!(
                    "{}: {} → {step}",
                    module.interner.resolve(reg),
                    show(r.start)
                )
            })
            .collect()
    }

    #[test]
    fn a_counter_steps_by_its_add() {
        let body = "mov.u32 %r1, 0;\n$L:\nadd.s32 %r1, %r1, 1;\n\
                    setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L;\nret;";
        assert_eq!(recurrences_of(body, 0), ["%r1: w0 → +1"]);
    }

    #[test]
    fn a_chain_of_registers_is_one_recurrence_of_the_header_phi() {
        // LLVM's two-register counter: only %r4's φ is fed back itself.
        let two = "mov.u32 %r4, 0;\n$L:\nadd.s64 %r6, %r4, 1;\n\
                   setp.lt.u64 %p1, %r6, %r3;\nmov.b64 %r4, %r6;\n@%p1 bra $L;\nret;";
        assert_eq!(recurrences_of(two, 0), ["%r4: w0 → +1"]);
        // %r2 = %r1 + 2 then %r1 = %r2 - 1: %r1 steps by 1.
        let offset = "mov.u32 %r1, 0;\n$L:\nadd.u32 %r2, %r1, 2;\nadd.s32 %r1, %r2, -1;\n\
                      setp.lt.u32 %p1, %r1, 4;\n@%p1 bra $L;\nret;";
        assert_eq!(recurrences_of(offset, 0), ["%r1: w0 → +1"]);
    }

    #[test]
    fn a_register_step_is_read_where_it_is_added() {
        let body = "mov.u32 %r1, 0;\n$L:\nsub.s32 %r1, %r1, %r9;\n\
                    setp.gt.s32 %p1, %r1, %r2;\n@%p1 bra $L;\nret;";
        assert_eq!(recurrences_of(body, 0), ["%r1: w0 → +0 -%r9@2"]);
    }

    #[test]
    fn two_latches_adding_one_register_step_alike_are_one_recurrence() {
        // blocks: 0 lines 0-1, 1 lines 2-4 (header), 2 lines 5-6, 3 lines
        // 7-10, 4 line 11; each latch adds %r9, written before the loop.
        let body = "mov.u32 %r9, 3;\nmov.u32 %r1, 0;\n$L:\nsetp.lt.u32 %p1, %r1, %r2;\n\
                    @%p1 bra $L__B;\nadd.u32 %r1, %r1, %r9;\nbra $L;\n\
                    $L__B:\nadd.u32 %r1, %r1, %r9;\nsetp.lt.u32 %p2, %r1, 99;\n\
                    @%p2 bra $L;\nret;";
        let found = recurrences_of(body, 0);
        assert_eq!(found.len(), 1);
        assert!(found[0].starts_with("%r1: w1 → +0 +%r9@"), "{found:?}");
    }

    #[test]
    fn a_guarded_or_saturating_or_reversed_update_is_no_recurrence() {
        let with = |update: &str| {
            recurrences_of(
                &format!(
                    "mov.u32 %r1, 0;\n$L:\n{update}\nadd.s32 %r5, %r5, 1;\n\
                     setp.lt.s32 %p1, %r5, 4;\n@%p1 bra $L;\nret;"
                ),
                0,
            )
        };
        // %r5 is always a counter; %r1 only when its update qualifies.
        assert_eq!(
            with("add.s32 %r1, %r1, 1;"),
            ["%r1: w0 → +1", "%r5: undef → +1"]
        );
        assert_eq!(with("@%p2 add.s32 %r1, %r1, 1;"), ["%r5: undef → +1"]);
        assert_eq!(with("add.sat.s32 %r1, %r1, 1;"), ["%r5: undef → +1"]);
        assert_eq!(with("add.f32 %r1, %r1, 1;"), ["%r5: undef → +1"]);
        assert_eq!(with("sub.s32 %r1, 1, %r1;"), ["%r5: undef → +1"]);
        assert_eq!(with("addc.u32 %r1, %r1, 1;"), ["%r5: undef → +1"]);
        assert_eq!(
            with("add.cc.u32 %r1, %r1, 1;"),
            ["%r1: w0 → +1", "%r5: undef → +1"]
        );
    }

    #[test]
    fn an_inner_loops_write_is_no_step_of_the_outer_loop() {
        // blocks: 0 lines 0-1, 1 line 2-3 (outer header), 2 lines 4-7
        // (inner), 3 lines 8-10 (outer latch), 4 line 11
        let body = "mov.u32 %r1, 0;\nmov.u32 %r5, 0;\n$L__O:\nmov.u32 %r6, 0;\n\
                    $L__I:\nadd.s32 %r1, %r1, 1;\nadd.s32 %r6, %r6, 1;\n@%p1 bra $L__I;\n\
                    add.s32 %r5, %r5, 1;\nsetp.lt.u32 %p2, %r5, 8;\n@%p2 bra $L__O;\nret;";
        assert_eq!(recurrences_of(body, 0), ["%r5: w1 → +1"]);
        assert_eq!(recurrences_of(body, 1), ["%r1: phi1 → +1", "%r6: w3 → +1"]);
    }

    #[test]
    fn entries_that_disagree_on_the_start_are_no_recurrence() {
        let body = "mov.u32 %r1, 0;\n@%p2 bra $L;\nmov.u32 %r1, 5;\n\
                    $L:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, 9;\n@%p1 bra $L;\nret;";
        assert_eq!(recurrences_of(body, 0), Vec::<String>::new());
    }
}
