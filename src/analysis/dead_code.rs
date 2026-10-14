//! Which instructions that run have results nothing with an effect
//! reads, directly or through other instructions: LLVM's aggressive dead
//! code elimination (ADCE) over PTX registers. Every instruction that
//! runs is live unless it is pure: integer, predicate, conversion or move
//! arithmetic, a CUDA-core or SFU flop, or a weak load (PTX ISA §9.7.9.8:
//! "a memory instruction with no synchronization", the default). Liveness
//! follows each read of a live instruction to the definitions that may
//! reach it; a guarded definition also reads the value it keeps where its
//! guard is false, unless every thread reads the guard as true, and one
//! that never runs reads only that.

use crate::analysis::instruction_counts::classify::{InstructionCategory, Pipe, classify};
use crate::analysis::instruction_counts::measurement::MeasureKind;
use crate::analysis::register_effects::{reads_cc, registers, split_operands, writes};
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Register, Stmt};
use crate::ssa::{RegisterSsa, Value};

/// Per statement, whether it is an instruction that runs and is not
/// live. `guard` tells, per statement, `Some(false)` when it never runs,
/// `Some(true)` when every thread that reaches it reads its guard as
/// true, `None` when not known.
pub fn unused(
    module: &Module,
    kernel: &Kernel,
    ssa: &RegisterSsa,
    guard: impl Fn(usize) -> Option<bool>,
) -> Vec<bool> {
    let n = kernel.stmts.len();
    let mut work: Vec<usize> = (0..n)
        .filter(|&s| match &kernel.stmts[s] {
            Stmt::Instr(i) => guard(s) != Some(false) && !pure(module, i),
            Stmt::Unparsed { .. } => true,
            _ => false,
        })
        .collect();
    let mut live = vec![false; n];
    work.iter().for_each(|&s| live[s] = true);
    while let Some(s) = work.pop() {
        for reg in reads(module, &kernel.stmts[s], guard(s)) {
            for v in ssa
                .value_before(s, reg)
                .map(|v| ssa.sources(v))
                .unwrap_or_default()
            {
                if let Value::Result { op, .. } = v
                    && !live[op]
                {
                    live[op] = true;
                    work.push(op);
                }
            }
        }
    }
    (0..n)
        .map(|s| matches!(kernel.stmts[s], Stmt::Instr(_)) && guard(s) != Some(false) && !live[s])
        .collect()
}

fn pure(module: &Module, instr: &Instr) -> bool {
    let c = classify(module, instr);
    match c.category {
        InstructionCategory::NonFlopArith { .. } => !matches!(
            instr.mnemonic,
            Opcode::Stacksave | Opcode::Stackrestore | Opcode::Alloca
        ),
        InstructionCategory::Flop => c.contributions.iter().all(|k| {
            matches!(
                k.kind,
                MeasureKind::Flops {
                    pipe: Pipe::CudaCore | Pipe::Sfu,
                    ..
                }
            )
        }),
        _ => {
            instr.mnemonic == Opcode::Ld
                && !(module.modifiers(instr).iter()).any(|&m| {
                    matches!(
                        module.interner.resolve(m),
                        "volatile" | "relaxed" | "acquire" | "mmio"
                    )
                })
        }
    }
}

/// The registers statement `stmt`, whose guard reads as `guard`, reads:
/// a guarded one also the values it keeps, and only those when it never
/// runs.
fn reads(module: &Module, stmt: &Stmt, guard: Option<bool>) -> Vec<Register> {
    match stmt {
        Stmt::Instr(i) => {
            let mut regs = match i.predicate {
                Some(_) if guard != Some(true) => writes(module, stmt),
                _ => Vec::new(),
            };
            if guard != Some(false) {
                let (_, read) = split_operands(module, i);
                regs.extend(
                    read.iter()
                        .flat_map(|&o| registers(module, o))
                        .map(Register::Named),
                );
                regs.extend(i.predicate.map(|p| Register::Named(p.reg)));
                regs.extend(reads_cc(module, i).then_some(Register::Cc));
            }
            regs
        }
        Stmt::Unparsed { registers, .. } => registers.iter().map(|&r| Register::Named(r)).collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::analysis::scalar::trace::AffineValueTracer;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    /// The unused instructions of `body`, as text; `decided` gives the
    /// guard of the statements with those opcodes.
    fn unused_in(body: &str, decided: &[(&str, bool)]) -> Vec<String> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u64 k_param_0\n)\n{{\n\
             ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n{body}\nret;\n}}\n"
        );
        let m = parse(&src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let t = AffineValueTracer::new(&m, k, &cfg, &forest);
        let text = |s: usize| match &k.stmts[s] {
            Stmt::Instr(i) => m.opcode(i),
            _ => String::new(),
        };
        let guard = |s: usize| decided.iter().find(|d| text(s) == d.0).map(|d| d.1);
        let flags = unused(&m, k, &t.ssa, guard);
        (0..flags.len()).filter(|&s| flags[s]).map(text).collect()
    }

    #[test]
    fn arithmetic_and_weak_loads_nothing_reads_are_unused() {
        let body = "add.s32 %r2, %r1, 1;\nmul.lo.s32 %r3, %r1, 2;\n\
                    ld.global.u32 %r4, [%rd1];\nld.volatile.global.u32 %r5, [%rd1];\n\
                    st.global.u32 [%rd1], %r2;";
        assert_eq!(unused_in(body, &[]), ["mul.lo.s32", "ld.global.u32"]);
    }

    #[test]
    fn a_guarded_write_keeps_its_old_value_and_guard_live() {
        let body = "setp.lt.u32 %p1, %r1, 16;\nmov.u32 %r2, 7;\n@%p1 mov.u32 %r2, %r1;\n\
                    st.global.u32 [%rd1], %r2;";
        assert_eq!(unused_in(body, &[]), Vec::<String>::new());
    }

    #[test]
    fn a_load_that_never_runs_reads_only_the_value_it_keeps() {
        let body = "mov.u32 %r2, %r1;\nadd.s64 %rd2, %rd1, 64;\nmov.pred %p1, 0;\n\
                    @%p1 ld.global.u32 %r2, [%rd2];\nst.global.u32 [%rd1], %r2;";
        assert_eq!(
            unused_in(body, &[("ld.global.u32", false)]),
            ["add.s64", "mov.pred"]
        );
    }

    #[test]
    fn a_counter_read_only_by_itself_is_unused() {
        let body = "mov.u32 %r2, 0;\nmov.u32 %r3, 0;\n$L__L:\nadd.s32 %r2, %r2, 1;\n\
                    add.s32 %r3, %r3, 1;\nsetp.lt.u32 %p1, %r3, 8;\n@%p1 bra $L__L;\n\
                    st.global.u32 [%rd1], %r1;";
        assert_eq!(unused_in(body, &[]), ["mov.u32", "add.s32"]);
    }

    #[test]
    fn a_guard_every_thread_reads_as_true_keeps_no_old_value() {
        let body = "mov.u32 %r2, 0;\nmov.pred %p1, -1;\n@%p1 mov.u32 %r2, %r1;\n\
                    st.global.u32 [%rd1], %r2;";
        assert_eq!(unused_in(body, &[]), Vec::<String>::new());
        assert_eq!(unused_in(body, &[("mov.u32", true)]), ["mov.u32"]);
    }
}
