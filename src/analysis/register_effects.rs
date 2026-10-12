//! Which registers an instruction writes.

use crate::ptx::ir::{Instr, Module, Opcode, Operand, OperandId};
use crate::support::intern::Symbol;

/// How many of `instr`'s leading operands it writes, `None` for a form
/// not decoded. In every instruction's Syntax in PTX ISA 9.3 §9.7, the
/// written operands precede the read ones.
pub fn written_operands(module: &Module, instr: &Instr) -> Option<usize> {
    use Opcode::*;
    let mut modifiers = (module.modifiers(instr).iter()).map(|&m| module.interner.resolve(m));
    match instr.mnemonic {
        Bar | Barrier => match modifiers.find(|&m| m != "cta")? {
            "red" => Some(1),
            "sync" | "arrive" | "warp" | "cluster" => Some(0),
            _ => None,
        },
        Mbarrier => match modifiers.next()? {
            "init" | "inval" | "expect_tx" | "complete_tx" => Some(0),
            "arrive" | "arrive_drop" | "pending_count" | "check_layout" => Some(1),
            "test_wait" | "try_wait" => (module.operand_ids(instr.operands).iter())
                .position(|&op| matches!(module.operand(op), Operand::Memory { .. })),
            _ => None,
        },
        Tcgen05 => match modifiers.next()? {
            "ld" => Some(1 + usize::from(modifiers.next() == Some("red"))),
            "alloc"
            | "dealloc"
            | "relinquish_alloc_permit"
            | "st"
            | "wait::ld"
            | "wait::st"
            | "cp"
            | "shift"
            | "mma"
            | "fence::before_thread_sync"
            | "fence::after_thread_sync"
            | "commit" => Some(0),
            _ => None,
        },
        Wgmma => match modifiers.next()? {
            "mma_async" => Some(1),
            "fence" | "commit_group" | "wait_group" => Some(0),
            _ => None,
        },
        Wmma => match modifiers.next()? {
            "load" | "mma" => Some(1),
            "store" => Some(0),
            _ => None,
        },
        Abs | Activemask | Add | Addc | Alloca | And | Atom | Bfe | Bfi | Bfind | Bmsk | Brev
        | Clmad | Clz | Cnot | Copysign | Cos | Createpolicy | Cvt | Cvta | Div | Dp2a | Dp4a
        | Elect | Ex2 | Fma | Fns | Getctarank | Isspacep | Ld | Ldmatrix | Ldu | Lg2 | Lop3
        | Mad | Mad24 | Madc | Mapa | Match | Max | Min | Mov | Movmatrix | Mma | Mul | Mul24
        | Neg | Not | Or | Popc | Prmt | Rcp | Redux | Rem | Rsqrt | Sad | Selp | Set | Setp
        | Shf | Shfl | Shl | Shr | Sin | Slct | Sqrt | Stacksave | Sub | Subc | Szext | Tanh
        | Testp | Vote | Xor => Some(1),
        Applypriority | Bra | Brkpt | Brx | Cp | Discard | Exit | Fence | Griddepcontrol
        | Membar | Nanosleep | Pmevent | Prefetch | Prefetchu | Red | Ret | Setmaxnreg | St
        | Stackrestore | Stmatrix | Tensormap | Trap => Some(0),
        _ => None,
    }
}

/// The registers operand `op` names: itself, or the elements of a `{…}`
/// or `p|q`.
pub(crate) fn registers(module: &Module, op: OperandId) -> Vec<Symbol> {
    let ops = match module.operand(op) {
        Operand::MultipleDestinations { children } | Operand::VectorList { children } => {
            module.operand_ids(*children)
        }
        _ => std::slice::from_ref(&op),
    };
    (ops.iter())
        .filter_map(|&id| match module.operand(id) {
            Operand::Register(r) => Some(*r),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;

    /// The registers `instr`'s written operands name, by name.
    fn written(instr: &str) -> Option<String> {
        let src = format!(
            ".version 8.7\n.target sm_90\n.address_size 64\n\
             .visible .entry k()\n{{\n{instr}\nret;\n}}\n"
        );
        let m = parse(&src).expect("test source parses");
        let Stmt::Instr(i) = &m.kernels[0].stmts[0] else {
            panic!("{instr}: not an instruction");
        };
        let n = written_operands(&m, i)?;
        let regs = (m.operand_ids(i.operands).iter().take(n))
            .flat_map(|&op| registers(&m, op))
            .map(|r| m.interner.resolve(r).to_owned());
        Some(regs.collect::<Vec<_>>().join(" "))
    }

    #[test]
    fn keywords_outside_the_corpus_write_their_first_operand() {
        for (instr, regs) in [
            ("activemask.b32 %r1;", "%r1"),
            ("addc.cc.u32 %r1, %r2, %r3;", "%r1"),
            ("alloca.u32 %r1, %r2, 8;", "%r1"),
            ("bfind.shiftamt.s64 %r1, %rd2;", "%r1"),
            ("bmsk.clamp.b32 %r1, %r2, %r3;", "%r1"),
            ("brev.b32 %r1, %r2;", "%r1"),
            ("clmad.lo.u64 %rd1, %rd2, %rd3, %rd4;", "%rd1"),
            ("clz.b64 %r1, %rd2;", "%r1"),
            ("cnot.b32 %r1, %r2;", "%r1"),
            ("copysign.f32 %f1, %f2, %f3;", "%f1"),
            ("cos.approx.ftz.f32 %f1, %f2;", "%f1"),
            ("dp2a.lo.u32.u32 %r1, %r2, %r3, %r4;", "%r1"),
            ("dp4a.u32.s32 %r1, %r2, %r3, %r4;", "%r1"),
            ("fns.b32 %r1, %r2, %r3, %r4;", "%r1"),
            ("getctarank.shared::cluster.u32 %r1, %r2;", "%r1"),
            ("isspacep.global %p1, %rd2;", "%p1"),
            (
                "ldu.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];",
                "%f1 %f2 %f3 %f4",
            ),
            ("lg2.approx.ftz.f32 %f1, %f2;", "%f1"),
            ("lop3.or.b32 %r1|%p1, %r2, %r3, %r4, 0x40, %p2;", "%r1 %p1"),
            ("lop3.and.b32 _|%p1, %r2, %r3, %r4, 0x40, %p2;", "%p1"),
            ("mad24.lo.s32 %r1, %r2, %r3, %r4;", "%r1"),
            ("madc.hi.cc.u32 %r1, %r2, %r3, %r4;", "%r1"),
            ("mapa.shared::cluster.u32 %r1, %r2, %r3;", "%r1"),
            ("match.all.sync.b64 %r1|%p1, %rd2, %r3;", "%r1 %p1"),
            ("match.any.sync.b32 %r1, %r2, 0xffffffff;", "%r1"),
            ("movmatrix.sync.aligned.m8n8.trans.b16 %r1, %r2;", "%r1"),
            ("mul24.lo.s32 %r1, %r2, %r3;", "%r1"),
            ("popc.b64 %r1, %rd2;", "%r1"),
            ("rcp.rn.ftz.f32 %f1, %f2;", "%f1"),
            ("sad.u32 %r1, %r2, %r3, %r1;", "%r1"),
            ("set.eq.u32.u32 %r1, %r2, %r3;", "%r1"),
            ("shf.l.clamp.b32 %r1, %r2, %r3, 16;", "%r1"),
            ("sin.approx.ftz.f32 %f1, %f2;", "%f1"),
            ("slct.u32.s32 %r1, %r2, %r3, %r4;", "%r1"),
            ("stacksave.u32 %r1;", "%r1"),
            ("subc.cc.u32 %r1, %r2, %r3;", "%r1"),
            ("szext.clamp.s32 %r1, %r2, %r3;", "%r1"),
            ("tanh.approx.f32 %f1, %f2;", "%f1"),
            ("testp.notanumber.f32 %p1, %f2;", "%p1"),
            ("vote.ballot.b32 %r1, %p2;", "%r1"),
        ] {
            assert_eq!(written(instr).as_deref(), Some(regs), "{instr}");
        }
    }

    #[test]
    fn keywords_outside_the_corpus_write_nothing() {
        for instr in [
            "applypriority.global.L2::evict_normal [%rd1], 128;",
            "brkpt;",
            "discard.global.L2 [%rd1], 128;",
            "exit;",
            "membar.gl;",
            "pmevent.mask 7;",
            "prefetch.global.L2::evict_last [%rd1];",
            "prefetchu.L1 [%rd1];",
            "stackrestore.u32 %r1;",
            "tensormap.replace.tile.global_address.global.b1024.b64 [%rd1], %rd2;",
            "trap;",
        ] {
            assert_eq!(written(instr).as_deref(), Some(""), "{instr}");
        }
    }

    #[test]
    fn a_barrier_writes_its_first_operand_when_it_reduces() {
        for (instr, regs) in [
            ("bar.red.popc.u32 %r1, 0, %p1;", Some("%r1")),
            ("bar.cta.red.and.pred %p1, 0, 64, %p2;", Some("%p1")),
            (
                "barrier.cta.red.or.aligned.pred %p1, %r2, %p2;",
                Some("%p1"),
            ),
            ("bar.arrive 1, %r1;", Some("")),
            ("barrier.cta.sync.aligned %r1, 64;", Some("")),
            ("barrier.cluster.arrive.release.aligned;", Some("")),
            ("barrier.cluster.wait;", Some("")),
            ("bar.warp.sync %r1;", Some("")),
            ("barrier.cta;", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }

    #[test]
    fn an_mbarrier_wait_writes_the_operands_before_its_address() {
        for (instr, regs) in [
            (
                "mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 \
                 %p1|%p2, %r3, [%r4], %r5;",
                Some("%p1 %p2 %r3"),
            ),
            (
                "mbarrier.try_wait.phase_type::primary.shared::cta.b64 \
                 %p1|%p2, [%r4], %rd5, %r6;",
                Some("%p1 %p2"),
            ),
            (
                "mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r4], %r5, %r6;",
                Some("%p1"),
            ),
            (
                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r2];",
                Some(""),
            ),
            ("mbarrier.arrive_drop.shared.b64 %rd1, [%r2];", Some("%rd1")),
            ("mbarrier.pending_count.b64 %r1, %rd2;", Some("%r1")),
            (
                "mbarrier.check_layout.layout::v1.shared::cta.b64 %p1, [%r2];",
                Some("%p1"),
            ),
            (
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r1], 64;",
                Some(""),
            ),
            (
                "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%r1], 64;",
                Some(""),
            ),
            ("mbarrier.test_wait.b64 %p1, %rd2;", None),
            ("mbarrier.wait.b64 %p1, [%r2];", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }

    #[test]
    fn tcgen05_writes_registers_only_when_it_loads() {
        for (instr, regs) in [
            (
                "tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max {%r1, %r2}, %r3, [%r4], 16;",
                Some("%r1 %r2 %r3"),
            ),
            ("tcgen05.shift.cta_group::1.down [%r1];", Some("")),
            ("tcgen05.fence::before_thread_sync;", Some("")),
            ("tcgen05.fence::after_thread_sync;", Some("")),
            ("tcgen05.wait::xx.sync.aligned;", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }

    #[test]
    fn wgmma_writes_its_accumulator() {
        for (instr, regs) in [
            (
                "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 \
                 {%f1, %f2, %f3, %f4}, %rd1, %rd2, 1, 1, 1, 0, 0;",
                Some("%f1 %f2 %f3 %f4"),
            ),
            (
                "wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16 \
                 {%f1, %f2, %f3, %f4}, %rd1, %rd2, %r1, 0, 1, 1, 1, 0, 0;",
                Some("%f1 %f2 %f3 %f4"),
            ),
            ("wgmma.commit_group.sync.aligned;", Some("")),
            ("wgmma.wait_group.sync.aligned 0;", Some("")),
            ("wgmma.mma.sync.aligned;", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }

    #[test]
    fn wmma_writes_unless_it_stores() {
        for (instr, regs) in [
            (
                "wmma.load.c.sync.aligned.row.m16n16k16.shared.f32 {%f1, %f2}, [%r1], 16;",
                Some("%f1 %f2"),
            ),
            (
                "wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd1], {%f1, %f2};",
                Some(""),
            ),
            ("wmma.fill.sync.aligned %r1;", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }
}
