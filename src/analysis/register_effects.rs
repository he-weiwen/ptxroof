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
        Multimem => match modifiers.next()? {
            "ld_reduce" => Some(1),
            "st" | "red" | "cp" => Some(0),
            _ => None,
        },
        Clusterlaunchcontrol => match modifiers.next()? {
            "query_cancel" => Some(1),
            "try_cancel" => Some(0),
            _ => None,
        },
        Fabric => match modifiers.next()? {
            "try_get" | "try_put" | "try_red" | "try_pullred" | "submit" | "wait" => Some(0),
            _ => None,
        },
        Abs | Activemask | Add | Addc | Alloca | And | Atom | Bfe | Bfi | Bfind | Bmsk | Brev
        | Clmad | Clz | Cnot | Copysign | Cos | Createpolicy | Cvt | Cvta | Div | Dp2a | Dp4a
        | Elect | Ex2 | Fma | Fns | Getctarank | Isspacep | Ld | Ldmatrix | Ldu | Lg2 | Lop3
        | Mad | Mad24 | Madc | Mapa | Match | Max | Min | Mov | Movmatrix | Mma | Mul | Mul24
        | Neg | Not | Or | Popc | Prmt | Rcp | Redux | Rem | Rsqrt | Sad | Selp | Set | Setp
        | Shf | Shfl | Shl | Shr | Sin | Slct | Sqrt | Stacksave | Sub | Subc | Szext | Tanh
        | Testp | Vote | Xor => Some(1),
        Istypep | Suld | Suq | Tex | Tld4 | Txq => Some(1),
        Vabsdiff | Vabsdiff2 | Vabsdiff4 | Vadd | Vadd2 | Vadd4 | Vavrg2 | Vavrg4 | Vmad | Vmax
        | Vmax2 | Vmax4 | Vmin | Vmin2 | Vmin4 | Vset | Vset2 | Vset4 | Vshl | Vshr | Vsub
        | Vsub2 | Vsub4 => Some(1),
        Applypriority | Bra | Brkpt | Brx | Cp | Discard | Exit | Fence | Griddepcontrol
        | Membar | Nanosleep | Pmevent | Prefetch | Prefetchu | Red | Ret | Setmaxnreg | St
        | Stackrestore | Stmatrix | Sured | Sust | Tensormap | Trap => Some(0),
        _ => None,
    }
}

/// Whether `instr` writes the carry flag CC.CF, and whether it reads it:
/// `.cc` writes it, `addc`, `subc` and `madc` read it, and "no other
/// instructions access the condition code" (PTX ISA §9.7.2); a form not
/// decoded may do both.
pub(crate) fn carry(module: &Module, instr: &Instr) -> (bool, bool) {
    use Opcode::*;
    if written_operands(module, instr).is_none() {
        return (true, true);
    }
    let cc = (module.modifiers(instr).iter()).any(|&m| module.interner.resolve(m) == "cc");
    match instr.mnemonic {
        Add | Sub | Mad => (cc, false),
        Addc | Subc | Madc => (cc, true),
        _ => (false, false),
    }
}

/// `instr`'s written operands, and the operands after them, which it
/// reads; a form not decoded is taken to write and read all of them.
pub(crate) fn split_operands<'m>(
    module: &'m Module,
    instr: &Instr,
) -> (&'m [OperandId], &'m [OperandId]) {
    let ops = module.operand_ids(instr.operands);
    match written_operands(module, instr) {
        Some(n) => ops.split_at(n.min(ops.len())),
        None => (ops, ops),
    }
}

/// The registers `instr` writes.
pub(crate) fn written_registers(module: &Module, instr: &Instr) -> Vec<Symbol> {
    (split_operands(module, instr).0.iter())
        .flat_map(|&op| registers(module, op))
        .collect()
}

/// The registers operand `op` names: itself, the elements of a `{…}` or
/// `p|q`, as in `tex`'s `{…}|p`, or an address's base.
pub(crate) fn registers(module: &Module, op: OperandId) -> Vec<Symbol> {
    match module.operand(op) {
        Operand::Register(r) => vec![*r],
        Operand::MultipleDestinations { children } | Operand::VectorList { children } => {
            (module.operand_ids(*children).iter())
                .flat_map(|&c| registers(module, c))
                .collect()
        }
        Operand::Memory { base, .. } => registers(module, *base),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;

    /// The registers `instr` writes, by name, and whether it is decoded.
    fn written_by(instr: &str) -> (bool, String) {
        let src = format!(
            ".version 8.7\n.target sm_90\n.address_size 64\n\
             .visible .entry k()\n{{\n{instr}\nret;\n}}\n"
        );
        let m = parse(&src).expect("test source parses");
        let Stmt::Instr(i) = &m.kernels[0].stmts[0] else {
            panic!("{instr}: not an instruction");
        };
        let names = (written_registers(&m, i).into_iter())
            .map(|r| m.interner.resolve(r).to_owned())
            .collect::<Vec<_>>();
        (written_operands(&m, i).is_some(), names.join(" "))
    }

    /// The registers a decoded `instr` writes, by name.
    fn written(instr: &str) -> Option<String> {
        let (decoded, regs) = written_by(instr);
        decoded.then_some(regs)
    }

    #[test]
    fn a_form_not_decoded_writes_every_register_it_names() {
        assert_eq!(
            written_by("mbarrier.wait.b64 %p1, [%r2], %rd3;"),
            (false, "%p1 %r2 %rd3".to_owned())
        );
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

    #[test]
    fn multimem_writes_only_when_it_loads() {
        for (instr, regs) in [
            (
                "multimem.ld_reduce.relaxed.gpu.global.add.u32 %r1, [%rd1];",
                Some("%r1"),
            ),
            ("multimem.st.relaxed.gpu.global.u32 [%rd1], %r2;", Some("")),
            (
                "multimem.red.relaxed.gpu.global.add.u32 [%rd1], %r2;",
                Some(""),
            ),
            (
                "multimem.red.async.release.gpu.global.add.u32 [%rd1], %r2;",
                Some(""),
            ),
            (
                "multimem.cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r2], 128;",
                Some(""),
            ),
            ("multimem.ld.global.u32 %r1, [%rd1];", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }

    #[test]
    fn a_cluster_launch_query_writes_its_first_operand() {
        for (instr, regs) in [
            (
                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p1, %rq1;",
                Some("%p1"),
            ),
            (
                "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 \
                 {%r1, %r2, %r3, _}, %rq1;",
                Some("%r1 %r2 %r3"),
            ),
            (
                "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 \
                 [%r1], [%r2];",
                Some(""),
            ),
            ("clusterlaunchcontrol.cancel.b128 [%r1];", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }

    #[test]
    fn textures_and_surfaces_write_their_first_operand_unless_they_store() {
        for (instr, regs) in [
            (
                "tex.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}|%p1, [tex1, {%f5, %f6}];",
                "%f1 %f2 %f3 %f4 %p1",
            ),
            (
                "tld4.r.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [tex1, {%f5, %f6}];",
                "%f1 %f2 %f3 %f4",
            ),
            ("txq.level.width.b32 %r1, [tex1], %r2;", "%r1"),
            ("istypep.texref %p1, %rd2;", "%p1"),
            (
                "suld.b.1d.v2.b32.trap {%r1, %r2}, [surf1, {%r3}];",
                "%r1 %r2",
            ),
            ("suq.width.b32 %r1, [surf1];", "%r1"),
            ("sust.b.1d.v2.b32.trap [surf1, {%r3}], {%r1, %r2};", ""),
            ("sured.b.add.2d.u32.trap [surf1, {%r3, %r4}], %r1;", ""),
        ] {
            assert_eq!(written(instr).as_deref(), Some(regs), "{instr}");
        }
    }

    #[test]
    fn video_instructions_write_their_first_operand() {
        for instr in [
            "vabsdiff.u32.u32.u32 %r1, %r2, %r3;",
            "vadd.s32.u32.s32.sat.add %r1, %r2, %r3, %r4;",
            "vshl.u32.u32.u32.clamp %r1, %r2, %r3;",
            "vmad.s32.s32.u32.sat %r1, %r2, %r3, %r4;",
            "vset.s32.u32.lt %r1, %r2, %r3;",
            "vadd2.s32.s32.u32 %r1, %r2, %r3, %r4;",
            "vavrg4.u32.u32.u32 %r1, %r2, %r3, %r4;",
            "vset4.u32.u32.ne %r1, %r2, %r3, %r4;",
        ] {
            assert_eq!(written(instr).as_deref(), Some("%r1"), "{instr}");
        }
    }

    #[test]
    fn fabric_writes_nothing() {
        for (instr, regs) in [
            (
                "fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.\
                 mbarrier::report::fabric.relaxed.sys.b128 [%r1], [%r2, %rd3], 256, [%r4];",
                Some(""),
            ),
            (
                "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.\
                 mbarrier::report::fabric.relaxed.sys.b128 [%r2, %rd3], [%r1], 256, [%r4];",
                Some(""),
            ),
            ("fabric.submit;", Some("")),
            ("fabric.wait.sync_restrict::reads;", Some("")),
            ("fabric.get [%r1];", None),
        ] {
            assert_eq!(written(instr).as_deref(), regs, "{instr}");
        }
    }
}
