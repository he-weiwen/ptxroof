//! Which registers an instruction writes.

use crate::ptx::ir::{Instr, Module, Opcode, Operand, OperandId};
use crate::support::intern::Symbol;

/// How many of `instr`'s leading operands it writes, `None` for a form
/// not decoded. In every instruction's Syntax in PTX ISA 9.3 §9.7, the
/// written operands precede the read ones.
pub fn written_operands(_module: &Module, instr: &Instr) -> Option<usize> {
    use Opcode::*;
    match instr.mnemonic {
        Abs | Activemask | Add | Addc | Alloca | And | Atom | Bfe | Bfi | Bfind | Bmsk | Brev
        | Clmad | Clz | Cnot | Copysign | Cos | Createpolicy | Cvt | Cvta | Div | Dp2a | Dp4a
        | Elect | Ex2 | Fma | Fns | Getctarank | Isspacep | Ld | Ldmatrix | Ldu | Lg2 | Lop3
        | Mad | Mad24 | Madc | Mapa | Match | Max | Min | Mov | Movmatrix | Mma | Mul | Mul24
        | Neg | Not | Or | Popc | Prmt | Rcp | Redux | Rem | Rsqrt | Sad | Selp | Set | Setp
        | Shf | Shfl | Shl | Shr | Sin | Slct | Sqrt | Stacksave | Sub | Subc | Szext | Tanh
        | Testp | Vote | Xor => Some(1),
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
}
