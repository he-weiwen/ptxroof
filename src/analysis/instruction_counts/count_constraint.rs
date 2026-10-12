use crate::ptx::ir::{Instr, Module};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountConstraint {
    Preserved,
    Constrained,
    Free,
}

pub fn count_constraint(module: &Module, instr: &Instr) -> CountConstraint {
    let mnemonic = instr.mnemonic.name();
    let mods: Vec<&str> = module
        .modifiers(instr)
        .iter()
        .map(|&m| module.interner.resolve(m))
        .collect();
    let has = |names: &[&str]| mods.iter().any(|m| names.contains(m));
    // PTX ISA §8.4.1: mmio reads and writes are "always performed" and
    // "never combined"; §8.4.2: "The number of volatile instructions
    // (not operations) executed by a program is preserved."
    if has(&["volatile", "mmio"]) {
        return CountConstraint::Preserved;
    }
    // §8.4: strong and synchronizing operations. atom and red default to
    // .relaxed (§9.7.14.5, §9.7.14.6), fence to .acq_rel and membar is
    // fence.sc (§9.7.14.4), except their .proxy forms; mbarrier.arrive and
    // arrive_drop default to .release, test_wait and try_wait to .acquire
    // (§9.7.14.16.16, §9.7.14.16.17, §9.7.14.16.19).
    let fence = matches!(mnemonic, "fence" | "membar") && !has(&["proxy"]);
    let mbarrier =
        mnemonic == "mbarrier" && has(&["arrive", "arrive_drop", "test_wait", "try_wait"]);
    let strong = has(&["relaxed", "acquire", "release", "acq_rel"])
        || matches!(mnemonic, "atom" | "red" | "bar" | "barrier")
        || fence
        || mbarrier;
    // §9.7.3.3–§9.7.5.2: add, sub and mul "with an explicit rounding
    // modifier [are] treated conservatively by the code optimizer";
    // §9.7.3.6, §9.7.3.7: fma and mad round once, by .rnd.
    let rounded =
        matches!(mnemonic, "add" | "sub" | "mul" | "fma" | "mad") && has(&["rn", "rz", "rm", "rp"]);
    if strong || rounded {
        CountConstraint::Constrained
    } else {
        CountConstraint::Free
    }
}

#[cfg(test)]
mod tests {
    use super::CountConstraint::{Constrained, Free, Preserved};
    use super::*;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;

    fn constraint_of(text: &str) -> CountConstraint {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{text}\n}}\n"
        );
        let m = parse(&src).expect("snippet parses");
        let instr = m.kernels[0]
            .stmts
            .iter()
            .find_map(|s| match s {
                Stmt::Instr(i) => Some(*i),
                _ => None,
            })
            .expect("one instruction");
        count_constraint(&m, &instr)
    }

    #[test]
    fn volatile_and_mmio_counts_are_preserved() {
        assert_eq!(
            constraint_of("ld.volatile.global.u32 %r1, [%rd1];"),
            Preserved
        );
        assert_eq!(
            constraint_of("st.volatile.shared.u32 [%r1], %r2;"),
            Preserved
        );
        assert_eq!(
            constraint_of("st.mmio.relaxed.sys.global.u32 [%rd1], %r1;"),
            Preserved
        );
    }

    #[test]
    fn explicitly_rounded_and_fused_arithmetic_is_constrained() {
        for text in [
            "add.rn.f32 %f1, %f2, %f3;",
            "sub.rz.f64 %fd1, %fd2, %fd3;",
            "mul.rm.f16 %rs1, %rs2, %rs3;",
            "fma.rn.f32 %f1, %f2, %f3, %f4;",
            "mad.rp.f64 %fd1, %fd2, %fd3, %fd4;",
        ] {
            assert_eq!(constraint_of(text), Constrained, "{text}");
        }
        for text in [
            "add.f32 %f1, %f2, %f3;",
            "mul.f32 %f1, %f2, %f3;",
            "add.s32 %r1, %r2, %r3;",
            "mad.lo.s32 %r1, %r2, %r3, %r4;",
            "mul.wide.u32 %rd1, %r1, 4;",
            "cvt.rn.f16.f32 %rs1, %f1;",
        ] {
            assert_eq!(constraint_of(text), Free, "{text}");
        }
    }

    #[test]
    fn strong_and_synchronizing_operations_are_constrained() {
        for text in [
            "ld.relaxed.gpu.global.u32 %r1, [%rd1];",
            "ld.acquire.gpu.global.u32 %r1, [%rd1];",
            "st.release.gpu.global.u32 [%rd1], %r1;",
            "atom.global.add.u32 %r1, [%rd1], 1;",
            "red.global.add.u32 [%rd1], 1;",
            "bar.sync 0;",
            "barrier.sync 0;",
            "membar.gl;",
            "fence.sc.gpu;",
            "fence.gpu;",
            "mbarrier.arrive.shared::cta.b64 %rd1, [%r1];",
            "mbarrier.arrive.release.cta.shared::cta.b64 %rd1, [%r1];",
            "mbarrier.arrive_drop.shared.b64 %rd1, [%r1];",
            "mbarrier.test_wait.parity.shared::cta.b64 %p1, [%r1], %r2;",
            "mbarrier.try_wait.acquire.cta.shared::cta.b64 %p1, [%r1], %rd1;",
        ] {
            assert_eq!(constraint_of(text), Constrained, "{text}");
        }
        for text in [
            "ld.global.u32 %r1, [%rd1];",
            "ld.weak.global.u32 %r1, [%rd1];",
            "st.shared.u16 [%r1], %rs1;",
            "membar.proxy.alias;",
            "fence.proxy.alias;",
            "mbarrier.init.shared::cta.b64 [%r1], %r2;",
        ] {
            assert_eq!(constraint_of(text), Free, "{text}");
        }
    }
}
