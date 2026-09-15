//! Trip-count matcher for the nvcc loop shapes.
//!
//! Per loop: extract the latch condition (`setp` + `@%p bra`),
//! recognize the induction variable (the loop's single in-loop
//! `add r, r, const` definition), and normalize the latch condition as
//! an affine expression of the IV and loop invariants with a scalar
//! affine tracer — real nvcc latches compare *derived* registers, not
//! the IV (k2 exits on `setp.ne.s32 %p6, %r29, 0` where
//! `%r29 = %r7 + %r35`, `%r7 = (K&3) − K`). The tracer walks reaching
//! definitions through `mov/add/sub/and-mask/shl/mul/mad/cvt` down to
//! `ld.param`/constants, following the dominator chain for
//! loop-invariant values.
//!
//! With the latch normalized to `continue while A1·k + A0  cmp  0`
//! (k = iteration number; a counter read after its increment is
//! init + k·step, read before it init + (k − 1)·step), trips
//! solve to:  `ne` → −A0/A1 (exact division — compiler-generated
//! not-equal latches step in divisor units);  `lt` → ceildiv(−A0, A1);
//! `le` → floordiv(−A0, A1) + 1;  `gt`/`ge` → mirrored.
//!
//! Anything else — multiple exits, exits not at the latch, latch
//! values loaded in the loop, special-register dependence, two IVs in
//! one condition — degrades to an `Err(reason)`: a *named* unknown,
//! never a guess (honesty principle). The symbols are assumed
//! nonnegative, and unsigned/signed comparison width is deliberately
//! not modeled (documented domain assumption).
//!
//! The kernel-level pass links nvcc's unroll main+remainder pair —
//! sibling loops on the same source line whose trips match
//! `(X − X mod c)/c` and `X mod c` — into one logical loop with
//! factor c.

use crate::affine::Var;
use crate::cfg::loops::{LoopForest, LoopId};
use crate::cfg::naming::LoopName;
use crate::cfg::{BlockId, Cfg};
use crate::core::symexpr::SymExpr;
use crate::core::{Kernel, Module, Operand, Stmt, Symbol};
use crate::parse::parser::parse_int;
use crate::tracer::{Reach, Tracer};

/// Trip count of one loop: an expression, or a named reason there
/// isn't one.
pub type TripCount = Result<SymExpr, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnrollPair {
    pub main: LoopId,
    pub remainder: LoopId,
    pub factor: i64,
}

#[derive(Debug)]
pub struct TripInfo {
    /// Indexed by `LoopId`.
    pub trips: Vec<TripCount>,
    pub unroll_pairs: Vec<UnrollPair>,
}

pub fn trip_counts(
    module: &Module,
    kernel: &Kernel,
    cfg: &Cfg,
    forest: &LoopForest,
    names: &[LoopName],
) -> TripInfo {
    let tracer = Tracer::new(module, kernel, cfg, forest);
    let trips: Vec<TripCount> = (0..forest.loops.len() as u32)
        .map(|i| tracer.loop_trips(LoopId(i)))
        .collect();

    // -- unroll main+remainder linking ----------------------------------
    let mut unroll_pairs = Vec::new();
    for main in 0..forest.loops.len() {
        for rem in 0..forest.loops.len() {
            if main == rem
                || forest.loops[main].parent != forest.loops[rem].parent
                || names[main].line.is_none()
                || names[main].line != names[rem].line
                || names[main].file != names[rem].file
            {
                continue;
            }
            if let (Ok(m), Ok(r)) = (&trips[main], &trips[rem])
                && let Some(factor) = unroll_factor(m, r)
            {
                unroll_pairs.push(UnrollPair {
                    main: LoopId(main as u32),
                    remainder: LoopId(rem as u32),
                    factor,
                });
            }
        }
    }

    TripInfo {
        trips,
        unroll_pairs,
    }
}

/// `main = (X − X mod c)/c` and `rem = X mod c` → factor c.
fn unroll_factor(main: &SymExpr, rem: &SymExpr) -> Option<i64> {
    let SymExpr::Mod(rem_x, c) = rem else {
        return None;
    };
    let SymExpr::FloorDiv(inner, mc) = main else {
        return None;
    };
    if mc != c {
        return None;
    }
    let SymExpr::Sum(terms) = inner.as_ref() else {
        return None;
    };
    if terms.len() != 2 {
        return None;
    }
    let has_x = terms.iter().any(|t| t == rem_x.as_ref());
    let has_neg_mod = terms.iter().any(|t| match t {
        SymExpr::Prod(fs) => {
            fs.len() == 2 && fs[0] == SymExpr::Const(-1) && fs[1] == SymExpr::Mod(rem_x.clone(), *c)
        }
        _ => false,
    });
    (has_x && has_neg_mod).then_some(*c)
}

// -------------------------------------------------------------------------
// The scalar affine tracer.

impl<'a> Tracer<'a> {
    fn loop_trips(&self, id: LoopId) -> TripCount {
        let l = self.forest.get(id);

        // Structural requirements: one latch, one exit edge, at the latch.
        if l.latches.len() != 1 {
            return Err("loop has multiple latches".to_owned());
        }
        let latch = l.latches[0];
        let in_loop = |b: BlockId| l.blocks.binary_search(&b).is_ok();
        let mut exit_edges = Vec::new();
        for &b in &l.blocks {
            for &s in &self.cfg.block(b).succs {
                if !in_loop(s) {
                    exit_edges.push((b, s));
                }
            }
        }
        if exit_edges.len() != 1 {
            return Err(format!("loop has {} exit edges", exit_edges.len()));
        }
        if exit_edges[0].0 != latch {
            return Err("loop exit is not at the latch".to_owned());
        }

        // The latch terminator: `@%p bra HEADER` (or inverted).
        let latch_block = self.cfg.block(latch);
        let branch = self
            .last_instr(latch)
            .ok_or_else(|| "latch block has no instructions".to_owned())?;
        let pred = branch
            .1
            .predicate
            .ok_or_else(|| "latch branch is unconditional".to_owned())?;
        // succs[0] is the taken target (graph contract). Taken = header
        // means continue-if-true; predicate negation flips once more.
        let mut continue_if_true = latch_block.succs.first() == Some(&l.header);
        if pred.negated {
            continue_if_true = !continue_if_true;
        }

        // The setp defining the branch predicate, inside the latch block.
        let Some((setp_idx, setp)) = self.find_setp(latch, branch.0, pred.reg) else {
            return self
                .predicate_phi_trips(id, latch, branch.0, pred.reg, continue_if_true)
                .unwrap_or_else(|| Err(self.predicate_reason(latch, branch.0, pred.reg)));
        };
        let cmp = self
            .module
            .modifiers(setp)
            .first()
            .map(|&m| self.module.interner.resolve(m).to_owned())
            .ok_or_else(|| "setp without comparison modifier".to_owned())?;

        // Trace both operands at the setp.
        let ops = self.module.operand_ids(setp.operands);
        if ops.len() < 3 {
            return Err("setp with unexpected operand count".to_owned());
        }
        let a = self.trace_operand(ops[1], setp_idx, Some(id), 0)?;
        let b = self.trace_operand(ops[2], setp_idx, Some(id), 0)?;
        let d = a - b; // D = A − B; condition is `D cmp 0`

        // D(k) = A1·k + A0 in this loop's iteration number alone.
        let k = Var::Iter(id);
        if let Some(v) = d.terms.keys().find(|&&v| v != k) {
            return Err(match v {
                Var::Iter(_) => "latch condition depends on an enclosing loop's counter".to_owned(),
                other => format!("latch condition depends on special register {other}"),
            });
        }
        let Some(coeff) = d.terms.get(&k) else {
            return Err("latch condition does not involve an induction variable".to_owned());
        };
        let a1 = coeff
            .as_const()
            .ok_or_else(|| "induction step is not a constant".to_owned())?;
        solve(&cmp, continue_if_true, a1, d.base)
    }

    /// No setp defines the latch predicate: say what does, if anything
    /// in the latch block does (`mbarrier.test_wait` in a spin-wait).
    fn predicate_reason(&self, latch: BlockId, before: usize, pred: Symbol) -> String {
        let blk = self.cfg.block(latch);
        let definer = self.kernel.stmts[blk.start..before]
            .iter()
            .rev()
            .find_map(|s| match s {
                Stmt::Instr(i)
                    if self
                        .module
                        .operand_ids(i.operands)
                        .first()
                        .is_some_and(|&id| {
                            matches!(self.module.operand(id),
                                 Operand::Register(r) | Operand::SymbolRef(r) if *r == pred)
                        }) =>
                {
                    Some(self.module.opcode(i))
                }
                _ => None,
            });
        match definer {
            Some(op) => format!("latch predicate is defined by `{op}`, not a comparison"),
            None => "latch predicate is not defined in the latch block".to_owned(),
        }
    }

    /// LLVM's two-trip loop: the latch predicate is a copy of a register
    /// set true before the loop and false in the latch (`mov.pred %q,
    /// -1; header: mov.pred %p, %q; latch: mov.pred %q, 0; @%p bra
    /// header`). Trips: 2 if the initial value continues, else 1.
    fn predicate_phi_trips(
        &self,
        id: LoopId,
        latch: BlockId,
        branch_idx: usize,
        pred: Symbol,
        continue_if_true: bool,
    ) -> Option<TripCount> {
        let mov_src = |site: usize| -> Option<&Operand> {
            let Stmt::Instr(instr) = &self.kernel.stmts[site] else {
                return None;
            };
            if self.module.interner.resolve(instr.mnemonic) != "mov" {
                return None;
            }
            let [_, src] = self.module.operand_ids(instr.operands) else {
                return None;
            };
            Some(self.module.operand(*src))
        };
        let Reach::Def(copy) = self.reach_def(pred, branch_idx, None) else {
            return None;
        };
        let Operand::Register(phi) = mov_src(copy)? else {
            return None;
        };
        if !self.in_loop(id, copy) {
            return None;
        }
        let latch_blk = self.cfg.block(latch);
        let mut in_loop = self.defs.get(phi)?.iter().filter(|&&d| self.in_loop(id, d));
        let (&latch_def, None) = (in_loop.next()?, in_loop.next()) else {
            return None;
        };
        if latch_def < latch_blk.start || latch_def >= latch_blk.end {
            return None;
        }
        let header = self.forest.get(id).header;
        let Reach::Def(init_def) = self.reach_def(*phi, self.cfg.block(header).start, Some(id))
        else {
            return None;
        };
        let (Operand::Immediate(latch_val), Operand::Immediate(init_val)) =
            (mov_src(latch_def)?, mov_src(init_def)?)
        else {
            return None;
        };
        let continues = |text: &Symbol| {
            parse_int(self.module.interner.resolve(*text)).map(|v| (v != 0) == continue_if_true)
        };
        if continues(latch_val)? {
            return None;
        }
        Some(Ok(SymExpr::Const(if continues(init_val)? { 2 } else { 1 })))
    }
}

/// Solve `continue while A1·k + A0 cmp 0` (k = 1, 2, ...) for the
/// iteration count.
fn solve(cmp: &str, continue_if_true: bool, a1: i64, a0: SymExpr) -> TripCount {
    // Normalize to the continue-condition comparison.
    let cond = if continue_if_true {
        cmp.to_owned()
    } else {
        match cmp {
            "lt" => "ge",
            "le" => "gt",
            "gt" => "le",
            "ge" => "lt",
            "ne" => "eq",
            "eq" => "ne",
            other => other,
        }
        .to_owned()
    };
    let neg = |e: SymExpr| SymExpr::mul(SymExpr::Const(-1), e);

    match cond.as_str() {
        // D ≠ 0: exits exactly when A1·k = −A0.
        "ne" => {
            if a1 == 0 {
                return Err("latch condition does not involve an induction variable".to_owned());
            }
            let num = if a1 > 0 { neg(a0) } else { a0 };
            Ok(SymExpr::floor_div(num, a1.abs()))
        }
        // D < 0: false at the smallest k with A1·k ≥ −A0.
        "lt" => match a1 {
            a1 if a1 > 0 => Ok(SymExpr::ceil_div(neg(a0), a1)),
            0 => Err("latch condition does not involve an induction variable".to_owned()),
            _ => Err("loop bound moves away from the exit condition".to_owned()),
        },
        // D ≤ 0: false at the smallest k with A1·k > −A0.
        "le" => match a1 {
            a1 if a1 > 0 => Ok(SymExpr::add(
                SymExpr::floor_div(neg(a0), a1),
                SymExpr::Const(1),
            )),
            0 => Err("latch condition does not involve an induction variable".to_owned()),
            _ => Err("loop bound moves away from the exit condition".to_owned()),
        },
        // D > 0 ⇔ −D < 0;   D ≥ 0 ⇔ −D ≤ 0.
        "gt" => solve("lt", true, -a1, neg(a0)),
        "ge" => solve("le", true, -a1, neg(a0)),
        "eq" => Err("loop continues only while values are equal — shape not recognized".to_owned()),
        other => Err(format!("unsupported latch comparison `{other}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{build_cfg, loop_forest, loop_names};
    use crate::parse::parser::parse;

    /// Trips of every loop in a one-kernel source, by display name.
    fn trips_of(src: &str) -> Vec<(String, TripCount)> {
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names);
        names
            .into_iter()
            .zip(info.trips)
            .map(|(n, t)| (n.display, t))
            .collect()
    }

    fn kernel_with(params: &str, body: &str) -> String {
        format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n{params}\n)\n{{\n{body}\n}}\n"
        )
    }

    const N_PARAM: &str = ".param .u32 k_param_0";

    #[test]
    fn up_counting_lt_stride_1() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn up_counting_ne_form() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.ne.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn stride_gt_1_is_ceil_div() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 8;\n\
             setp.lt.u32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "⌈param_0/8⌉");
    }

    #[test]
    fn signed_division_by_power_of_two_bound() {
        // nvcc's lowering of `n / 16` for a signed `n`, verbatim from the
        // k14 fixture: sign bit, shifted down, added, arithmetic shift.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\n\
             shr.s32 %r2, %r1, 31;\nshr.u32 %r3, %r2, 28;\n\
             add.s32 %r4, %r1, %r3;\nshr.s32 %r5, %r4, 4;\n\
             mov.u32 %r6, 0;\n\
             $L__L:\nadd.s32 %r6, %r6, 1;\n\
             setp.lt.u32 %p1, %r6, %r5;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0 / 16");
    }

    #[test]
    fn increment_through_an_intermediate_register() {
        // k14's main loop: `t = i + 2` is also read by the body, so nvcc
        // increments as `i = t - 1`.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r3, %r2, 2;\nsetp.ge.u32 %p2, %r3, %r1;\n\
             add.s32 %r2, %r3, -1;\n\
             setp.lt.u32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn countdown_to_zero() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, %r1;\n\
             $L__L:\nadd.s32 %r2, %r2, -1;\n\
             setp.ne.s32 %p1, %r2, 0;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn sixty_four_bit_iv() {
        let src = kernel_with(
            ".param .u64 k_param_0",
            "ld.param.u64 %rd1, [k_param_0];\nmov.u64 %rd2, 0;\n\
             $L__L:\nadd.s64 %rd2, %rd2, 1;\n\
             setp.lt.s64 %p1, %rd2, %rd1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn rotated_do_while_constant_bound() {
        // do { } while (++i != 8) — the k5 inner shape with constants.
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.ne.s32 %p1, %r2, 8;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "8");
    }

    #[test]
    fn derived_register_latch_verbatim_from_k2() {
        // The verified k2 shape, pinned verbatim: the latch compares
        // %r29 = %r7 + %r35 against 0, where %r7 = (K&3) − K.
        let src = kernel_with(
            ".param .u32 k_param_0,\n.param .u32 k_param_1,\n.param .u32 k_param_2",
            "ld.param.u32 %r19, [k_param_2];\n\
             and.b32 %r37, %r19, 3;\n\
             sub.s32 %r7, %r37, %r19;\n\
             mov.u32 %r35, 0;\n\
             $L__BB0_4:\n\
             add.s32 %r35, %r35, 4;\n\
             add.s32 %r29, %r7, %r35;\n\
             setp.ne.s32 %p6, %r29, 0;\n@%p6 bra $L__BB0_4;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(
            trips[0].1.as_ref().unwrap().to_string(),
            "(param_2 - param_2 mod 4) / 4"
        );
    }

    #[test]
    fn negated_predicate_inverts_the_condition() {
        // @!%p bra header with setp.ge: continue while NOT(i >= N) = i < N.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.ge.s32 %p1, %r2, %r1;\n@!%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn multi_exit_is_a_named_unknown() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.gt.s32 %p2, %r2, 100;\n@%p2 bra $L__OUT;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\n\
             $L__OUT:\nret;",
        );
        let trips = trips_of(&src);
        let err = trips[0].1.as_ref().unwrap_err();
        assert!(err.contains("exit"), "{err}");
    }

    #[test]
    fn data_dependent_latch_is_a_named_unknown() {
        let src = kernel_with(
            ".param .u64 k_param_0",
            "ld.param.u64 %rd1, [k_param_0];\n\
             $L__L:\nld.global.f32 %f1, [%rd1];\nadd.s64 %rd1, %rd1, 4;\n\
             setp.gt.f32 %p1, %f1, 0f00000000;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(
            trips[0].1.as_ref().unwrap_err(),
            "latch condition depends on a value loaded inside the loop"
        );
    }

    #[test]
    fn the_reason_is_the_obstacle_behind_unread_arithmetic() {
        // LLVM writes m_start + 127 as `or` on a value with zero low bits;
        // the bound still comes from the CTA index, and that is the reason.
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, %ctaid.x;\nshl.b32 %r2, %r1, 7;\nor.b32 %r3, %r2, 127;\n\
             mov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\n\
             setp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let err = trips_of(&src)[0].1.clone().unwrap_err();
        assert!(err.contains("special register %ctaid.x"), "{err}");
        // A loaded bound behind `bfe` is a loaded bound.
        let src = kernel_with(
            ".param .u64 k_param_0",
            "ld.param.u64 %rd1, [k_param_0];\nld.global.u32 %r1, [%rd1];\n\
             bfe.s32 %r3, %r1, 0, 26;\nmov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\n\
             setp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let err = trips_of(&src)[0].1.clone().unwrap_err();
        assert_eq!(err, "latch condition depends on a value loaded from memory");
        // Nothing fundamental behind it: the instruction is still named.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nor.b32 %r3, %r1, 1;\nmov.u32 %r4, 0;\n\
             $L__L:\nadd.s32 %r4, %r4, 1;\nsetp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let err = trips_of(&src)[0].1.clone().unwrap_err();
        assert_eq!(err, "value defined by unsupported instruction `or`");
    }

    #[test]
    fn a_spin_wait_names_what_defines_its_predicate() {
        let src = kernel_with(
            N_PARAM,
            "{ .reg .pred complete;\nwaitLoop:\n\
             mbarrier.test_wait.parity.shared::cta.b64 complete, [%r1], %r2;\n\
             @!complete bra.uni waitLoop; }\nret;",
        );
        let err = trips_of(&src)[0].1.clone().unwrap_err();
        assert_eq!(
            err,
            "latch predicate is defined by `mbarrier.test_wait.parity.shared::cta.b64`, not a comparison"
        );
    }

    #[test]
    fn a_read_before_the_increment_is_the_previous_iterations_value() {
        // t = i (before i += 1), compared with n: continue while k − 1 < n.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nmov.u32 %r5, %r2;\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r5, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0 + 1");
    }

    #[test]
    fn a_value_carried_around_an_enclosing_loop_is_refused() {
        // Triangular: j < i, where i is the outer counter. The old walk
        // resolved i to its initial value and reported 0 trips.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__O:\nmov.u32 %r3, 0;\n$L__I:\nadd.s32 %r3, %r3, 1;\n\
             setp.lt.s32 %p1, %r3, %r2;\n@%p1 bra $L__I;\n\
             add.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n@%p2 bra $L__O;\nret;",
        );
        let trips = trips_of(&src);
        let inner = trips
            .iter()
            .find(|(n, _)| n == "$L__I")
            .expect("inner loop");
        let err = inner.1.clone().unwrap_err();
        assert_eq!(
            err,
            "latch condition depends on an enclosing loop's counter"
        );
        let outer = trips
            .iter()
            .find(|(n, _)| n == "$L__O")
            .expect("outer loop");
        assert_eq!(outer.1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn a_value_merged_from_two_paths_is_refused() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nsetp.lt.s32 %p2, %r1, 8;\n@%p2 bra $L__A;\n\
             mov.u32 %r3, 5;\nbra.uni $L__J;\n$L__A:\nmov.u32 %r3, 7;\n$L__J:\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        let err = trips[0].1.clone().unwrap_err();
        assert_eq!(err, "%r3 has more than one reaching definition");
    }

    #[test]
    fn special_register_dependence_is_named() {
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, %tid.x;\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        let err = trips[0].1.as_ref().unwrap_err();
        assert!(err.contains("special register %tid.x"), "{err}");
    }

    #[test]
    fn unroll_pair_links_main_and_remainder() {
        // Two sibling loops on the same source line with the
        // (X − X mod 4)/4 and X mod 4 trip shapes.
        let src = String::from(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u32 k_param_0\n)\n{\n\
             ld.param.u32 %r1, [k_param_0];\n\
             and.b32 %r9, %r1, 3;\n\
             sub.s32 %r7, %r9, %r1;\n\
             mov.u32 %r2, 0;\n\
             .loc 1 14 9\n\
             $L__M:\n\
             add.s32 %r2, %r2, 4;\n\
             add.s32 %r3, %r7, %r2;\n\
             setp.ne.s32 %p1, %r3, 0;\n@%p1 bra $L__M;\n\
             mov.u32 %r4, %r9;\n\
             $L__R:\n\
             .loc 1 14 9\n\
             add.s32 %r4, %r4, -1;\n\
             setp.ne.s32 %p2, %r4, 0;\n@%p2 bra $L__R;\n\
             ret;\n}\n.file 1 \"kern.cu\"\n",
        );
        let m = parse(&src).expect("parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names);
        assert_eq!(info.unroll_pairs.len(), 1, "{:?}", info);
        assert_eq!(info.unroll_pairs[0].factor, 4);
        let main = info.unroll_pairs[0].main;
        assert_eq!(
            info.trips[main.0 as usize].as_ref().unwrap().to_string(),
            "(param_0 - param_0 mod 4) / 4"
        );
    }
}
