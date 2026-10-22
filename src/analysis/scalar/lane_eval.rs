//! Evaluate the lane-dependent part of affine values.

use super::affine::{Affine, Axis, Var};

pub fn depends_on_lane(v: &Var) -> bool {
    match v {
        Var::Tid(_) => true,
        Var::Ctaid(_) | Var::Iter(_) | Var::Uniform(_) => false,
        Var::Div(inner, _) | Var::Mod(inner, _) => depends_on_lane(inner),
    }
}

/// The value of a form with constant coefficients for one thread;
/// variables that are not lane-dependent and symbols count as 0, so
/// the constant offset of the base is kept.
pub fn eval_lane(a: &Affine, tid: [i64; 3]) -> i64 {
    a.base.const_part()
        + a.terms
            .iter()
            .map(|(v, c)| c.as_const().unwrap_or(0) * v.eval(&lanes(tid)))
            .sum::<i64>()
}

/// Each thread index at its value in thread `tid`, every other
/// variable at 0.
pub(in crate::analysis) fn lanes(tid: [i64; 3]) -> impl Fn(&Var) -> i64 {
    move |v| match v {
        Var::Tid(Axis::X) => tid[0],
        Var::Tid(Axis::Y) => tid[1],
        Var::Tid(Axis::Z) => tid[2],
        _ => 0,
    }
}
