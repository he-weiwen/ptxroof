//! Evaluate the lane-dependent part of affine values.

use super::affine::{Affine, Axis, Var};
use std::ops::Range;

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

/// The linear index of thread `tid` in a block of `shape`, `%tid.x`
/// varying fastest.
pub(in crate::analysis) fn linear(tid: [i64; 3], shape: [u32; 3]) -> i64 {
    let [nx, ny, _] = shape.map(i64::from);
    tid[0] + nx * (tid[1] + ny * tid[2])
}

/// The thread of linear index `t` in a block of `shape`.
pub(in crate::analysis) fn thread(t: i64, shape: [u32; 3]) -> [i64; 3] {
    let [nx, ny, _] = shape.map(i64::from);
    [t % nx, (t / nx) % ny, t / (nx * ny)]
}

/// The warps of a block of `shape` as ranges of linear thread indices:
/// PTX ISA §3.1, "each warp contains threads of consecutive, increasing
/// thread IDs with the first warp containing thread 0".
pub(in crate::analysis) fn warps_of(shape: [u32; 3]) -> impl Iterator<Item = Range<i64>> {
    let threads: i64 = shape.iter().map(|&n| i64::from(n)).product();
    (0..threads)
        .step_by(32)
        .map(move |base| base..(base + 32).min(threads))
}
