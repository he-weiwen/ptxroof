//! Loop-exit inequalities over symbolic expressions, solved for a trip
//! count.

use super::sym_expr::SymExpr;
use crate::ptx::modifier::Cmp;

/// `e cmp 0` as `e' < 0` over the integers, for an ordered comparison.
pub(crate) fn below_zero(cmp: Cmp, e: SymExpr) -> Option<SymExpr> {
    let minus = |e: SymExpr| SymExpr::mul(SymExpr::Const(-1), e);
    Some(undivided(match cmp {
        Cmp::Lt => e,
        Cmp::Le => SymExpr::sub(e, SymExpr::Const(1)),
        Cmp::Gt => minus(e),
        Cmp::Ge => SymExpr::sub(minus(e), SymExpr::Const(1)),
        Cmp::Eq | Cmp::Ne => return None,
    }))
}

/// `±⌊x/d⌋ + b < 0` or `±⌈x/d⌉ + b < 0` as the same test of x, as
/// ⌊x/d⌋ ≤ m ⇔ x < (m + 1)·d and ⌈x/d⌉ ≤ m ⇔ x ≤ m·d for d > 0.
fn undivided(e: SymExpr) -> SymExpr {
    let terms = match &e {
        SymExpr::Sum(ts) => ts.clone(),
        other => vec![other.clone()],
    };
    let (consts, rest): (Vec<_>, Vec<_>) = terms.into_iter().partition(|t| t.as_const().is_some());
    let [rest] = &rest[..] else {
        return e;
    };
    let b: i128 = (consts.iter())
        .filter_map(SymExpr::as_const)
        .map(i128::from)
        .sum();
    let (x, k) = match SymExpr::split_const(rest.clone()) {
        (1, SymExpr::FloorDiv(x, d)) => (*x, b * i128::from(d)),
        (-1, SymExpr::FloorDiv(x, d)) => (
            SymExpr::mul(SymExpr::Const(-1), *x),
            (b + 1) * i128::from(d) - 1,
        ),
        (1, SymExpr::CeilDiv(x, d)) => (*x, (b + 1) * i128::from(d) - 1),
        (-1, SymExpr::CeilDiv(x, d)) => (SymExpr::mul(SymExpr::Const(-1), *x), b * i128::from(d)),
        _ => return e,
    };
    match i64::try_from(k) {
        Ok(k) => SymExpr::add(x, SymExpr::Const(k)),
        Err(_) => e,
    }
}

/// Solve `continue while A1·k + A0 cmp 0` (k = 1, 2, ...) for the
/// iteration count: `ne` → −A0/A1, the division taken exact
/// (compiler-generated not-equal latches step in divisor units; a
/// constant −A0/A1 that is not a positive integer is refused);
/// `lt` → ceildiv(−A0, A1); `le` → floordiv(−A0, A1) + 1; `gt`/`ge`
/// mirrored.
pub(crate) fn exit_limit(cond: Cmp, a1: i64, a0: SymExpr) -> Result<SymExpr, String> {
    let neg = |e: SymExpr| SymExpr::mul(SymExpr::Const(-1), e);

    match cond {
        // D ≠ 0: exits exactly when A1·k = −A0.
        Cmp::Ne => {
            if a1 == 0 {
                return Err("latch condition does not involve an induction variable".to_owned());
            }
            let num = if a1 > 0 { neg(a0) } else { a0 };
            if let Some(n) = num.as_const()
                && (n <= 0 || n % a1.abs() != 0)
            {
                return Err("latch condition never meets its `!=` bound".to_owned());
            }
            Ok(SymExpr::floor_div(num, a1.abs()))
        }
        // D < 0: false at the smallest k with A1·k ≥ −A0.
        Cmp::Lt => match a1 {
            a1 if a1 > 0 => Ok(SymExpr::ceil_div(neg(a0), a1)),
            0 => Err("latch condition does not involve an induction variable".to_owned()),
            _ => Err("loop bound moves away from the exit condition".to_owned()),
        },
        // D ≤ 0: false at the smallest k with A1·k > −A0.
        Cmp::Le => match a1 {
            a1 if a1 > 0 => Ok(SymExpr::add(
                SymExpr::floor_div(neg(a0), a1),
                SymExpr::Const(1),
            )),
            0 => Err("latch condition does not involve an induction variable".to_owned()),
            _ => Err("loop bound moves away from the exit condition".to_owned()),
        },
        // D > 0 ⇔ −D < 0;   D ≥ 0 ⇔ −D ≤ 0.
        Cmp::Gt => exit_limit(Cmp::Lt, -a1, neg(a0)),
        Cmp::Ge => exit_limit(Cmp::Le, -a1, neg(a0)),
        Cmp::Eq => {
            Err("loop continues only while values are equal — shape not recognized".to_owned())
        }
    }
}
