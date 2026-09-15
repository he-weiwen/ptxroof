//! Affine values over the variables a kernel's scalars are affine in:
//! thread and CTA indices and loop iteration numbers, with symbolic
//! coefficients over the kernel parameters (`K * %tid.y` is a term).
//! The trip matcher reads latch conditions in this form; addresses and
//! branch conditions will be read in it too.

use crate::cfg::loops::LoopId;
use crate::core::symexpr::SymExpr;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Var {
    Tid(Axis),
    Ctaid(Axis),
    Ntid(Axis),
    Nctaid(Axis),
    /// Iteration number k = 1, 2, ... of the loop, as counted at its latch.
    Iter(LoopId),
}

impl Var {
    /// `%tid.x`, `%ctaid.y`, `%ntid.z`, `%nctaid.x`; other names are not
    /// affine-structured and stay opaque.
    pub fn special(name: &str) -> Option<Var> {
        let (reg, axis) = name.strip_prefix('%')?.split_once('.')?;
        let axis = match axis {
            "x" => Axis::X,
            "y" => Axis::Y,
            "z" => Axis::Z,
            _ => return None,
        };
        Some(match reg {
            "tid" => Var::Tid(axis),
            "ctaid" => Var::Ctaid(axis),
            "ntid" => Var::Ntid(axis),
            "nctaid" => Var::Nctaid(axis),
            _ => return None,
        })
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let axis = |a: &Axis| match a {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        };
        match self {
            Var::Tid(a) => write!(f, "%tid.{}", axis(a)),
            Var::Ctaid(a) => write!(f, "%ctaid.{}", axis(a)),
            Var::Ntid(a) => write!(f, "%ntid.{}", axis(a)),
            Var::Nctaid(a) => write!(f, "%nctaid.{}", axis(a)),
            Var::Iter(l) => write!(f, "k[loop {}]", l.0),
        }
    }
}

/// `Σ coeff·var + base`; a missing variable has coefficient 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Affine {
    pub terms: BTreeMap<Var, SymExpr>,
    pub base: SymExpr,
}

impl Affine {
    pub fn invariant(base: SymExpr) -> Affine {
        Affine {
            terms: BTreeMap::new(),
            base,
        }
    }

    pub fn var(v: Var) -> Affine {
        Affine::term(v, SymExpr::Const(1))
    }

    pub fn term(v: Var, coeff: SymExpr) -> Affine {
        let mut a = Affine::invariant(SymExpr::Const(0));
        if coeff != SymExpr::Const(0) {
            a.terms.insert(v, coeff);
        }
        a
    }

    /// Multiply by a loop-invariant value.
    pub fn scale(self, c: SymExpr) -> Affine {
        if c == SymExpr::Const(0) {
            return Affine::invariant(SymExpr::Const(0));
        }
        Affine {
            terms: self
                .terms
                .into_iter()
                .map(|(v, k)| (v, SymExpr::mul(c.clone(), k)))
                .collect(),
            base: SymExpr::mul(c, self.base),
        }
    }

    pub fn is_invariant(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn as_const(&self) -> Option<i64> {
        self.is_invariant().then(|| self.base.as_const()).flatten()
    }
}

impl std::ops::Add for Affine {
    type Output = Affine;

    fn add(mut self, other: Affine) -> Affine {
        for (v, c) in other.terms {
            let sum = match self.terms.remove(&v) {
                Some(mine) => SymExpr::add(mine, c),
                None => c,
            };
            if sum != SymExpr::Const(0) {
                self.terms.insert(v, sum);
            }
        }
        self.base = SymExpr::add(self.base, other.base);
        self
    }
}

impl std::ops::Sub for Affine {
    type Output = Affine;

    fn sub(self, other: Affine) -> Affine {
        self + other.scale(SymExpr::Const(-1))
    }
}

impl fmt::Display for Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (v, c) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            match c {
                SymExpr::Const(1) => write!(f, "{v}")?,
                SymExpr::Const(_) | SymExpr::Sym(_) => write!(f, "{c} * {v}")?,
                _ => write!(f, "({c}) * {v}")?,
            }
        }
        if first {
            write!(f, "{}", self.base)
        } else if self.base == SymExpr::Const(0) {
            Ok(())
        } else {
            write!(f, " + {}", self.base)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tid() -> Affine {
        Affine::var(Var::Tid(Axis::X))
    }

    #[test]
    fn special_registers_with_an_axis_are_variables() {
        assert_eq!(Var::special("%ctaid.y"), Some(Var::Ctaid(Axis::Y)));
        assert_eq!(Var::special("%laneid"), None);
        assert_eq!(Var::special("%r5"), None);
    }

    #[test]
    fn terms_combine_cancel_and_scale_symbolically() {
        let k = SymExpr::sym("param_2");
        let a = tid().scale(SymExpr::Const(2)) + Affine::invariant(SymExpr::Const(8));
        assert_eq!(a.to_string(), "2 * %tid.x + 8");
        assert_eq!(
            (a.clone() - tid().scale(SymExpr::Const(2))).as_const(),
            Some(8)
        );
        let row = Affine::var(Var::Tid(Axis::Y)).scale(k.clone());
        assert_eq!(row.to_string(), "param_2 * %tid.y");
        assert_eq!((row + a).to_string(), "2 * %tid.x + param_2 * %tid.y + 8");
        assert!(tid().scale(SymExpr::Const(0)).as_const() == Some(0));
        assert_eq!(Affine::var(Var::Iter(LoopId(3))).to_string(), "k[loop 3]");
    }
}
