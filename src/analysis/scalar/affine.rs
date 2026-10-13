//! Affine values over the variables a kernel's scalars are affine in:
//! thread and CTA indices and loop iteration numbers, with symbolic
//! coefficients over the kernel parameters and the launch shape
//! (`K * %tid.y` is a term; `%ntid.x` is a symbol, bound when the
//! shape is known).
//! Trip matching, address analysis, and thread-participation analysis
//! read latch conditions, addresses, and branch conditions in this form.

use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::scalar::symexpr::SymExpr;
use crate::ptx::ir::{Element, SpecialReg, SpecialVector};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Var {
    Tid(Axis),
    Ctaid(Axis),
    /// Iteration number k = 1, 2, ... of the loop, as counted at its latch.
    Iter(LoopId),
    /// `⌊v / d⌋`, d > 0: a 2D tile's thread row, `tid.x / 8`.
    Div(Box<Var>, i64),
    /// `v mod m`, m > 0: its thread column, `tid.x % 8`.
    Mod(Box<Var>, i64),
}

impl Var {
    /// `%tid.x`, `%ctaid.y`; the launch shape (`%ntid`, `%nctaid`) is a
    /// symbol, not a variable, and other special registers stay opaque.
    pub fn special(reg: SpecialReg) -> Option<Var> {
        let SpecialReg::Vector(vector, Some(element)) = reg else {
            return None;
        };
        let axis = match element {
            Element::X => Axis::X,
            Element::Y => Axis::Y,
            Element::Z => Axis::Z,
            Element::W => return None,
        };
        match vector {
            SpecialVector::Tid => Some(Var::Tid(axis)),
            SpecialVector::Ctaid => Some(Var::Ctaid(axis)),
            _ => None,
        }
    }
}

impl Var {
    /// The variable with the leaf indices printed by `name`.
    pub fn render(&self, name: &impl Fn(&Var) -> String) -> String {
        match self {
            Var::Div(v, d) => format!("⌊{}/{d}⌋", v.render(name)),
            Var::Mod(v, m) => format!("({} mod {m})", v.render(name)),
            leaf => name(leaf),
        }
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
            Var::Iter(l) => write!(f, "k[loop {}]", l.0),
            Var::Div(v, d) => write!(f, "⌊{v}/{d}⌋"),
            Var::Mod(v, m) => write!(f, "({v} mod {m})"),
        }
    }
}

/// `Σ coeff·var + base`; a missing variable has coefficient 0.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    /// Substitute bound symbols in every coefficient and the base.
    pub fn bind(&self, bindings: &std::collections::HashMap<String, i64>) -> Affine {
        let mut out = Affine::invariant(self.base.bind(bindings));
        for (v, c) in &self.terms {
            out = out + Affine::term(v.clone(), c.bind(bindings));
        }
        out
    }

    /// The form `1·v` with no base: a variable on its own.
    pub fn single_var(&self) -> Option<Var> {
        if self.base != SymExpr::Const(0) || self.terms.len() != 1 {
            return None;
        }
        let (v, c) = self.terms.iter().next()?;
        (*c == SymExpr::Const(1)).then(|| v.clone())
    }

    /// Divide exactly by `d`, when every coefficient and the base are
    /// known multiples of it.
    pub fn div_exact(self, d: i64) -> Option<Affine> {
        if !self
            .terms
            .values()
            .chain([&self.base])
            .all(|e| e.divisible_by(d))
        {
            return None;
        }
        Some(Affine {
            terms: self
                .terms
                .into_iter()
                .map(|(v, c)| (v, SymExpr::floor_div(c, d)))
                .collect(),
            base: SymExpr::floor_div(self.base, d),
        })
    }

    /// `⌊self / d⌋`, d > 0: exact when every part is a multiple of d; a
    /// derived variable when the form is `c·v` with c dividing d; a
    /// floor division when invariant. None otherwise.
    pub fn div_const(self, d: i64) -> Option<Affine> {
        if self.is_invariant() {
            return Some(Affine::invariant(SymExpr::floor_div(self.base, d)));
        }
        if let Some(exact) = self.clone().div_exact(d) {
            return Some(exact);
        }
        let (v, c) = self.terms.iter().next()?;
        let c = c.as_const()?;
        if self.terms.len() != 1 || self.base != SymExpr::Const(0) || c <= 0 || d % c != 0 {
            return None;
        }
        Some(Affine::var(Var::Div(Box::new(v.clone()), d / c)))
    }

    /// `self mod m`, m > 0: zero when every part is a multiple of m; for
    /// `c·v + b` with c dividing m and b a multiple of m, `c·(v mod m/c)`;
    /// a symbolic modulus when invariant. None otherwise.
    pub fn modulo_const(self, m: i64) -> Option<Affine> {
        if self.is_invariant() {
            return Some(Affine::invariant(SymExpr::modulo(self.base, m)));
        }
        if self.clone().div_exact(m).is_some() {
            return Some(Affine::invariant(SymExpr::Const(0)));
        }
        let (v, c) = self.terms.iter().next()?;
        let c = c.as_const()?;
        if self.terms.len() != 1 || !self.base.divisible_by(m) || c <= 0 || m % c != 0 {
            return None;
        }
        Some(Affine::var(Var::Mod(Box::new(v.clone()), m / c)).scale(SymExpr::Const(c)))
    }

    /// `self − self mod m`, the multiple of m at or below it: for
    /// `c·v + b` with c dividing m and b a multiple of m, `m·⌊v/(m/c)⌋ + b`.
    pub fn align_down(self, m: i64) -> Option<Affine> {
        if self.is_invariant() {
            let rem = SymExpr::modulo(self.base.clone(), m);
            return Some(Affine::invariant(SymExpr::sub(self.base, rem)));
        }
        if self.clone().div_exact(m).is_some() {
            return Some(self);
        }
        let (v, c) = self.terms.iter().next()?;
        let c = c.as_const()?;
        if self.terms.len() != 1 || !self.base.divisible_by(m) || c <= 0 || m % c != 0 {
            return None;
        }
        let quotient = Affine::var(Var::Div(Box::new(v.clone()), m / c));
        Some(quotient.scale(SymExpr::Const(m)) + Affine::invariant(self.base))
    }

    /// Substitute every variable by a form; None if any substitution is.
    pub fn map_vars(&self, f: impl Fn(&Var) -> Option<Affine>) -> Option<Affine> {
        let mut out = Affine::invariant(self.base.clone());
        for (v, c) in &self.terms {
            out = out + f(v)?.scale(c.clone());
        }
        Some(out)
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

impl Affine {
    /// The form with each variable printed by `name`.
    pub fn render(&self, name: impl Fn(&Var) -> String) -> String {
        let mut out = String::new();
        for (v, c) in &self.terms {
            let (k, rest) = SymExpr::split_const(c.clone());
            let coeff = SymExpr::mul(SymExpr::Const(k.abs()), rest);
            let v = v.render(&name);
            let term = match coeff {
                SymExpr::Const(1) => v,
                SymExpr::Const(_) | SymExpr::Sym(_) => format!("{coeff} * {v}"),
                _ => format!("({coeff}) * {v}"),
            };
            match (out.is_empty(), k < 0) {
                (true, false) => out.push_str(&term),
                (true, true) => out.push_str(&format!("-{term}")),
                (false, false) => out.push_str(&format!(" + {term}")),
                (false, true) => out.push_str(&format!(" - {term}")),
            }
        }
        if out.is_empty() {
            return self.base.to_string();
        }
        match &self.base {
            SymExpr::Const(0) => out,
            SymExpr::Const(c) if *c < 0 => format!("{out} - {}", -c),
            base => format!("{out} + {base}"),
        }
    }

    /// The largest value the form can take, when every term is a
    /// modulus with a positive constant coefficient and the base is a
    /// constant: `Σ c·(m − 1) + base`.
    pub fn max_value(&self) -> Option<i64> {
        let mut max = self.base.as_const()?;
        for (v, c) in &self.terms {
            let (Var::Mod(_, m), Some(c)) = (v, c.as_const()) else {
                return None;
            };
            if c <= 0 {
                return None;
            }
            max = max.checked_add(c.checked_mul(m - 1)?)?;
        }
        Some(max)
    }

    /// The least and the greatest value of the form when its base and
    /// coefficients are constants and `bound` gives every variable's
    /// least and greatest value; `None` when one is unbounded, the whole
    /// `i128` line when a partial sum leaves it.
    pub fn range(&self, bound: &dyn Fn(&Var) -> Option<(i64, i64)>) -> Option<(i128, i128)> {
        let base = i128::from(self.base.as_const()?);
        let mut range = Some((base, base));
        for (v, c) in &self.terms {
            let c = i128::from(c.as_const()?);
            let (lo, hi) = var_range(v, bound)?;
            let (a, b) = (c * i128::from(lo), c * i128::from(hi));
            range =
                range.and_then(|(x, y)| Some((x.checked_add(a.min(b))?, y.checked_add(a.max(b))?)));
        }
        Some(range.unwrap_or((i128::MIN, i128::MAX)))
    }
}

/// A variable's least and greatest value, a floor or a mod from its
/// operand's.
fn var_range(v: &Var, bound: &dyn Fn(&Var) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
    match v {
        Var::Div(inner, d) => {
            let (lo, hi) = var_range(inner, bound)?;
            Some((lo.div_euclid(*d), hi.div_euclid(*d)))
        }
        Var::Mod(inner, m) => match var_range(inner, bound)? {
            (lo, hi) if lo >= 0 && hi < *m => Some((lo, hi)),
            _ => Some((0, m - 1)),
        },
        _ => bound(v),
    }
}

impl fmt::Display for Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render(|v| v.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_range_spans_each_term_at_both_ends_of_its_variable() {
        let tid = Affine::var(Var::Tid(Axis::X));
        let k = Affine::var(Var::Iter(LoopId(0)));
        let bound = |v: &Var| match v {
            Var::Tid(Axis::X) => Some((0, 127)),
            Var::Iter(_) => Some((1, 8)),
            _ => None,
        };
        let a = tid.scale(SymExpr::Const(4))
            + k.scale(SymExpr::Const(-16))
            + Affine::invariant(SymExpr::Const(100));
        assert_eq!(a.range(&bound), Some((100 - 128, 100 + 508 - 16)));
        let floor = Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 32));
        assert_eq!(floor.range(&bound), Some((0, 3)));
        let modulo = Affine::var(Var::Mod(Box::new(Var::Iter(LoopId(0))), 4));
        assert_eq!(modulo.range(&bound), Some((0, 3)));
        let ctaid = Affine::var(Var::Ctaid(Axis::X));
        assert_eq!(ctaid.range(&bound), None);
        let param = Affine::invariant(SymExpr::sym("param_0"));
        assert_eq!(param.range(&bound), None);
    }

    #[test]
    fn a_range_whose_partial_sum_overflows_spans_everything() {
        // Three M·M overflow i128 before two -M·M bring the sum back.
        let m = i64::MAX;
        let a = [m, m, m, -m, -m]
            .into_iter()
            .zip(0..)
            .fold(Affine::invariant(SymExpr::Const(0)), |a, (c, l)| {
                a + Affine::var(Var::Iter(LoopId(l))).scale(SymExpr::Const(c))
            });
        let bound = |_: &Var| Some((m, m));
        assert_eq!(a.range(&bound), Some((i128::MIN, i128::MAX)));
    }

    fn tid() -> Affine {
        Affine::var(Var::Tid(Axis::X))
    }

    #[test]
    fn special_registers_with_an_axis_are_variables() {
        let special = |name| Var::special(SpecialReg::from_name(name).expect(name));
        assert_eq!(special("%ctaid.y"), Some(Var::Ctaid(Axis::Y)));
        assert_eq!(special("%laneid"), None);
        assert_eq!(special("%ntid.x"), None);
        assert_eq!(special("%tid.w"), None);
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
        let row = Var::Div(Box::new(Var::Tid(Axis::X)), 8);
        let col = Var::Mod(Box::new(Var::Tid(Axis::X)), 8);
        let a =
            Affine::var(row).scale(SymExpr::Const(16)) + Affine::var(col).scale(SymExpr::Const(2));
        assert_eq!(a.to_string(), "16 * ⌊%tid.x/8⌋ + 2 * (%tid.x mod 8)");
        assert_eq!(
            tid()
                .scale(SymExpr::Const(8))
                .div_exact(4)
                .unwrap()
                .to_string(),
            "2 * %tid.x"
        );
        assert!(tid().scale(SymExpr::Const(6)).div_exact(4).is_none());
        // (8·tid) mod 64 = 8·(tid mod 8); (tid + 64) mod 8 = tid mod 8; (8·tid)/2 = 4·tid;
        // tid/8 is a derived variable; (2·tid)/8 is ⌊tid/4⌋; (3·tid) mod 8 is not affine.
        let m = |a: Affine, k| a.modulo_const(k).map(|x| x.to_string());
        let d = |a: Affine, k| a.div_const(k).map(|x| x.to_string());
        assert_eq!(
            m(tid().scale(SymExpr::Const(8)), 64).as_deref(),
            Some("8 * (%tid.x mod 8)")
        );
        assert_eq!(
            m(tid() + Affine::invariant(SymExpr::Const(64)), 8).as_deref(),
            Some("(%tid.x mod 8)")
        );
        assert_eq!(
            d(tid().scale(SymExpr::Const(8)), 2).as_deref(),
            Some("4 * %tid.x")
        );
        assert_eq!(d(tid(), 8).as_deref(), Some("⌊%tid.x/8⌋"));
        assert_eq!(
            d(tid().scale(SymExpr::Const(2)), 8).as_deref(),
            Some("⌊%tid.x/4⌋")
        );
        assert_eq!(m(tid().scale(SymExpr::Const(3)), 8), None);
        let neg = tid().scale(SymExpr::Const(16))
            - tid().modulo_const(8).unwrap().scale(SymExpr::Const(16));
        assert_eq!(neg.to_string(), "16 * %tid.x - 16 * (%tid.x mod 8)");
        assert_eq!(
            (tid().scale(SymExpr::Const(-2)) + Affine::invariant(SymExpr::Const(-4))).to_string(),
            "-2 * %tid.x - 4"
        );
        assert_eq!(
            tid()
                .modulo_const(8)
                .unwrap()
                .scale(SymExpr::Const(8))
                .max_value(),
            Some(56)
        );
        assert_eq!(tid().max_value(), None);
        assert_eq!(tid().align_down(8).unwrap().to_string(), "8 * ⌊%tid.x/8⌋");
        assert_eq!(
            (tid().scale(SymExpr::Const(2)) + Affine::invariant(SymExpr::Const(64)))
                .align_down(16)
                .unwrap()
                .to_string(),
            "16 * ⌊%tid.x/8⌋ + 64"
        );
    }
}
