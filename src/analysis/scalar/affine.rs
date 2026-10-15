//! Affine values over the variables a kernel's scalars are affine in:
//! thread and CTA indices and loop iteration numbers, with symbolic
//! coefficients over the kernel parameters and the launch shape
//! (`K * %tid.y` is a term; `%ntid.x` is a symbol, bound when the
//! shape is known).
//! Trip matching, address analysis, and thread-participation analysis
//! read latch conditions, addresses, and branch conditions in this form.

use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::scalar::symexpr::{SymExpr, product};
use crate::ptx::ir::{Element, SpecialReg, SpecialVector};
use crate::ssa::Value;
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
    /// A register value the tracer cannot interpret that every lane
    /// executing its read sees alike.
    Uniform(Value),
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
    /// The loop whose counter this is, or a floor or mod of.
    pub fn counter(&self) -> Option<LoopId> {
        match self {
            Var::Iter(l) => Some(*l),
            Var::Div(inner, _) | Var::Mod(inner, _) => inner.counter(),
            Var::Tid(_) | Var::Ctaid(_) | Var::Uniform(_) => None,
        }
    }

    /// The uniform value this is, or a floor or mod of.
    pub fn uniform(&self) -> Option<Value> {
        match self {
            Var::Uniform(v) => Some(*v),
            Var::Div(inner, _) | Var::Mod(inner, _) => inner.uniform(),
            Var::Tid(_) | Var::Ctaid(_) | Var::Iter(_) => None,
        }
    }

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
            Var::Uniform(v) => write!(f, "uniform({v:?})"),
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
            let (k, rest) = match SymExpr::split_const(c.clone()) {
                (i64::MIN, _) => (1, c.clone()),
                split => split,
            };
            let coeff = SymExpr::mul(SymExpr::Const(k.abs()), rest);
            let v = v.render(&name);
            let term = match coeff {
                SymExpr::Const(1) => v,
                SymExpr::Const(c) if c < 0 => format!("({coeff}) * {v}"),
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
            SymExpr::Const(c) if *c < 0 => format!("{out} - {}", c.unsigned_abs()),
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

    /// The least and the greatest value of the form when `bound` gives
    /// every variable's least and greatest value, and a `%ctaid` symbol's
    /// as its variable's, and no other symbol is left; `None` when one is
    /// unbounded, the whole `i128` line when a partial sum leaves it.
    pub fn range(&self, bound: &dyn Fn(&Var) -> Option<(i64, i64)>) -> Option<(i128, i128)> {
        let sym = |s: &str| {
            let axis = match s.strip_prefix("%ctaid.")? {
                "x" => Axis::X,
                "y" => Axis::Y,
                "z" => Axis::Z,
                _ => return None,
            };
            bound(&Var::Ctaid(axis))
        };
        let mut range = Some(self.base.range(&sym)?);
        let mut groups: BTreeMap<&Var, Vec<(&Var, i128)>> = BTreeMap::new();
        for (v, c) in &self.terms {
            let x = match v {
                Var::Div(x, _) | Var::Mod(x, _) => x,
                _ => v,
            };
            match c.as_const() {
                Some(c) => (groups.entry(x).or_default()).push((v, i128::from(c))),
                None => {
                    let (lo, hi) = var_range(v, bound)?;
                    let (a, b) = product(c.range(&sym)?, (i128::from(lo), i128::from(hi)));
                    range = range.and_then(|(x, y)| Some((x.checked_add(a)?, y.checked_add(b)?)));
                }
            }
        }
        for (x, terms) in &groups {
            let (a, b) = group_range(x, terms, bound)?;
            range = range.and_then(|(lo, hi)| Some((lo.checked_add(a)?, hi.checked_add(b)?)));
        }
        Some(range.unwrap_or((i128::MIN, i128::MAX)))
    }
}

impl Affine {
    /// The bits a value of the form may set, in two's complement. For
    /// every k: the bits from k up, and those of the remainder, the terms
    /// whose coefficients are not known multiples of 2^k and the base
    /// modulo 2^k, when it lies below 2^k, the rest being a multiple of
    /// 2^k that leaves the low k bits to it; and the whole form's bits
    /// when it is a bounded nonnegative sum. `largest` gives a
    /// variable's greatest value.
    pub fn may_set(&self, largest: &dyn Fn(&Var) -> Option<i64>) -> u64 {
        let all: Vec<_> = self.terms.iter().collect();
        let mut set = (self.base.as_const())
            .and_then(|b| bits(&all, b, largest))
            .unwrap_or(u64::MAX);
        for k in 1..63 {
            let p = 1i64 << k;
            let base = match self.base.as_const() {
                Some(c) => c.rem_euclid(p),
                None if self.base.divisible_by(p) => 0,
                None => continue,
            };
            let rest: Vec<_> = all
                .iter()
                .copied()
                .filter(|(_, c)| !c.divisible_by(p))
                .collect();
            if let Some(r) = bits(&rest, base, largest)
                && r >> k == 0
            {
                set &= r | (u64::MAX << k);
            }
        }
        set
    }
}

/// The bits a sum of `terms` and `base` may set, when the base is
/// nonnegative and each term a positive constant times a variable that
/// is never negative and has a greatest value: a term's from its
/// coefficient's lowest set bit to its greatest value's highest. Where
/// no two terms' bits meet the sum carries nothing and sets their
/// union; else from the lowest bit any sets up to the greatest sum's.
fn bits(
    terms: &[(&Var, &SymExpr)],
    base: i64,
    largest: &dyn Fn(&Var) -> Option<i64>,
) -> Option<u64> {
    let upto = |n: u128| match 128 - n.leading_zeros() {
        len if len >= 64 => u64::MAX,
        len => (1u64 << len) - 1,
    };
    let base = u64::try_from(base).ok()?;
    let (mut masks, mut total) = (vec![base], u128::from(base));
    for &(v, c) in terms {
        let c = u64::try_from(c.as_const()?).ok().filter(|&c| c > 0)?;
        let top = u128::from(c) * u128::from(u64::try_from(greatest(v, largest)?).ok()?);
        total = total.checked_add(top)?;
        masks.push(upto(top) & (u64::MAX << c.trailing_zeros()));
    }
    let union = masks.iter().fold(0, |a, m| a | m);
    if masks.iter().map(|m| m.count_ones()).sum::<u32>() == union.count_ones() {
        return Some(union);
    }
    let low = (masks.iter().filter(|&&m| m != 0))
        .map(|m| m.trailing_zeros())
        .min();
    Some(upto(total) & (u64::MAX << low.unwrap_or(0)))
}

/// A variable's greatest value when it is never negative: a mod's
/// modulus less one, a floor's operand's over its divisor, an index's
/// or a counter's from `largest`; a uniform variable's sign is unknown.
fn greatest(v: &Var, largest: &dyn Fn(&Var) -> Option<i64>) -> Option<i64> {
    match v {
        Var::Mod(_, m) => Some(m - 1),
        Var::Div(x, d) => Some(greatest(x, largest)?.div_euclid(*d)),
        Var::Tid(_) | Var::Ctaid(_) | Var::Iter(_) => largest(v),
        Var::Uniform(_) => None,
    }
}

/// The range of `Σ c·t` over terms `t` that are `x`, `⌊x/d⌋` or
/// `x mod d`: each term's range summed, and, when the floors and
/// remainders share one divisor `d`, no wider than the range of
/// `(c₁·d + c₂)·q + (c₁ + c₃)·r` over `x = d·q + r`, `q` and `r` taken
/// apart.
fn group_range(
    x: &Var,
    terms: &[(&Var, i128)],
    bound: &dyn Fn(&Var) -> Option<(i64, i64)>,
) -> Option<(i128, i128)> {
    let span = |c: i128, (lo, hi): (i64, i64)| {
        let (a, b) = (c.checked_mul(lo.into())?, c.checked_mul(hi.into())?);
        Some((a.min(b), a.max(b)))
    };
    let mut sum = Some((0i128, 0i128));
    let (mut d, mut c) = (None, [0i128; 3]);
    for &(v, k) in terms {
        let (a, b) = span(k, var_range(v, bound)?).expect("i64 by i64 fits i128");
        sum = sum.and_then(|(lo, hi)| Some((lo.checked_add(a)?, hi.checked_add(b)?)));
        let (slot, e) = match v {
            Var::Div(_, e) => (1, *e),
            Var::Mod(_, e) => (2, *e),
            _ => (0, 0),
        };
        c[slot] += k;
        if slot > 0 {
            d = if d.is_none_or(|d| d == e) {
                Some(e)
            } else {
                Some(0)
            };
        }
    }
    let sum = sum.unwrap_or((i128::MIN, i128::MAX));
    let joint = || {
        let d = d.filter(|&d| d > 0)?;
        let (lo, hi) = var_range(x, bound)?;
        let q = (lo.div_euclid(d), hi.div_euclid(d));
        let r = match q.0 == q.1 {
            true => (lo.rem_euclid(d), hi.rem_euclid(d)),
            false => (0, d - 1),
        };
        let (a, b) = span(c[0].checked_mul(d.into())?.checked_add(c[1])?, q)?;
        let (e, f) = span(c[0].checked_add(c[2])?, r)?;
        Some((a.checked_add(e)?, b.checked_add(f)?))
    };
    Some(match joint() {
        Some((a, b)) => (sum.0.max(a), sum.1.min(b)),
        None => sum,
    })
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
    fn a_range_takes_a_variable_with_its_floor_and_remainder() {
        let x = || Var::Ctaid(Axis::X);
        let floor = |d| Affine::var(Var::Div(Box::new(x()), d));
        let rem = |d| Affine::var(Var::Mod(Box::new(x()), d));
        let c = |k| SymExpr::Const(k);
        let grid = |lo, hi| move |v: &Var| (*v == x()).then_some((lo, hi));
        // scaled_mm's 1048576·(x mod 448), traced as x and ⌊x/448⌋.
        let a = Affine::var(x()).scale(c(1048576)) + floor(448).scale(c(-469762048));
        assert_eq!(a.range(&grid(0, 28671)), Some((0, 1048576 * 447)));
        // Each term's range stays a bound: 2x − ⌊x/4⌋ is 6, 7, 9 at x = 3, 4, 5.
        let b = Affine::var(x()).scale(c(2)) + floor(4).scale(c(-1));
        assert_eq!(b.range(&grid(3, 5)), Some((5, 10)));
        // One quotient: r runs from lo mod d to hi mod d.
        assert_eq!(b.range(&grid(4, 6)), Some((7, 11)));
        // A floor and remainder by one divisor, negative coefficients.
        let d = floor(8).scale(c(-8)) + rem(8).scale(c(-1)) + Affine::var(x());
        assert_eq!(d.range(&grid(-20, 100)), Some((0, 0)));
        // Two divisors stay per term.
        let e = Affine::var(x()) + floor(4).scale(c(-4)) + floor(8);
        assert_eq!(e.range(&grid(0, 15)), Some((-12, 16)));
    }

    #[test]
    fn may_set_takes_the_low_bits_from_the_bounded_part() {
        let c = |k| SymExpr::Const(k);
        let tid = || Var::Tid(Axis::X);
        let modulo = |m| Affine::var(Var::Mod(Box::new(tid()), m));
        let floor = |d| Affine::var(Var::Div(Box::new(tid()), d));
        let block = |v: &Var| matches!(v, Var::Tid(_)).then_some(127);
        // rope_norm's 64·%ctaid.x + (⌊%tid.x/8⌋ mod 16): bits 0–3 and 6 up.
        let rope = Affine::var(Var::Ctaid(Axis::X)).scale(c(64))
            + Affine::var(Var::Mod(Box::new(Var::Div(Box::new(tid()), 8)), 16));
        assert_eq!(rope.may_set(&block), !0x30);
        // A multiple of 64 of any sign, and a base below 0.
        let signed = Affine::var(Var::Iter(LoopId(0))).scale(c(-64))
            + modulo(16)
            + Affine::invariant(c(-64));
        assert_eq!(signed.may_set(&block), !0x30);
        // Disjoint terms add as an or; others carry.
        assert_eq!((modulo(4) + modulo(4).scale(c(4))).may_set(&block), 0xf);
        assert_eq!((modulo(4) + floor(32)).may_set(&block), 0x7);
        // 128·⌊%tid.x/32⌋ and 16·%tid.x below 128 share bits 7 and 8.
        assert_eq!(floor(32).scale(c(128)).may_set(&block), 0x180);
        assert_eq!(Affine::var(tid()).scale(c(16)).may_set(&block), 0x7f0);
        // A parameter's bits are unknown, a negative constant's all high.
        assert_eq!(
            Affine::invariant(SymExpr::sym("param_0")).may_set(&block),
            u64::MAX
        );
        assert_eq!(Affine::invariant(c(-4)).may_set(&block), !0x3);
        assert_eq!(Affine::invariant(c(0)).may_set(&block), 0);
        // A symbolic coefficient known a multiple of 64.
        let p = SymExpr::mul(c(64), SymExpr::sym("param_1"));
        assert_eq!(
            (Affine::var(Var::Ctaid(Axis::X)).scale(p) + modulo(16)).may_set(&block),
            !0x30
        );
        // A uniform value's remainder is bounded, its floor is not.
        let u = || {
            Box::new(Var::Uniform(crate::ssa::Value::Undef(
                crate::ptx::ir::Register::Cc,
            )))
        };
        assert_eq!(Affine::var(Var::Mod(u(), 8)).may_set(&block), 0x7);
        assert_eq!(
            (Affine::var(Var::Div(u(), 8)) + modulo(2)).may_set(&block),
            u64::MAX
        );
        // Greatest values whose sum leaves u128.
        let huge = |v: Var| Affine::var(v).scale(c(i64::MAX));
        let wide = huge(tid())
            + huge(Var::Tid(Axis::Y))
            + huge(Var::Tid(Axis::Z))
            + huge(Var::Ctaid(Axis::X));
        assert_eq!(wide.may_set(&|_| Some(i64::MAX)), u64::MAX);
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
    fn a_coefficient_or_base_of_minus_2_63_renders_whole() {
        let min = || SymExpr::Const(i64::MIN);
        let a = tid().scale(min()) + Affine::invariant(min());
        assert_eq!(
            a.to_string(),
            "(-9223372036854775808) * %tid.x - 9223372036854775808"
        );
        let b = tid() + Affine::var(Var::Ctaid(Axis::X)).scale(min());
        assert_eq!(b.to_string(), "%tid.x + (-9223372036854775808) * %ctaid.x");
        let c = SymExpr::add(SymExpr::sym("param_0"), min());
        assert_eq!(c.to_string(), "param_0 + -9223372036854775808");
        // −1 times −2^63 overflows and stays a product of the two.
        let negated = SymExpr::mul(SymExpr::Const(-1), min());
        let d = tid().scale(negated.clone());
        assert_eq!(d.to_string(), "-(-9223372036854775808) * %tid.x");
        assert_eq!(negated.to_string(), "-1 * -9223372036854775808");
        let e = SymExpr::add(SymExpr::sym("param_0"), negated);
        assert_eq!(e.to_string(), "param_0 - (-9223372036854775808)");
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
