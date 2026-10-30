//! AffineExpr expressions over the variables a kernel's scalars are affine in:
//! thread and CTA indices and loop iteration numbers, with symbolic
//! coefficients over the kernel parameters and the launch shape
//! (`K * %tid.y` is a term; `%ntid.x` is a symbol, bound when the
//! shape is known).
//! Trip matching, address analysis, and thread-participation analysis
//! read latch conditions, addresses, and branch conditions in this form.

use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant, product};
use crate::analysis::control_flow::loops::LoopId;
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
    Iteration(LoopId),
    /// `⌊v / d⌋`, d > 0: a 2D tile's thread row, `tid.x / 8`.
    FloorDiv(Box<Var>, i64),
    /// `v mod m`, m > 0: its thread column, `tid.x % 8`.
    Mod(Box<Var>, i64),
    /// A register value affine analysis cannot interpret, as each lane
    /// reads it. It is read so only where the lanes of a warp executing
    /// the read see it alike; a use that takes it as one value for
    /// several lanes checks that again where it reads it, for the lanes
    /// of a warp (an address's warp footprint, a shuffle) or every
    /// thread of the CTA (a trip count, a loop's footprint).
    Unknown(Value),
    /// An xor of fields of `%tid.x`.
    Swizzle(XorSwizzle),
}

/// `constant ⊕ ⊕ᵢ tᵢ·columns[i]` over the bits tᵢ of `%tid.x`, which is
/// below 1024 ("Maximum values of %ntid.{x,y,z}", PTX ISA §10.2).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct XorSwizzle {
    pub columns: [u64; 10],
    pub constant: u64,
}

impl XorSwizzle {
    pub fn xor(&self, other: &XorSwizzle) -> XorSwizzle {
        XorSwizzle {
            columns: std::array::from_fn(|i| self.columns[i] ^ other.columns[i]),
            constant: self.constant ^ other.constant,
        }
    }

    /// The value as a form: the constant's bits no column sets in the
    /// base, the rest a `Swizzle`.
    pub fn affine(self) -> AffineExpr {
        let image = self.columns.iter().fold(0, |a, &c| a | c);
        let base = AffineExpr::from_base(SymExpr::Const((self.constant & !image) as i64));
        match image {
            0 => base,
            _ => {
                let constant = self.constant & image;
                let columns = self.columns;
                base + AffineExpr::var(Var::Swizzle(XorSwizzle { columns, constant }))
            }
        }
    }

    pub fn eval(&self, tid: i64) -> i64 {
        let set = (0..10).filter(|&i| tid >> i & 1 == 1);
        set.fold(self.constant, |x, i| x ^ self.columns[i]) as i64
    }

    /// The least and greatest values over `%tid.x` in `tid`, or over every
    /// thread index.
    fn range(&self, tid: Option<(i64, i64)>) -> (i64, i64) {
        let (lo, hi) = tid.map_or((0, 1023), |(lo, hi)| (lo.max(0), hi.min(1023)));
        let values = (lo..=hi.max(lo)).map(|t| self.eval(t));
        (values.clone().min().unwrap_or(0), values.max().unwrap_or(0))
    }

    /// The xor of `2^p · (⌊%tid.x/2^i⌋ mod 2^n)` over each run of n thread
    /// bits from i that set the bits from p, and of the constant.
    fn render(&self, name: &impl Fn(&Var) -> String) -> String {
        let mut pairs: Vec<(i64, usize)> = (0..10)
            .flat_map(|i| {
                let column = self.columns[i];
                (0..64i64)
                    .filter(move |p| column >> p & 1 == 1)
                    .map(move |p| (p - i as i64, i))
            })
            .collect();
        pairs.sort_unstable();
        let mut runs: Vec<(i64, usize, usize)> = Vec::new();
        for (shift, i) in pairs {
            match runs.last_mut() {
                Some((s, from, n)) if *s == shift && *from + *n == i => *n += 1,
                _ => runs.push((shift, i, 1)),
            }
        }
        let tid = name(&Var::Tid(Axis::X));
        let mut fields: Vec<String> = (runs.iter())
            .map(|&(shift, i, n)| {
                let x = match i {
                    0 => tid.clone(),
                    _ => format!("⌊{tid}/{}⌋", 1 << i),
                };
                let x = match i + n {
                    10 => x,
                    _ => format!("({x} mod {})", 1 << n),
                };
                match shift + i as i64 {
                    0 => x,
                    p => format!("{} * {x}", 1i64 << p),
                }
            })
            .collect();
        if self.constant != 0 {
            fields.push(self.constant.to_string());
        }
        format!("({})", fields.join(" ⊕ "))
    }
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
            Var::Iteration(l) => Some(*l),
            Var::FloorDiv(inner, _) | Var::Mod(inner, _) => inner.counter(),
            Var::Tid(_) | Var::Ctaid(_) | Var::Unknown(_) | Var::Swizzle(_) => None,
        }
    }

    /// The unknown value this is, or a floor or mod of.
    pub fn unknown(&self) -> Option<Value> {
        match self {
            Var::Unknown(v) => Some(*v),
            Var::FloorDiv(inner, _) | Var::Mod(inner, _) => inner.unknown(),
            Var::Tid(_) | Var::Ctaid(_) | Var::Iteration(_) | Var::Swizzle(_) => None,
        }
    }

    /// The variable's value, `leaf` giving each thread or CTA index's,
    /// counter's and uniform value's.
    pub fn eval(&self, leaf: &dyn Fn(&Var) -> i64) -> i64 {
        match self {
            Var::FloorDiv(v, d) => v.eval(leaf).div_euclid(*d),
            Var::Mod(v, m) => v.eval(leaf).rem_euclid(*m),
            Var::Swizzle(b) => b.eval(leaf(&Var::Tid(Axis::X))),
            v => leaf(v),
        }
    }

    /// The variable with the leaf indices printed by `name`.
    pub fn render(&self, name: &impl Fn(&Var) -> String) -> String {
        match self {
            Var::FloorDiv(v, d) => format!("⌊{}/{d}⌋", v.render(name)),
            Var::Mod(v, m) => format!("({} mod {m})", v.render(name)),
            Var::Swizzle(b) => b.render(name),
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
            Var::Iteration(l) => write!(f, "k[loop {}]", l.0),
            Var::FloorDiv(v, d) => write!(f, "⌊{v}/{d}⌋"),
            Var::Mod(v, m) => write!(f, "({v} mod {m})"),
            Var::Unknown(v) => write!(f, "uniform({v:?})"),
            Var::Swizzle(b) => f.write_str(&b.render(&|v| v.to_string())),
        }
    }
}

/// A signed integer comparison, `a cmp b` (PTX ISA §9.3.1.1, Table 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    /// The comparison a `setp` modifier names: `eq`, `ne`, `lt`, `le`,
    /// `gt` or `ge`.
    pub fn from_name(name: &str) -> Option<Cmp> {
        Some(match name {
            "eq" => Cmp::Eq,
            "ne" => Cmp::Ne,
            "lt" => Cmp::Lt,
            "le" => Cmp::Le,
            "gt" => Cmp::Gt,
            "ge" => Cmp::Ge,
            _ => return None,
        })
    }

    /// The comparison that holds exactly when this one does not (LLVM's
    /// `CmpInst::getInversePredicate`).
    pub fn inverse(self) -> Cmp {
        match self {
            Cmp::Eq => Cmp::Ne,
            Cmp::Ne => Cmp::Eq,
            Cmp::Lt => Cmp::Ge,
            Cmp::Ge => Cmp::Lt,
            Cmp::Le => Cmp::Gt,
            Cmp::Gt => Cmp::Le,
        }
    }
}

/// `Σ coeff·var + base`; a missing variable has coefficient 0.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineExpr {
    pub terms: BTreeMap<Var, SymExpr>,
    pub base: SymExpr,
}

impl AffineExpr {
    pub fn from_base(base: SymExpr) -> AffineExpr {
        AffineExpr {
            terms: BTreeMap::new(),
            base,
        }
    }

    pub fn var(v: Var) -> AffineExpr {
        AffineExpr::term(v, SymExpr::Const(1))
    }

    pub fn term(v: Var, coeff: SymExpr) -> AffineExpr {
        let mut a = AffineExpr::from_base(SymExpr::Const(0));
        if coeff != SymExpr::Const(0) {
            a.terms.insert(v, coeff);
        }
        a
    }

    /// Multiply by a symbolic or constant value.
    pub fn scale(self, c: SymExpr) -> AffineExpr {
        if c == SymExpr::Const(0) {
            return AffineExpr::from_base(SymExpr::Const(0));
        }
        AffineExpr {
            terms: self
                .terms
                .into_iter()
                .map(|(v, k)| (v, SymExpr::mul(c.clone(), k)))
                .collect(),
            base: SymExpr::mul(c, self.base),
        }
    }

    /// Substitute bound symbols in every coefficient and the base.
    pub fn bind(&self, bindings: &std::collections::HashMap<SymbolicConstant, i64>) -> AffineExpr {
        let mut out = AffineExpr::from_base(self.base.bind(bindings));
        for (v, c) in &self.terms {
            out = out + AffineExpr::term(v.clone(), c.bind(bindings));
        }
        out
    }

    /// Divide exactly by `d`, when every coefficient and the base are
    /// known multiples of it.
    pub fn div_exact(self, d: i64) -> Option<AffineExpr> {
        if !self
            .terms
            .values()
            .chain([&self.base])
            .all(|e| e.divisible_by(d))
        {
            return None;
        }
        Some(AffineExpr {
            terms: self
                .terms
                .into_iter()
                .map(|(v, c)| (v, SymExpr::floor_div(c, d)))
                .collect(),
            base: SymExpr::floor_div(self.base, d),
        })
    }

    /// `⌊self / d⌋`, d > 0: exact when every part is a multiple of d; the
    /// multiple of d divided and the floor of what remains, as `split`
    /// takes it; a floor division when symbolic or constant. None
    /// otherwise.
    pub fn div_const(
        self,
        d: i64,
        bound: &dyn Fn(&Var) -> Option<(i64, i64)>,
    ) -> Option<AffineExpr> {
        if self.is_symbolic_or_constant() {
            return Some(AffineExpr::from_base(SymExpr::floor_div(self.base, d)));
        }
        if let Some(exact) = self.clone().div_exact(d) {
            return Some(exact);
        }
        let (q, floor, _) = self.split(d, bound)?;
        Some(q + floor)
    }

    /// `self mod m`, m > 0: zero when every part is a multiple of m; the
    /// remainder of what remains past the multiple of m, as `split` takes
    /// it; a symbolic modulus when symbolic or constant. None otherwise.
    pub fn modulo_const(
        self,
        m: i64,
        bound: &dyn Fn(&Var) -> Option<(i64, i64)>,
    ) -> Option<AffineExpr> {
        if self.is_symbolic_or_constant() {
            return Some(AffineExpr::from_base(SymExpr::modulo(self.base, m)));
        }
        if self.clone().div_exact(m).is_some() {
            return Some(AffineExpr::from_base(SymExpr::Const(0)));
        }
        let (_, _, rem) = self.split(m, bound)?;
        Some(rem)
    }

    /// `self − self mod m`, the multiple of m at or below it: m times the
    /// quotient `div_const` gives.
    pub fn align_down(
        self,
        m: i64,
        bound: &dyn Fn(&Var) -> Option<(i64, i64)>,
    ) -> Option<AffineExpr> {
        if self.is_symbolic_or_constant() {
            let rem = SymExpr::modulo(self.base.clone(), m);
            return Some(AffineExpr::from_base(SymExpr::sub(self.base, rem)));
        }
        if self.clone().div_exact(m).is_some() {
            return Some(self);
        }
        let (q, floor, _) = self.split(m, bound)?;
        Some((q + floor).scale(SymExpr::Const(m)))
    }

    /// `(q, ⌊r/d⌋, r mod d)` for the form `d·q + r`: q holds the terms
    /// whose coefficients are known multiples of d and the multiple of d
    /// in the base, r at most one other term `c·v`, c a positive divisor
    /// of d, and a constant k in [0, c), or one in [0, d) alone. A k of c
    /// or more is read where `c·v + k` stays in [0, d) over v's range,
    /// from `bound` or the one a remainder has whatever its operand.
    fn split(
        &self,
        d: i64,
        bound: &dyn Fn(&Var) -> Option<(i64, i64)>,
    ) -> Option<(AffineExpr, AffineExpr, AffineExpr)> {
        let mut q = AffineExpr::from_base(SymExpr::Const(0));
        let mut rest = None;
        for (v, c) in &self.terms {
            if c.divisible_by(d) {
                q = q + AffineExpr::term(v.clone(), SymExpr::floor_div(c.clone(), d));
            } else if rest.is_some() {
                return None;
            } else {
                rest = Some((v, c.as_const().filter(|&c| c > 0 && d % c == 0)?));
            }
        }
        let summands = match &self.base {
            SymExpr::Sum(ts) => ts.clone(),
            base => vec![base.clone()],
        };
        let (mut k, mut whole, mut multiple) = (0i128, SymExpr::Const(0), SymExpr::Const(0));
        for t in summands {
            match t.as_const() {
                Some(c) => {
                    whole = SymExpr::add(whole, SymExpr::Const(c.div_euclid(d)));
                    k += i128::from(c.rem_euclid(d));
                }
                None => multiple = SymExpr::add(multiple, t),
            }
        }
        let carry = SymExpr::Const(i64::try_from(k / i128::from(d)).ok()?);
        let k = i64::try_from(k % i128::from(d)).ok()?;
        if !multiple.divisible_by(d) {
            return None;
        }
        let whole = SymExpr::add(whole, carry);
        q.base = SymExpr::add(SymExpr::floor_div(multiple, d), whole);
        let rem = AffineExpr::from_base(SymExpr::Const(k));
        let Some((v, c)) = rest else {
            return Some((q, AffineExpr::from_base(SymExpr::Const(0)), rem));
        };
        if k >= c {
            let (lo, hi) = var_range(v, bound).or_else(|| own_range(v))?;
            let at = |x: i64| i128::from(c) * i128::from(x) + i128::from(k);
            return (at(lo) >= 0 && at(hi) < i128::from(d)).then(|| {
                let r = AffineExpr::term(v.clone(), SymExpr::Const(c)) + rem;
                (q, AffineExpr::from_base(SymExpr::Const(0)), r)
            });
        }
        let floor = AffineExpr::var(Var::FloorDiv(Box::new(v.clone()), d / c));
        let modulo = AffineExpr::term(Var::Mod(Box::new(v.clone()), d / c), SymExpr::Const(c));
        Some((q, floor, modulo + rem))
    }

    /// Substitute every variable by a form; None if any substitution is.
    pub fn map_vars(&self, f: impl Fn(&Var) -> Option<AffineExpr>) -> Option<AffineExpr> {
        let mut out = AffineExpr::from_base(self.base.clone());
        for (v, c) in &self.terms {
            out = out + f(v)?.scale(c.clone());
        }
        Some(out)
    }

    /// The form with each term in a variable `leaf` names, or a floor or
    /// mod of one, as that symbol in its base.
    pub fn as_symbols(mut self, leaf: &dyn Fn(&Var) -> Option<SymExpr>) -> AffineExpr {
        fn symbol(v: &Var, leaf: &dyn Fn(&Var) -> Option<SymExpr>) -> Option<SymExpr> {
            match v {
                Var::FloorDiv(v, d) => symbol(v, leaf).map(|e| SymExpr::floor_div(e, *d)),
                Var::Mod(v, m) => symbol(v, leaf).map(|e| SymExpr::modulo(e, *m)),
                v => leaf(v),
            }
        }
        for (v, c) in std::mem::take(&mut self.terms) {
            match symbol(&v, leaf) {
                Some(e) => self.base = SymExpr::add(self.base, SymExpr::mul(c, e)),
                None => {
                    self.terms.insert(v, c);
                }
            }
        }
        self
    }

    /// No term: the base alone, of symbols and constants (MLIR's
    /// `AffineExpr::isSymbolicOrConstant`).
    pub fn is_symbolic_or_constant(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn as_const(&self) -> Option<i64> {
        self.is_symbolic_or_constant()
            .then(|| self.base.as_const())
            .flatten()
    }
}

impl std::ops::Add for AffineExpr {
    type Output = AffineExpr;

    fn add(mut self, other: AffineExpr) -> AffineExpr {
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

impl std::ops::Sub for AffineExpr {
    type Output = AffineExpr;

    fn sub(self, other: AffineExpr) -> AffineExpr {
        self + other.scale(SymExpr::Const(-1))
    }
}

impl AffineExpr {
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

    /// The least and the greatest value of the form when `bound` gives
    /// every variable's least and greatest value, and a `%ctaid` symbol's
    /// as its variable's, and no other symbol is left; `None` when one is
    /// unbounded, the whole `i128` line when a partial sum leaves it.
    pub fn range(&self, bound: &dyn Fn(&Var) -> Option<(i64, i64)>) -> Option<(i128, i128)> {
        let sym = |s: &SymbolicConstant| match s {
            SymbolicConstant::SpecialReg(reg @ SpecialReg::Vector(SpecialVector::Ctaid, _)) => {
                bound(&Var::special(*reg)?)
            }
            _ => None,
        };
        let mut range = match self.base.span(&sym) {
            Ok(r) => Some(r),
            Err(true) => None,
            Err(false) => return None,
        };
        let mut groups: BTreeMap<&Var, Vec<(&Var, i128)>> = BTreeMap::new();
        for (v, c) in &self.terms {
            let x = match v {
                Var::FloorDiv(x, _) | Var::Mod(x, _) => x,
                _ => v,
            };
            match c.as_const() {
                Some(c) => (groups.entry(x).or_default()).push((v, i128::from(c))),
                None => {
                    let (lo, hi) = var_range(v, bound)?;
                    let term = match c.span(&sym) {
                        Ok(c) => product(c, (i128::from(lo), i128::from(hi))),
                        Err(true) => None,
                        Err(false) => return None,
                    };
                    range = (range.zip(term))
                        .and_then(|((x, y), (a, b))| Some((x.checked_add(a)?, y.checked_add(b)?)));
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

impl Var {
    /// The variable as an xor of `%tid.x`'s bits: the thread index, a
    /// floor or remainder of one by a power of two, or a `Swizzle`.
    fn xor_swizzle(&self) -> Option<XorSwizzle> {
        let power = |n: i64| u64::try_from(n).ok().filter(|n| n.is_power_of_two());
        match self {
            Var::Tid(Axis::X) => Some(XorSwizzle {
                columns: std::array::from_fn(|i| 1 << i),
                constant: 0,
            }),
            Var::FloorDiv(v, d) => {
                let k = power(*d)?.trailing_zeros();
                let b = v.xor_swizzle()?;
                Some(XorSwizzle {
                    columns: b.columns.map(|c| c >> k),
                    constant: b.constant >> k,
                })
            }
            Var::Mod(v, m) => {
                let mask = power(*m)? - 1;
                let b = v.xor_swizzle()?;
                Some(XorSwizzle {
                    columns: b.columns.map(|c| c & mask),
                    constant: b.constant & mask,
                })
            }
            Var::Swizzle(b) => Some(b.clone()),
            _ => None,
        }
    }
}

impl AffineExpr {
    /// The form as an xor of `%tid.x`'s bits: a constant base and terms
    /// `Var::xor_swizzle` reads with power-of-two coefficients, no two of
    /// which, nor the base, may set one bit, so that their sum is their
    /// xor; below 2^62.
    pub fn xor_swizzle(&self) -> Option<XorSwizzle> {
        let constant = u64::try_from(self.base.as_const()?).ok()?;
        let mut out = XorSwizzle {
            columns: [0; 10],
            constant,
        };
        let mut used = constant;
        for (v, c) in &self.terms {
            let c = u64::try_from(c.as_const()?)
                .ok()
                .filter(|c| c.is_power_of_two())?;
            let b = v.xor_swizzle()?;
            let image = b.columns.iter().fold(b.constant, |a, &c| a | c);
            let shifted = image.checked_mul(c).filter(|&s| s >> 62 == 0)?;
            if used & shifted != 0 {
                return None;
            }
            used |= shifted;
            for (to, from) in out.columns.iter_mut().zip(b.columns) {
                *to |= from * c;
            }
            out.constant |= b.constant * c;
        }
        (used >> 62 == 0).then_some(out)
    }

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
        Var::FloorDiv(x, d) => Some(greatest(x, largest)?.div_euclid(*d)),
        Var::Swizzle(b) => Some(b.range(largest(&Var::Tid(Axis::X)).map(|hi| (0, hi))).1),
        Var::Tid(_) | Var::Ctaid(_) | Var::Iteration(_) => largest(v),
        Var::Unknown(_) => None,
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
            Var::FloorDiv(_, e) => (1, *e),
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
/// The values a remainder, or a floor of one, takes whatever its operand.
fn own_range(v: &Var) -> Option<(i64, i64)> {
    match v {
        Var::Mod(_, m) => Some((0, m - 1)),
        Var::FloorDiv(inner, d) => {
            own_range(inner).map(|(lo, hi)| (lo.div_euclid(*d), hi.div_euclid(*d)))
        }
        Var::Swizzle(b) => Some(b.range(None)),
        _ => None,
    }
}

fn var_range(v: &Var, bound: &dyn Fn(&Var) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
    match v {
        Var::FloorDiv(inner, d) => {
            let (lo, hi) = var_range(inner, bound)?;
            Some((lo.div_euclid(*d), hi.div_euclid(*d)))
        }
        Var::Mod(inner, m) => match var_range(inner, bound)? {
            (lo, hi) if lo >= 0 && hi < *m => Some((lo, hi)),
            _ => Some((0, m - 1)),
        },
        Var::Swizzle(b) => Some(b.range(bound(&Var::Tid(Axis::X)))),
        _ => bound(v),
    }
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render(|v| v.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free(_: &Var) -> Option<(i64, i64)> {
        None
    }
    #[test]
    fn lane_bits_evaluate_render_and_bound_an_xor_of_fields() {
        // The Gluon swizzle ((t << 4) & 2032) ^ ((t << 1) & 48)
        // (fp8_gemm_kernel.c_fc.sm_89.ptx:213): thread bit i sets bit i + 4,
        // and bits 3 and 4 also bits 4 and 5.
        let mut columns: [u64; 10] = std::array::from_fn(|i| (1 << (i + 4)) & 2032);
        columns[3] |= 1 << 4;
        columns[4] |= 1 << 5;
        let swizzle = XorSwizzle {
            columns,
            constant: 0,
        };
        let want = |t: i64| ((t << 4) & 2032) ^ ((t << 1) & 48);
        assert!((0..1024).all(|t| swizzle.eval(t) == want(t)));
        let shifted = swizzle.xor(&XorSwizzle {
            columns: [0; 10],
            constant: 64,
        });
        assert!((0..1024).all(|t| shifted.eval(t) == want(t) ^ 64));
        let bits = Var::Swizzle(swizzle);
        assert_eq!(
            bits.to_string(),
            "(16 * (⌊%tid.x/8⌋ mod 4) ⊕ 16 * (%tid.x mod 128))"
        );
        let block = |v: &Var| (*v == Var::Tid(Axis::X)).then_some((0, 127));
        let range = (0..128).map(want);
        let expected = (
            range.clone().min().unwrap().into(),
            range.max().unwrap().into(),
        );
        assert_eq!(AffineExpr::var(bits).range(&block), Some(expected));
    }

    #[test]
    fn an_xor_of_thread_index_fields_reads_as_lane_bits() {
        let c = |n: i64| SymExpr::Const(n);
        let x = || Box::new(Var::Tid(Axis::X));
        let rem = |m, k| AffineExpr::var(Var::Mod(x(), m)).scale(c(k));
        let field =
            |d, m, k| AffineExpr::var(Var::Mod(Box::new(Var::FloorDiv(x(), d)), m)).scale(c(k));
        // The Gluon swizzle's operands as traced: (t << 4) & 2032 and
        // (t << 1) & 48 (fp8_gemm_kernel.c_fc.sm_89.ptx:213).
        let swizzle =
            (rem(128, 16).xor_swizzle().unwrap()).xor(&field(8, 4, 16).xor_swizzle().unwrap());
        let want = |t: i64| ((t << 4) & 2032) ^ ((t << 1) & 48);
        assert!((0..1024).all(|t| swizzle.eval(t) == want(t)));
        // A floor of the xor is one too, and a sum with a disjoint field.
        let bits = Var::Swizzle(swizzle);
        let floor = AffineExpr::var(Var::FloorDiv(Box::new(bits.clone()), 16)) + rem(2, 4096);
        let read = floor.xor_swizzle().unwrap();
        assert!((0..1024).all(|t| read.eval(t) == (want(t) >> 4) + 4096 * (t % 2)));
        // A sum whose fields may set one bit is no xor, nor an odd multiple.
        assert_eq!((rem(4, 1) + rem(2, 2)).xor_swizzle(), None);
        assert_eq!(
            AffineExpr::var(Var::Tid(Axis::X)).scale(c(3)).xor_swizzle(),
            None
        );
        assert_eq!(
            (AffineExpr::var(bits) + AffineExpr::from_base(c(16))).xor_swizzle(),
            None
        );
    }

    #[test]
    fn a_range_spans_each_term_at_both_ends_of_its_variable() {
        let tid = AffineExpr::var(Var::Tid(Axis::X));
        let k = AffineExpr::var(Var::Iteration(LoopId(0)));
        let bound = |v: &Var| match v {
            Var::Tid(Axis::X) => Some((0, 127)),
            Var::Iteration(_) => Some((1, 8)),
            _ => None,
        };
        let a = tid.scale(SymExpr::Const(4))
            + k.scale(SymExpr::Const(-16))
            + AffineExpr::from_base(SymExpr::Const(100));
        assert_eq!(a.range(&bound), Some((100 - 128, 100 + 508 - 16)));
        let floor = AffineExpr::var(Var::FloorDiv(Box::new(Var::Tid(Axis::X)), 32));
        assert_eq!(floor.range(&bound), Some((0, 3)));
        let modulo = AffineExpr::var(Var::Mod(Box::new(Var::Iteration(LoopId(0))), 4));
        assert_eq!(modulo.range(&bound), Some((0, 3)));
        let ctaid = AffineExpr::var(Var::Ctaid(Axis::X));
        assert_eq!(ctaid.range(&bound), None);
        let param = AffineExpr::from_base(SymExpr::Sym(SymbolicConstant::KernelParam(0)));
        assert_eq!(param.range(&bound), None);
    }

    #[test]
    fn a_range_takes_a_variable_with_its_floor_and_remainder() {
        let x = || Var::Ctaid(Axis::X);
        let floor = |d| AffineExpr::var(Var::FloorDiv(Box::new(x()), d));
        let rem = |d| AffineExpr::var(Var::Mod(Box::new(x()), d));
        let c = |k| SymExpr::Const(k);
        let grid = |lo, hi| move |v: &Var| (*v == x()).then_some((lo, hi));
        // scaled_mm's 1048576·(x mod 448), traced as x and ⌊x/448⌋.
        let a = AffineExpr::var(x()).scale(c(1048576)) + floor(448).scale(c(-469762048));
        assert_eq!(a.range(&grid(0, 28671)), Some((0, 1048576 * 447)));
        // Each term's range stays a bound: 2x − ⌊x/4⌋ is 6, 7, 9 at x = 3, 4, 5.
        let b = AffineExpr::var(x()).scale(c(2)) + floor(4).scale(c(-1));
        assert_eq!(b.range(&grid(3, 5)), Some((5, 10)));
        // One quotient: r runs from lo mod d to hi mod d.
        assert_eq!(b.range(&grid(4, 6)), Some((7, 11)));
        // A floor and remainder by one divisor, negative coefficients.
        let d = floor(8).scale(c(-8)) + rem(8).scale(c(-1)) + AffineExpr::var(x());
        assert_eq!(d.range(&grid(-20, 100)), Some((0, 0)));
        // Two divisors stay per term.
        let e = AffineExpr::var(x()) + floor(4).scale(c(-4)) + floor(8);
        assert_eq!(e.range(&grid(0, 15)), Some((-12, 16)));
    }

    #[test]
    fn may_set_takes_the_low_bits_from_the_bounded_part() {
        let c = |k| SymExpr::Const(k);
        let tid = || Var::Tid(Axis::X);
        let modulo = |m| AffineExpr::var(Var::Mod(Box::new(tid()), m));
        let floor = |d| AffineExpr::var(Var::FloorDiv(Box::new(tid()), d));
        let block = |v: &Var| matches!(v, Var::Tid(_)).then_some(127);
        // rope_norm's 64·%ctaid.x + (⌊%tid.x/8⌋ mod 16): bits 0–3 and 6 up.
        let rope = AffineExpr::var(Var::Ctaid(Axis::X)).scale(c(64))
            + AffineExpr::var(Var::Mod(Box::new(Var::FloorDiv(Box::new(tid()), 8)), 16));
        assert_eq!(rope.may_set(&block), !0x30);
        // A multiple of 64 of any sign, and a base below 0.
        let signed = AffineExpr::var(Var::Iteration(LoopId(0))).scale(c(-64))
            + modulo(16)
            + AffineExpr::from_base(c(-64));
        assert_eq!(signed.may_set(&block), !0x30);
        // Disjoint terms add as an or; others carry.
        assert_eq!((modulo(4) + modulo(4).scale(c(4))).may_set(&block), 0xf);
        assert_eq!((modulo(4) + floor(32)).may_set(&block), 0x7);
        // 128·⌊%tid.x/32⌋ and 16·%tid.x below 128 share bits 7 and 8.
        assert_eq!(floor(32).scale(c(128)).may_set(&block), 0x180);
        assert_eq!(AffineExpr::var(tid()).scale(c(16)).may_set(&block), 0x7f0);
        // A parameter's bits are unknown, a negative constant's all high.
        assert_eq!(
            AffineExpr::from_base(SymExpr::Sym(SymbolicConstant::KernelParam(0))).may_set(&block),
            u64::MAX
        );
        assert_eq!(AffineExpr::from_base(c(-4)).may_set(&block), !0x3);
        assert_eq!(AffineExpr::from_base(c(0)).may_set(&block), 0);
        // A symbolic coefficient known a multiple of 64.
        let p = SymExpr::mul(c(64), SymExpr::Sym(SymbolicConstant::KernelParam(1)));
        assert_eq!(
            (AffineExpr::var(Var::Ctaid(Axis::X)).scale(p) + modulo(16)).may_set(&block),
            !0x30
        );
        // A uniform value's remainder is bounded, its floor is not.
        let u = || {
            Box::new(Var::Unknown(crate::ssa::Value::Undef(
                crate::ptx::ir::Register::Cc,
            )))
        };
        assert_eq!(AffineExpr::var(Var::Mod(u(), 8)).may_set(&block), 0x7);
        assert_eq!(
            (AffineExpr::var(Var::FloorDiv(u(), 8)) + modulo(2)).may_set(&block),
            u64::MAX
        );
        // Greatest values whose sum leaves u128.
        let huge = |v: Var| AffineExpr::var(v).scale(c(i64::MAX));
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
            .fold(AffineExpr::from_base(SymExpr::Const(0)), |a, (c, l)| {
                a + AffineExpr::var(Var::Iteration(LoopId(l))).scale(SymExpr::Const(c))
            });
        let bound = |_: &Var| Some((m, m));
        assert_eq!(a.range(&bound), Some((i128::MIN, i128::MAX)));
    }

    fn tid() -> AffineExpr {
        AffineExpr::var(Var::Tid(Axis::X))
    }

    #[test]
    fn a_coefficient_or_base_of_minus_2_63_renders_whole() {
        let min = || SymExpr::Const(i64::MIN);
        let a = tid().scale(min()) + AffineExpr::from_base(min());
        assert_eq!(
            a.to_string(),
            "(-9223372036854775808) * %tid.x - 9223372036854775808"
        );
        let b = tid() + AffineExpr::var(Var::Ctaid(Axis::X)).scale(min());
        assert_eq!(b.to_string(), "%tid.x + (-9223372036854775808) * %ctaid.x");
        let c = SymExpr::add(SymExpr::Sym(SymbolicConstant::KernelParam(0)), min());
        assert_eq!(c.to_string(), "param_0 + -9223372036854775808");
        // −1 times −2^63 overflows and stays a product of the two.
        let negated = SymExpr::mul(SymExpr::Const(-1), min());
        let d = tid().scale(negated.clone());
        assert_eq!(d.to_string(), "-(-9223372036854775808) * %tid.x");
        assert_eq!(negated.to_string(), "-1 * -9223372036854775808");
        let e = SymExpr::add(SymExpr::Sym(SymbolicConstant::KernelParam(0)), negated);
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
    fn a_floor_or_mod_divides_the_multiple_and_reads_the_rest() {
        let c = |k| SymExpr::Const(k);
        let x = || Var::Tid(Axis::X);
        let text = |a: Option<AffineExpr>| a.map(|a| a.to_string());
        // 01588.ptx's ((%tid.x & 127) | 128) >> 4 and its alignment.
        let rows = AffineExpr::var(Var::Mod(Box::new(x()), 128)) + AffineExpr::from_base(c(128));
        assert_eq!(
            text(rows.clone().div_const(16, &free)).as_deref(),
            Some("⌊(%tid.x mod 128)/16⌋ + 8")
        );
        assert_eq!(
            text(rows.align_down(16, &free)).as_deref(),
            Some("16 * ⌊(%tid.x mod 128)/16⌋ + 128")
        );
        // 00203.ptx's ((%ctaid.y << 6) | (%tid.x & 63)) >> 1.
        let row = AffineExpr::var(Var::Ctaid(Axis::Y)).scale(c(64))
            + AffineExpr::var(Var::Mod(Box::new(x()), 64));
        assert_eq!(
            text(row.div_const(2, &free)).as_deref(),
            Some("32 * %ctaid.y + ⌊(%tid.x mod 64)/2⌋")
        );
        // Lane 13 of a warp, 32·w + 13, and 4·w + 1: a constant below the
        // coefficient of the term the divisor does not divide.
        let w = || AffineExpr::var(Var::FloorDiv(Box::new(x()), 32));
        let lane = w().scale(c(32)) + AffineExpr::from_base(c(13));
        assert_eq!(
            text(lane.clone().div_const(8, &free)).as_deref(),
            Some("4 * ⌊%tid.x/32⌋ + 1")
        );
        assert_eq!(text(lane.modulo_const(8, &free)).as_deref(), Some("5"));
        let quad = w().scale(c(4)) + AffineExpr::from_base(c(1));
        assert_eq!(
            text(quad.clone().modulo_const(16, &free)).as_deref(),
            Some("4 * (⌊%tid.x/32⌋ mod 4) + 1")
        );
        assert_eq!(
            text(quad.div_const(16, &free)).as_deref(),
            Some("⌊⌊%tid.x/32⌋/4⌋")
        );
        // A symbolic coefficient known a multiple of the divisor.
        let p = SymExpr::mul(c(16), SymExpr::Sym(SymbolicConstant::KernelParam(0)));
        let scaled = AffineExpr::term(x(), p) + AffineExpr::from_base(c(1));
        assert_eq!(
            text(scaled.div_const(16, &free)).as_deref(),
            Some("param_0 * %tid.x")
        );
        // A constant at or above the coefficient, (%tid.x − 4) mod 8 and
        // (k − 1) mod 2; a negative or non-dividing coefficient; two terms
        // the divisor does not divide; a base not known a multiple.
        let tid = || AffineExpr::var(x());
        let k = AffineExpr::var(Var::Iteration(LoopId(0)));
        assert_eq!(
            (tid() + AffineExpr::from_base(c(-4))).modulo_const(8, &free),
            None
        );
        assert_eq!(
            (k + AffineExpr::from_base(c(-1))).modulo_const(2, &free),
            None
        );
        assert_eq!(tid().scale(c(-2)).div_const(4, &free), None);
        assert_eq!(tid().scale(c(3)).div_const(8, &free), None);
        assert_eq!(
            (tid() + AffineExpr::var(Var::Ctaid(Axis::X))).div_const(2, &free),
            None
        );
        let param = AffineExpr::from_base(SymExpr::Sym(SymbolicConstant::KernelParam(1)));
        assert_eq!((tid() + param).div_const(16, &free), None);
        let wide = tid() + AffineExpr::from_base(SymExpr::add(c(i64::MAX - 7), c(8)));
        assert!(matches!(&wide.base, SymExpr::Sum(ts) if ts.len() == 2));
        assert_eq!(
            text(wide.clone().modulo_const(8, &free)).as_deref(),
            Some("(%tid.x mod 8)")
        );
        assert_eq!(
            text(wide.div_const(8, &free)).as_deref(),
            Some("⌊%tid.x/8⌋ + 1152921504606846976")
        );
    }

    /// A constant at or past the coefficient of the term the divisor does
    /// not divide reads where that term's range keeps the remainder below
    /// the divisor: 01890.ptx's (64·%ctaid.y | ⌊%tid.x/8⌋ mod 16 | 16)
    /// rem 64 whatever the launch, ⌊%tid.x/64⌋ + 62 in blocks of 128.
    #[test]
    fn an_offset_the_terms_range_keeps_below_the_divisor_reads() {
        let c = |k| SymExpr::Const(k);
        let x = || Var::Tid(Axis::X);
        let text = |a: Option<AffineExpr>| a.map(|a| a.to_string());
        let row = AffineExpr::var(Var::Ctaid(Axis::Y)).scale(c(64))
            + AffineExpr::var(Var::Mod(Box::new(Var::FloorDiv(Box::new(x()), 8)), 16))
            + AffineExpr::from_base(c(16));
        assert_eq!(
            text(row.clone().modulo_const(64, &free)).as_deref(),
            Some("(⌊%tid.x/8⌋ mod 16) + 16")
        );
        assert_eq!(text(row.div_const(64, &free)).as_deref(), Some("%ctaid.y"));
        let half = AffineExpr::var(Var::FloorDiv(Box::new(x()), 64)) + AffineExpr::from_base(c(62));
        let block = |v: &Var| (*v == x()).then_some((0, 127));
        assert_eq!(
            text(half.clone().modulo_const(64, &block)).as_deref(),
            Some("⌊%tid.x/64⌋ + 62")
        );
        assert_eq!(half.modulo_const(64, &free), None);
        let wraps = AffineExpr::var(Var::Mod(Box::new(Var::FloorDiv(Box::new(x()), 8)), 32))
            + AffineExpr::from_base(c(48));
        assert_eq!(wraps.modulo_const(64, &free), None);
    }
    #[test]
    fn terms_combine_cancel_and_scale_symbolically() {
        let k = SymExpr::Sym(SymbolicConstant::KernelParam(2));
        let a = tid().scale(SymExpr::Const(2)) + AffineExpr::from_base(SymExpr::Const(8));
        assert_eq!(a.to_string(), "2 * %tid.x + 8");
        assert_eq!(
            (a.clone() - tid().scale(SymExpr::Const(2))).as_const(),
            Some(8)
        );
        let row = AffineExpr::var(Var::Tid(Axis::Y)).scale(k.clone());
        assert_eq!(row.to_string(), "param_2 * %tid.y");
        assert_eq!((row + a).to_string(), "2 * %tid.x + param_2 * %tid.y + 8");
        assert!(tid().scale(SymExpr::Const(0)).as_const() == Some(0));
        assert_eq!(
            AffineExpr::var(Var::Iteration(LoopId(3))).to_string(),
            "k[loop 3]"
        );
        let row = Var::FloorDiv(Box::new(Var::Tid(Axis::X)), 8);
        let col = Var::Mod(Box::new(Var::Tid(Axis::X)), 8);
        let a = AffineExpr::var(row).scale(SymExpr::Const(16))
            + AffineExpr::var(col).scale(SymExpr::Const(2));
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
        let m = |a: AffineExpr, k| a.modulo_const(k, &free).map(|x| x.to_string());
        let d = |a: AffineExpr, k| a.div_const(k, &free).map(|x| x.to_string());
        assert_eq!(
            m(tid().scale(SymExpr::Const(8)), 64).as_deref(),
            Some("8 * (%tid.x mod 8)")
        );
        assert_eq!(
            m(tid() + AffineExpr::from_base(SymExpr::Const(64)), 8).as_deref(),
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
            - tid()
                .modulo_const(8, &free)
                .unwrap()
                .scale(SymExpr::Const(16));
        assert_eq!(neg.to_string(), "16 * %tid.x - 16 * (%tid.x mod 8)");
        assert_eq!(
            (tid().scale(SymExpr::Const(-2)) + AffineExpr::from_base(SymExpr::Const(-4)))
                .to_string(),
            "-2 * %tid.x - 4"
        );
        assert_eq!(
            tid().align_down(8, &free).unwrap().to_string(),
            "8 * ⌊%tid.x/8⌋"
        );
        assert_eq!(
            (tid().scale(SymExpr::Const(2)) + AffineExpr::from_base(SymExpr::Const(64)))
                .align_down(16, &free)
                .unwrap()
                .to_string(),
            "16 * ⌊%tid.x/8⌋ + 64"
        );
    }
}
