//! SymExpr: the count datatype.
//!
//! Deliberately not a CAS — exactly the forms the trip matcher emits
//! and the report aggregates: integer constants, named symbols (kernel
//! params, opaque trip symbols), sums, products, integer ceil-/floor-
//! division by a constant, modulo by a constant (PTX lowers
//! `K mod 4` as `and.b32 r, K, 3`, so real trip counts are shapes like
//! `(K − K mod 4)/4`), and the min and max of two (Triton's
//! `max(cdiv(K, BK), 2)`).
//!
//! Symbols are assumed nonnegative (they are loop bounds and sizes);
//! `mod` evaluates with `rem_euclid` so a hostile negative binding
//! still yields a value in `[0, c)`.
//!
//! Construction goes through the smart constructors (`add`, `mul`,
//! `ceil_div`, ...), which fold constants and maintain the flattening
//! invariants; `bind` rebuilds through them, so a fully-bound
//! expression collapses to `Const`. Arithmetic that would overflow
//! `i64` stays symbolic (folding is skipped) and `eval` reports `None`
//! rather than wrapping.
//!
//! Printing is deterministic and matches the convention pinned in the
//! committed scenario expectations: `param_2 mod 4` binds tighter than
//! `+`/`-`, exact division prints as `(...) / c`, ceil-division prints
//! `⌈x/c⌉` (a sum or product numerator parenthesized), products
//! parenthesize sums and divisions, a `-c·x` summand prints as
//! `- c * x`, and a min or max as `max(a, b)`.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymExpr {
    Const(i64),
    Sym(String),
    /// n-ary sum; invariant: ≥ 2 terms, no nested `Sum`, constants
    /// folded into one trailing term while they fit in `i64` (an
    /// overflowing constant stays as its own term — never wraps).
    Sum(Vec<SymExpr>),
    /// n-ary product; invariant: ≥ 2 factors, no nested `Prod`,
    /// constants folded into one leading factor while they fit.
    Prod(Vec<SymExpr>),
    /// `⌈e/c⌉`, c > 0.
    CeilDiv(Box<SymExpr>, i64),
    /// `e / c` rounding toward −∞, c > 0. The matcher only emits this
    /// when divisibility holds by construction.
    FloorDiv(Box<SymExpr>, i64),
    /// `e mod c`, c > 0.
    Mod(Box<SymExpr>, i64),
    /// The lesser of two values.
    Min(Box<SymExpr>, Box<SymExpr>),
    /// The greater of two values.
    Max(Box<SymExpr>, Box<SymExpr>),
}

impl SymExpr {
    pub fn sym(name: impl Into<String>) -> SymExpr {
        SymExpr::Sym(name.into())
    }

    /// The constant summand: the value with every symbol at 0.
    pub fn const_part(&self) -> i64 {
        match self {
            SymExpr::Const(c) => *c,
            SymExpr::Sum(ts) => ts.iter().map(SymExpr::const_part).sum(),
            _ => 0,
        }
    }

    /// `(c, rest)` with `self = c·rest`: the leading constant of a
    /// product, or 1.
    pub fn split_const(e: SymExpr) -> (i64, SymExpr) {
        match e {
            SymExpr::Const(c) => (c, SymExpr::Const(1)),
            SymExpr::Prod(fs) => match fs.first() {
                Some(&SymExpr::Const(c)) => {
                    let rest = fs[1..]
                        .iter()
                        .cloned()
                        .reduce(SymExpr::mul)
                        .unwrap_or(SymExpr::Const(1));
                    (c, rest)
                }
                _ => (1, SymExpr::Prod(fs)),
            },
            other => (1, other),
        }
    }

    /// Every value of the expression is a multiple of `m` (m > 0): a
    /// constant, a product with a leading constant, or a sum of such.
    /// Symbols and divisions are not known to be.
    pub fn divisible_by(&self, m: i64) -> bool {
        match self {
            SymExpr::Const(c) => c % m == 0,
            SymExpr::Prod(fs) => matches!(fs.first(), Some(SymExpr::Const(c)) if c % m == 0),
            SymExpr::Sum(ts) => ts.iter().all(|t| t.divisible_by(m)),
            SymExpr::Min(a, b) | SymExpr::Max(a, b) => a.divisible_by(m) && b.divisible_by(m),
            SymExpr::Sym(_) | SymExpr::CeilDiv(..) | SymExpr::FloorDiv(..) | SymExpr::Mod(..) => {
                false
            }
        }
    }

    /// The least value with every symbol at 0 or above, where bounded.
    pub fn least(&self) -> Option<i64> {
        match self {
            SymExpr::Const(c) => Some(*c),
            SymExpr::Sym(_) | SymExpr::Mod(..) => Some(0),
            SymExpr::Sum(ts) => ts.iter().try_fold(0i64, |s, t| s.checked_add(t.least()?)),
            SymExpr::Prod(fs) => {
                (fs.iter()).try_fold(1i64, |p, f| p.checked_mul(f.least().filter(|&l| l >= 0)?))
            }
            SymExpr::CeilDiv(e, c) => Some(-e.least()?.checked_neg()?.div_euclid(*c)),
            SymExpr::FloorDiv(e, c) => e.least().map(|l| l.div_euclid(*c)),
            SymExpr::Min(a, b) => Some(a.least()?.min(b.least()?)),
            SymExpr::Max(a, b) => a.least().max(b.least()),
        }
    }

    /// The least and the greatest value, every symbol within `sym`'s
    /// bounds: `None` where one has none, the whole `i128` line where a
    /// bound leaves it.
    pub fn range(&self, sym: &dyn Fn(&str) -> Option<(i64, i64)>) -> Option<(i128, i128)> {
        match self.span(sym) {
            Ok(r) => Some(r),
            Err(true) => Some((i128::MIN, i128::MAX)),
            Err(false) => None,
        }
    }

    /// The same, `Err(false)` where a symbol has no bounds and
    /// `Err(true)` where a bound leaves `i128`.
    pub fn span(&self, sym: &dyn Fn(&str) -> Option<(i64, i64)>) -> Result<(i128, i128), bool> {
        let pair = |x: &SymExpr, y: &SymExpr| Ok((x.span(sym)?, y.span(sym)?));
        let fits = |x: Option<i128>| x.ok_or(true);
        Ok(match self {
            SymExpr::Const(c) => (i128::from(*c), i128::from(*c)),
            SymExpr::Sym(s) => sym(s)
                .map(|(a, b)| (i128::from(a), i128::from(b)))
                .ok_or(false)?,
            SymExpr::Sum(ts) => ts.iter().try_fold((0i128, 0i128), |(lo, hi), t| {
                let (a, b) = t.span(sym)?;
                Ok((fits(lo.checked_add(a))?, fits(hi.checked_add(b))?))
            })?,
            SymExpr::Prod(fs) => {
                (fs.iter()).try_fold((1, 1), |r, f| product(r, f.span(sym)?).ok_or(true))?
            }
            SymExpr::FloorDiv(e, c) => {
                let (a, b) = e.span(sym)?;
                (a.div_euclid(i128::from(*c)), b.div_euclid(i128::from(*c)))
            }
            SymExpr::CeilDiv(e, c) => {
                let up = |x: i128| fits(x.checked_neg().map(|n| -n.div_euclid(i128::from(*c))));
                let (a, b) = e.span(sym)?;
                (up(a)?, up(b)?)
            }
            SymExpr::Mod(e, c) => {
                let (c, (a, b)) = (i128::from(*c), e.span(sym)?);
                match a.div_euclid(c) == b.div_euclid(c) {
                    true => (a.rem_euclid(c), b.rem_euclid(c)),
                    false => (0, c - 1),
                }
            }
            SymExpr::Min(x, y) => {
                let ((a, b), (c, d)) = pair(x, y)?;
                (a.min(c), b.min(d))
            }
            SymExpr::Max(x, y) => {
                let ((a, b), (c, d)) = pair(x, y)?;
                (a.max(c), b.max(d))
            }
        })
    }

    pub fn as_const(&self) -> Option<i64> {
        match self {
            SymExpr::Const(c) => Some(*c),
            _ => None,
        }
    }

    // -- smart constructors ------------------------------------------------

    // Smart constructor, deliberately an associated function (no
    // self): it canonicalizes rather than implementing arithmetic.
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: SymExpr, b: SymExpr) -> SymExpr {
        let mut terms = Vec::new();
        let mut konst = 0i64;
        let fold = |c: i64, terms: &mut Vec<SymExpr>, konst: &mut i64| {
            match konst.checked_add(c) {
                Some(v) => *konst = v,
                // Refuse to fold: the constant stays a separate term,
                // nothing ever wraps.
                None => terms.push(SymExpr::Const(c)),
            }
        };
        for e in [a, b] {
            match e {
                SymExpr::Const(c) => fold(c, &mut terms, &mut konst),
                SymExpr::Sum(ts) => {
                    for t in ts {
                        match t {
                            SymExpr::Const(c) => fold(c, &mut terms, &mut konst),
                            other => terms.push(other),
                        }
                    }
                }
                other => terms.push(other),
            }
        }
        // Like terms: `2·x + 3·x` is `5·x`; a term whose multiplier sums
        // to zero disappears.
        let mut grouped: Vec<(SymExpr, i64)> = Vec::new();
        let mut kept: Vec<SymExpr> = Vec::new();
        for t in terms.drain(..) {
            let (c, rest) = SymExpr::split_const(t);
            match grouped.iter_mut().find(|(r, _)| *r == rest) {
                Some((_, n)) => match n.checked_add(c) {
                    Some(v) => *n = v,
                    None => kept.push(SymExpr::mul(SymExpr::Const(c), rest)),
                },
                None => grouped.push((rest, c)),
            }
        }
        for (rest, n) in grouped {
            if n != 0 {
                terms.push(SymExpr::mul(SymExpr::Const(n), rest));
            }
        }
        terms.extend(kept);
        if konst != 0 || terms.is_empty() {
            terms.push(SymExpr::Const(konst));
        }
        // Canonical order: positive terms first, negated terms last
        // (stable), so differences print as `a - b`, never `-(b) + a`.
        terms.sort_by_key(|t| match t {
            SymExpr::Const(c) => *c < 0,
            SymExpr::Prod(fs) => matches!(fs.first(), Some(SymExpr::Const(c)) if *c < 0),
            _ => false,
        });
        if terms.len() == 1 {
            terms.pop().expect("len checked")
        } else {
            SymExpr::Sum(terms)
        }
    }

    // Smart constructor, deliberately an associated function (no
    // self): it canonicalizes rather than implementing arithmetic.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(a: SymExpr, b: SymExpr) -> SymExpr {
        SymExpr::add(a, SymExpr::mul(SymExpr::Const(-1), b))
    }

    // Smart constructor, deliberately an associated function (no
    // self): it canonicalizes rather than implementing arithmetic.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: SymExpr, b: SymExpr) -> SymExpr {
        // Distribute a constant over a sum: keeps sums in the canonical
        // "positive terms first" form (so −1·(m − K) becomes K − m, not
        // an opaque negated sum) and is bounded — one factor is scalar.
        match (a, b) {
            (SymExpr::Const(c), SymExpr::Sum(ts)) | (SymExpr::Sum(ts), SymExpr::Const(c)) => ts
                .into_iter()
                .map(|t| SymExpr::mul(SymExpr::Const(c), t))
                .reduce(SymExpr::add)
                .expect("Sum invariant: nonempty"),
            (a2, b2) => Self::mul_flat(a2, b2),
        }
    }

    fn mul_flat(a: SymExpr, b: SymExpr) -> SymExpr {
        let mut factors = Vec::new();
        let mut konst = 1i64;
        let mut overflowed = Vec::new();
        let fold =
            |c: i64, overflowed: &mut Vec<SymExpr>, konst: &mut i64| match konst.checked_mul(c) {
                Some(v) => *konst = v,
                None => overflowed.push(SymExpr::Const(c)),
            };
        for e in [a, b] {
            match e {
                SymExpr::Const(c) => fold(c, &mut overflowed, &mut konst),
                SymExpr::Prod(fs) => {
                    for f in fs {
                        match f {
                            SymExpr::Const(c) => fold(c, &mut overflowed, &mut konst),
                            other => factors.push(other),
                        }
                    }
                }
                other => factors.push(other),
            }
        }
        factors.extend(overflowed);
        if konst == 0 {
            return SymExpr::Const(0);
        }
        if konst != 1 || factors.is_empty() {
            factors.insert(0, SymExpr::Const(konst));
        }
        if factors.len() == 1 {
            factors.pop().expect("len checked")
        } else {
            SymExpr::Prod(factors)
        }
    }

    /// `⌈e/c⌉`; `c` must be positive (matcher-supplied).
    pub fn ceil_div(e: SymExpr, c: i64) -> SymExpr {
        assert!(c > 0, "ceil_div by non-positive constant is a matcher bug");
        if c == 1 {
            return e;
        }
        match e.as_const() {
            Some(v) => SymExpr::Const(v.div_euclid(c) + i64::from(v.rem_euclid(c) != 0)),
            None if e.divisible_by(c) => SymExpr::floor_div(e, c),
            None => SymExpr::CeilDiv(Box::new(e), c),
        }
    }

    /// `e / c` rounding toward −∞; `c` must be positive. Three identities
    /// fold: `⌊(x + c − 1)/c⌋ = ⌈x/c⌉`, `⌊k·c·x / c⌋ = k·x`, and a sum of
    /// multiples of c divided term by term.
    pub fn floor_div(e: SymExpr, c: i64) -> SymExpr {
        assert!(c > 0, "floor_div by non-positive constant is a matcher bug");
        if c == 1 {
            return e;
        }
        match e {
            SymExpr::Const(v) => SymExpr::Const(v.div_euclid(c)),
            SymExpr::Sum(mut terms) if terms.last() == Some(&SymExpr::Const(c - 1)) => {
                terms.pop();
                let rest = terms
                    .into_iter()
                    .reduce(SymExpr::add)
                    .expect("Sum invariant: ≥ 2 terms");
                SymExpr::ceil_div(rest, c)
            }
            SymExpr::Sum(terms) if terms.iter().all(|t| t.divisible_by(c)) => (terms.into_iter())
                .map(|t| SymExpr::floor_div(t, c))
                .reduce(SymExpr::add)
                .expect("Sum invariant: ≥ 2 terms"),
            SymExpr::Prod(factors) if matches!(factors.first(), Some(SymExpr::Const(k)) if k % c == 0) =>
            {
                let mut rest = factors.into_iter();
                let k = rest
                    .next()
                    .and_then(|f| f.as_const())
                    .expect("guard checked");
                rest.fold(SymExpr::Const(k / c), SymExpr::mul)
            }
            other => SymExpr::FloorDiv(Box::new(other), c),
        }
    }

    /// The lesser of `a` and `b`, a constant when both are.
    pub fn min(a: SymExpr, b: SymExpr) -> SymExpr {
        match (a.as_const(), b.as_const()) {
            (Some(x), Some(y)) => SymExpr::Const(x.min(y)),
            _ => SymExpr::Min(Box::new(a), Box::new(b)),
        }
    }

    /// The greater of `a` and `b`, a constant when both are, and one when
    /// it is never less than the other constant.
    pub fn max(a: SymExpr, b: SymExpr) -> SymExpr {
        match (a.as_const(), b.as_const()) {
            (Some(x), Some(y)) => SymExpr::Const(x.max(y)),
            (_, Some(y)) if a.least().is_some_and(|l| l >= y) => a,
            (Some(x), _) if b.least().is_some_and(|l| l >= x) => b,
            _ => SymExpr::Max(Box::new(a), Box::new(b)),
        }
    }

    /// `e mod c`; `c` must be positive.
    pub fn modulo(e: SymExpr, c: i64) -> SymExpr {
        assert!(c > 0, "mod by non-positive constant is a matcher bug");
        if c == 1 {
            return SymExpr::Const(0);
        }
        match e {
            SymExpr::Const(v) => SymExpr::Const(v.rem_euclid(c)),
            // (x mod a) mod c = x mod c when c divides a.
            SymExpr::Mod(inner, a) if a % c == 0 => SymExpr::modulo(*inner, c),
            other => SymExpr::Mod(Box::new(other), c),
        }
    }

    // -- binding / evaluation -------------------------------------------------

    /// Substitute symbols by value and re-simplify. Unbound symbols
    /// survive; a fully-bound expression collapses to `Const`.
    pub fn bind(&self, bindings: &HashMap<String, i64>) -> SymExpr {
        match self {
            SymExpr::Const(c) => SymExpr::Const(*c),
            SymExpr::Sym(name) => match bindings.get(name) {
                Some(&v) => SymExpr::Const(v),
                None => SymExpr::Sym(name.clone()),
            },
            SymExpr::Sum(terms) => terms
                .iter()
                .map(|t| t.bind(bindings))
                .reduce(SymExpr::add)
                .expect("Sum invariant: nonempty"),
            SymExpr::Prod(factors) => factors
                .iter()
                .map(|f| f.bind(bindings))
                .reduce(SymExpr::mul)
                .expect("Prod invariant: nonempty"),
            SymExpr::CeilDiv(e, c) => SymExpr::ceil_div(e.bind(bindings), *c),
            SymExpr::FloorDiv(e, c) => SymExpr::floor_div(e.bind(bindings), *c),
            SymExpr::Mod(e, c) => SymExpr::modulo(e.bind(bindings), *c),
            SymExpr::Min(a, b) => SymExpr::min(a.bind(bindings), b.bind(bindings)),
            SymExpr::Max(a, b) => SymExpr::max(a.bind(bindings), b.bind(bindings)),
        }
    }

    /// `Some(value)` iff fully bound (and nothing overflowed).
    pub fn eval(&self) -> Option<i64> {
        self.bind(&HashMap::new()).as_const()
    }

    /// Symbol names appearing in the expression, sorted, deduplicated.
    pub fn symbols(&self) -> Vec<String> {
        fn walk(e: &SymExpr, out: &mut Vec<String>) {
            match e {
                SymExpr::Const(_) => {}
                SymExpr::Sym(name) => out.push(name.clone()),
                SymExpr::Sum(v) | SymExpr::Prod(v) => {
                    for x in v {
                        walk(x, out);
                    }
                }
                SymExpr::CeilDiv(e, _) | SymExpr::FloorDiv(e, _) | SymExpr::Mod(e, _) => {
                    walk(e, out)
                }
                SymExpr::Min(a, b) | SymExpr::Max(a, b) => {
                    walk(a, out);
                    walk(b, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(self, &mut out);
        out.sort();
        out.dedup();
        out
    }
}

// -- printing -------------------------------------------------------------

/// Where a subexpression is being printed; decides parenthesization.
#[derive(Clone, Copy, PartialEq)]
enum PrintContext {
    Top,
    SumTerm,
    ProdFactor,
    DivOrModLeft,
}

/// The product of two ranges; `None` past `i128`.
pub fn product((a, b): (i128, i128), (c, d): (i128, i128)) -> Option<(i128, i128)> {
    let p = [
        a.checked_mul(c)?,
        a.checked_mul(d)?,
        b.checked_mul(c)?,
        b.checked_mul(d)?,
    ];
    Some((p.into_iter().min()?, p.into_iter().max()?))
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", render(self, PrintContext::Top))
    }
}

fn parens_needed(e: &SymExpr, ctx: PrintContext) -> bool {
    match e {
        SymExpr::Const(_)
        | SymExpr::Sym(_)
        | SymExpr::CeilDiv(..)
        | SymExpr::Min(..)
        | SymExpr::Max(..) => false,
        SymExpr::Sum(_) => matches!(ctx, PrintContext::ProdFactor | PrintContext::DivOrModLeft),
        SymExpr::Prod(_) => ctx == PrintContext::DivOrModLeft,
        SymExpr::FloorDiv(..) => {
            matches!(ctx, PrintContext::ProdFactor | PrintContext::DivOrModLeft)
        }
        SymExpr::Mod(..) => matches!(ctx, PrintContext::ProdFactor | PrintContext::DivOrModLeft),
    }
}

fn render(e: &SymExpr, ctx: PrintContext) -> String {
    let body = match e {
        SymExpr::Const(c) => c.to_string(),
        SymExpr::Sym(name) => name.clone(),
        SymExpr::Sum(terms) => {
            let mut out = String::new();
            for (i, term) in terms.iter().enumerate() {
                // A negative summand prints as subtraction.
                let (neg, mag) = match term {
                    SymExpr::Const(c) if *c < 0 && *c != i64::MIN => (true, SymExpr::Const(-c)),
                    SymExpr::Prod(fs) => match fs.first() {
                        Some(SymExpr::Const(c)) if *c < 0 && *c != i64::MIN => {
                            let mut pos = fs.clone();
                            if *c == -1 {
                                pos.remove(0);
                                match pos.len() {
                                    1 => (true, pos.pop().expect("one factor")),
                                    _ => (true, SymExpr::Prod(pos)),
                                }
                            } else {
                                pos[0] = SymExpr::Const(-c);
                                (true, SymExpr::Prod(pos))
                            }
                        }
                        _ => (false, term.clone()),
                    },
                    _ => (false, term.clone()),
                };
                if i == 0 {
                    if neg {
                        out.push_str("-(");
                        out.push_str(&render(&mag, PrintContext::SumTerm));
                        out.push(')');
                        continue;
                    }
                    out.push_str(&render(&mag, PrintContext::SumTerm));
                    continue;
                }
                out.push_str(if neg { " - " } else { " + " });
                out.push_str(&match mag {
                    SymExpr::Const(m) if neg && m < 0 => format!("({m})"),
                    _ => render(&mag, PrintContext::SumTerm),
                });
            }
            out
        }
        SymExpr::Prod(factors) => match &factors[..] {
            // A bare negation reads better than "-1 * x".
            [SymExpr::Const(-1), x] if !matches!(x, SymExpr::Const(_)) => {
                format!("-{}", render(x, PrintContext::ProdFactor))
            }
            _ => factors
                .iter()
                .map(|f| render(f, PrintContext::ProdFactor))
                .collect::<Vec<_>>()
                .join(" * "),
        },
        SymExpr::CeilDiv(e, c) => format!("⌈{}/{c}⌉", render(e, PrintContext::DivOrModLeft)),
        SymExpr::FloorDiv(e, c) => format!("{} / {c}", render(e, PrintContext::DivOrModLeft)),
        SymExpr::Mod(e, c) => format!("{} mod {c}", render(e, PrintContext::DivOrModLeft)),
        SymExpr::Min(a, b) => format!("min({a}, {b})"),
        SymExpr::Max(a, b) => format!("max({a}, {b})"),
    };
    if parens_needed(e, ctx) {
        format!("({body})")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::SymExpr as E;
    use super::*;

    fn k() -> E {
        E::sym("param_2")
    }

    fn bindings(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
        pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect()
    }

    #[test]
    fn constant_folding_in_constructors() {
        assert_eq!(E::add(E::Const(2), E::Const(3)), E::Const(5));
        assert_eq!(E::mul(E::Const(2), E::Const(3)), E::Const(6));
        assert_eq!(E::add(k(), E::Const(0)), k());
        assert_eq!(E::mul(k(), E::Const(1)), k());
        assert_eq!(E::mul(k(), E::Const(0)), E::Const(0));
        assert_eq!(E::ceil_div(E::Const(9), 8), E::Const(2));
        assert_eq!(E::ceil_div(E::Const(8), 8), E::Const(1));
        assert_eq!(E::floor_div(E::Const(9), 8), E::Const(1));
        assert_eq!(E::modulo(E::Const(9), 8), E::Const(1));
        assert_eq!(E::ceil_div(k(), 1), k());
        assert_eq!(E::floor_div(k(), 1), k());
        assert_eq!(E::modulo(E::modulo(k(), 64), 8), E::modulo(k(), 8));
        assert!(matches!(E::modulo(E::modulo(k(), 8), 64), E::Mod(..)));
        assert_eq!(E::modulo(k(), 1), E::Const(0));
    }

    #[test]
    fn sums_and_products_flatten() {
        let s = E::add(E::add(k(), E::Const(1)), E::add(E::sym("n"), E::Const(2)));
        match &s {
            E::Sum(terms) => {
                assert_eq!(terms.len(), 3);
                assert_eq!(terms.last(), Some(&E::Const(3)));
            }
            other => panic!("expected flattened sum, got {other:?}"),
        }
        let p = E::mul(E::mul(E::Const(2), k()), E::mul(E::Const(4), E::sym("n")));
        match &p {
            E::Prod(fs) => {
                assert_eq!(fs.len(), 3);
                assert_eq!(fs.first(), Some(&E::Const(8)));
            }
            other => panic!("expected flattened product, got {other:?}"),
        }
    }

    #[test]
    fn floor_division_folds_the_ceiling_and_multiple_identities() {
        // LLVM writes ⌈K/64⌉ as (K + 63) >> 6, and a counter stepping by
        // 128 up to 128·x gives (128 * x) / 128.
        assert_eq!(
            E::floor_div(E::add(k(), E::Const(63)), 64),
            E::ceil_div(k(), 64)
        );
        let x = E::ceil_div(k(), 64);
        assert_eq!(E::floor_div(E::mul(E::Const(128), x.clone()), 128), x);
        assert_eq!(
            E::floor_div(E::mul(E::Const(6), k()), 3),
            E::mul(E::Const(2), k())
        );
        assert!(matches!(
            E::floor_div(E::add(k(), E::Const(62)), 64),
            E::FloorDiv(..)
        ));
        assert!(matches!(
            E::floor_div(E::mul(E::Const(6), k()), 4),
            E::FloorDiv(..)
        ));
        // A sum of multiples divides term by term (scaled_mm's
        // `(128·max(⌈K/128⌉, 2) − 128)/128`); a multiple's ceiling is exact.
        let m = E::max(E::ceil_div(k(), 128), E::Const(2));
        let scaled = E::sub(E::mul(E::Const(128), m.clone()), E::Const(128));
        assert_eq!(E::floor_div(scaled, 128), E::sub(m, E::Const(1)));
        assert_eq!(E::ceil_div(E::mul(E::Const(128), k()), 128), k());
        let off = E::add(E::mul(E::Const(128), k()), E::Const(64));
        assert!(matches!(E::floor_div(off.clone(), 128), E::FloorDiv(..)));
        assert!(matches!(E::ceil_div(off, 128), E::CeilDiv(..)));
    }

    #[test]
    fn k2_main_loop_trip_shape_prints_and_binds() {
        // (K - K mod 4) / 4 — the verified k2 shape.
        let trips = E::floor_div(E::sub(k(), E::modulo(k(), 4)), 4);
        assert_eq!(trips.to_string(), "(param_2 - param_2 mod 4) / 4");
        assert_eq!(trips.bind(&bindings(&[("param_2", 4096)])), E::Const(1024));
        assert_eq!(trips.bind(&bindings(&[("param_2", 4099)])), E::Const(1024));
        assert_eq!(trips.bind(&bindings(&[("param_2", 3)])), E::Const(0));
        // Unbound symbol survives binding of others.
        assert_eq!(trips.bind(&bindings(&[("other", 7)])), trips);
        assert_eq!(trips.eval(), None);
    }

    #[test]
    fn k5_outer_trip_shape() {
        let trips = E::ceil_div(k(), 8);
        assert_eq!(trips.to_string(), "⌈param_2/8⌉");
        let shifted = E::ceil_div(E::add(k(), E::Const(1)), 8);
        assert_eq!(shifted.to_string(), "⌈(param_2 + 1)/8⌉");
        assert_eq!(trips.bind(&bindings(&[("param_2", 4096)])), E::Const(512));
        assert_eq!(trips.bind(&bindings(&[("param_2", 4097)])), E::Const(513));
        assert_eq!(trips.bind(&bindings(&[("param_2", 0)])), E::Const(0));
    }

    #[test]
    fn kernel_total_shape_prints_with_subtraction_and_parens() {
        // 8·((K − K mod 4)/4) + 2·(K mod 4) + 3 — a k2-style total.
        let main = E::mul(E::Const(8), E::floor_div(E::sub(k(), E::modulo(k(), 4)), 4));
        let rem = E::mul(E::Const(2), E::modulo(k(), 4));
        let total = E::add(E::add(main, rem), E::Const(3));
        assert_eq!(
            total.to_string(),
            "8 * ((param_2 - param_2 mod 4) / 4) + 2 * (param_2 mod 4) + 3"
        );
        assert_eq!(total.bind(&bindings(&[("param_2", 4096)])), E::Const(8195));
        // 8*1024 + 2*0 + 3
    }

    #[test]
    fn negative_summand_prints_as_subtraction() {
        assert_eq!(E::sub(k(), E::sym("n")).to_string(), "param_2 - n");
        assert_eq!(E::sub(k(), E::Const(5)).to_string(), "param_2 - 5");
        assert_eq!(
            E::sub(k(), E::mul(E::Const(2), E::sym("n"))).to_string(),
            "param_2 - 2 * n"
        );
        // A bare negated product renders without the "-1 *" noise.
        assert_eq!(E::sub(E::Const(0), k()).to_string(), "-param_2");
        // Term ordering is canonical: positives first.
        let lead_neg = E::add(E::mul(E::Const(-1), k()), E::sym("n"));
        assert_eq!(lead_neg.to_string(), "n - param_2");
        // An all-negative sum parenthesizes its leading term.
        let all_neg = E::mul(E::Const(-1), E::add(k(), E::sym("n")));
        assert_eq!(all_neg.to_string(), "-(param_2) - n");
        // ...and a negative leading constant in a sum.
        let neg_const = E::add(E::Const(-5), E::sym("n"));
        assert_eq!(neg_const.to_string(), "n - 5");
    }

    #[test]
    fn mod_by_negative_binding_stays_in_range() {
        let m = E::modulo(k(), 4);
        assert_eq!(m.bind(&bindings(&[("param_2", -1)])), E::Const(3));
    }

    #[test]
    fn printing_is_deterministic_and_stable() {
        // A constant times a sum distributes (canonical form)...
        let e1 = E::mul(E::add(k(), E::Const(1)), E::Const(3));
        assert_eq!(e1.to_string(), "3 * param_2 + 3");
        // ...a symbolic product over a sum does not, and parenthesizes.
        let e2 = E::mul(E::sym("n"), E::add(k(), E::Const(1)));
        assert_eq!(e2.to_string(), "n * (param_2 + 1)");
    }

    #[test]
    fn division_edge_cases() {
        assert_eq!(E::ceil_div(E::Const(0), 8), E::Const(0));
        assert_eq!(E::floor_div(E::Const(0), 8), E::Const(0));
        assert_eq!(E::ceil_div(E::Const(1), 8), E::Const(1));
        // Nested division prints with parens on the left.
        let nested = E::floor_div(E::modulo(k(), 8), 2);
        assert_eq!(nested.to_string(), "(param_2 mod 8) / 2");
        // Division result feeding a product gets parenthesized.
        let p = E::mul(E::Const(2), E::ceil_div(k(), 8));
        assert_eq!(p.to_string(), "2 * ⌈param_2/8⌉");
    }

    #[test]
    fn overflow_refuses_to_wrap() {
        let big = E::add(E::Const(i64::MAX), E::Const(1));
        assert_eq!(big.as_const(), None); // stayed symbolic
        let bigp = E::mul(E::Const(i64::MAX), E::Const(2));
        assert_eq!(bigp.as_const(), None);
    }

    #[test]
    fn symbols_are_collected_sorted_dedup() {
        let e = E::add(
            E::mul(E::sym("b"), E::sym("a")),
            E::ceil_div(E::sym("a"), 4),
        );
        assert_eq!(e.symbols(), ["a", "b"]);
        assert_eq!(E::Const(3).symbols(), Vec::<String>::new());
    }

    /// A product subtracted prints without a unit coefficient.
    #[test]
    fn a_subtracted_product_of_symbols_prints_as_one() {
        let (p, q) = (E::sym("p"), E::sym("q"));
        let e = E::sub(E::sym("x"), E::mul(p.clone(), q.clone()));
        assert_eq!(e.to_string(), "x - p * q");
        assert_eq!(E::sub(E::sym("x"), p).to_string(), "x - p");
    }

    /// Ranges by interval arithmetic, `param_2` in [0, 100], `n` not
    /// bounded; past i128 the whole line, though a remainder follows.
    #[test]
    fn ranges_over_bounded_symbols() {
        let sym = |s: &str| (s == "param_2").then_some((0, 100));
        let row = E::sub(
            E::min(
                E::add(E::mul(E::Const(16), k()), E::Const(16)),
                E::Const(64),
            ),
            k(),
        );
        assert_eq!(row.range(&sym), Some((-84, 64)));
        assert_eq!(E::ceil_div(k(), 8).range(&sym), Some((0, 13)));
        assert_eq!(
            E::floor_div(E::sub(k(), E::Const(1)), 8).range(&sym),
            Some((-1, 12))
        );
        assert_eq!(E::modulo(k(), 64).range(&sym), Some((0, 63)));
        assert_eq!(E::max(k(), E::Const(1)).range(&sym), Some((1, 100)));
        assert_eq!(E::add(k(), E::sym("n")).range(&sym), None);
        let huge = E::mul(E::Const(i64::MAX), E::mul(k(), E::sym("param_2")));
        assert_eq!(
            huge.range(&|_| Some((i64::MIN, i64::MAX))).map(|r| r.1),
            Some(i128::MAX)
        );
    }

    #[test]
    fn least_values_with_nonnegative_symbols_fold_a_max() {
        let minus = |e: E, c: i64| E::sub(e, E::Const(c));
        assert_eq!(E::ceil_div(k(), 8).least(), Some(0));
        assert_eq!(minus(k(), 10).least(), Some(-10));
        assert_eq!(E::ceil_div(minus(k(), 128), 128).least(), Some(-1));
        assert_eq!(E::sub(k(), E::sym("n")).least(), None);
        let w8a8 = minus(E::max(E::ceil_div(k(), 128), E::Const(2)), 1);
        assert_eq!(w8a8.least(), Some(1));
        assert_eq!(E::max(w8a8.clone(), E::Const(1)), w8a8);
        assert_eq!(
            E::max(E::ceil_div(k(), 8), E::Const(0)),
            E::ceil_div(k(), 8)
        );
        assert!(matches!(E::max(minus(k(), 10), E::Const(0)), E::Max(..)));
    }

    #[test]
    fn min_and_max_fold_bind_and_print() {
        // Triton's `tl.maximum(cdiv(K, 128), 2)`, less one.
        let m = E::max(E::ceil_div(k(), 128), E::Const(2));
        assert_eq!(m.to_string(), "max(⌈param_2/128⌉, 2)");
        let trips = E::sub(m.clone(), E::Const(1));
        assert_eq!(trips.to_string(), "max(⌈param_2/128⌉, 2) - 1");
        assert_eq!(trips.bind(&bindings(&[("param_2", 7168)])), E::Const(55));
        assert_eq!(m.bind(&bindings(&[("param_2", 100)])), E::Const(2));
        assert_eq!(E::min(E::Const(3), E::Const(-1)), E::Const(-1));
        assert_eq!(
            E::mul(E::Const(2), E::min(k(), E::sym("n"))).to_string(),
            "2 * min(param_2, n)"
        );
        assert_eq!(E::min(k(), E::sym("n")).symbols(), ["n", "param_2"]);
        assert!(E::max(E::mul(E::Const(4), k()), E::Const(8)).divisible_by(4));
        assert!(!E::max(k(), E::Const(8)).divisible_by(4));
    }

    #[test]
    fn bind_collapses_through_every_node_kind() {
        let e = E::ceil_div(
            E::add(E::mul(E::Const(2), k()), E::modulo(E::floor_div(k(), 2), 8)),
            4,
        );
        // K = 10: 2*10 + (10/2 mod 8) = 20 + 5 = 25; ceildiv(25,4) = 7.
        assert_eq!(e.bind(&bindings(&[("param_2", 10)])), E::Const(7));
    }
}
