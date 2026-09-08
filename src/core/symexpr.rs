//! SymExpr: the count datatype.
//!
//! Deliberately not a CAS — exactly the forms the trip matcher emits
//! and the report aggregates: integer constants, named symbols (kernel
//! params, opaque trip symbols), sums, products, integer ceil-/floor-
//! division by a constant, and modulo by a constant (PTX lowers
//! `K mod 4` as `and.b32 r, K, 3`, so real trip counts are shapes like
//! `(K − K mod 4)/4`).
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
//! parenthesize sums and divisions, and a `-c·x` summand prints as
//! `- c * x`.

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
}

impl SymExpr {
    pub fn sym(name: impl Into<String>) -> SymExpr {
        SymExpr::Sym(name.into())
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
            None => SymExpr::CeilDiv(Box::new(e), c),
        }
    }

    /// `e / c` rounding toward −∞; `c` must be positive.
    pub fn floor_div(e: SymExpr, c: i64) -> SymExpr {
        assert!(c > 0, "floor_div by non-positive constant is a matcher bug");
        if c == 1 {
            return e;
        }
        match e.as_const() {
            Some(v) => SymExpr::Const(v.div_euclid(c)),
            None => SymExpr::FloorDiv(Box::new(e), c),
        }
    }

    /// `e mod c`; `c` must be positive.
    pub fn modulo(e: SymExpr, c: i64) -> SymExpr {
        assert!(c > 0, "mod by non-positive constant is a matcher bug");
        if c == 1 {
            return SymExpr::Const(0);
        }
        match e.as_const() {
            Some(v) => SymExpr::Const(v.rem_euclid(c)),
            None => SymExpr::Mod(Box::new(e), c),
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
enum Ctx {
    Top,
    SumTerm,
    ProdFactor,
    DivOrModLeft,
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", render(self, Ctx::Top))
    }
}

fn parens_needed(e: &SymExpr, ctx: Ctx) -> bool {
    match e {
        SymExpr::Const(_) | SymExpr::Sym(_) | SymExpr::CeilDiv(..) => false,
        SymExpr::Sum(_) => matches!(ctx, Ctx::ProdFactor | Ctx::DivOrModLeft),
        SymExpr::Prod(_) => ctx == Ctx::DivOrModLeft,
        SymExpr::FloorDiv(..) => matches!(ctx, Ctx::ProdFactor | Ctx::DivOrModLeft),
        SymExpr::Mod(..) => matches!(ctx, Ctx::ProdFactor | Ctx::DivOrModLeft),
    }
}

fn render(e: &SymExpr, ctx: Ctx) -> String {
    let body = match e {
        SymExpr::Const(c) => c.to_string(),
        SymExpr::Sym(name) => name.clone(),
        SymExpr::Sum(terms) => {
            let mut out = String::new();
            for (i, term) in terms.iter().enumerate() {
                // A negative summand prints as subtraction.
                let (neg, mag) = match term {
                    SymExpr::Const(c) if *c < 0 => (true, SymExpr::Const(-c)),
                    SymExpr::Prod(fs) => match fs.first() {
                        Some(SymExpr::Const(c)) if *c < 0 => {
                            let mut pos = fs.clone();
                            if *c == -1 && pos.len() == 2 {
                                pos.remove(0);
                                (true, pos.pop().expect("two factors"))
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
                        out.push_str(&render(&mag, Ctx::SumTerm));
                        out.push(')');
                        continue;
                    }
                    out.push_str(&render(&mag, Ctx::SumTerm));
                    continue;
                }
                out.push_str(if neg { " - " } else { " + " });
                out.push_str(&render(&mag, Ctx::SumTerm));
            }
            out
        }
        SymExpr::Prod(factors) => match &factors[..] {
            // A bare negation reads better than "-1 * x".
            [SymExpr::Const(-1), x] => format!("-{}", render(x, Ctx::ProdFactor)),
            _ => factors
                .iter()
                .map(|f| render(f, Ctx::ProdFactor))
                .collect::<Vec<_>>()
                .join(" * "),
        },
        SymExpr::CeilDiv(e, c) => format!("⌈{}/{c}⌉", render(e, Ctx::DivOrModLeft)),
        SymExpr::FloorDiv(e, c) => format!("{} / {c}", render(e, Ctx::DivOrModLeft)),
        SymExpr::Mod(e, c) => format!("{} mod {c}", render(e, Ctx::DivOrModLeft)),
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
