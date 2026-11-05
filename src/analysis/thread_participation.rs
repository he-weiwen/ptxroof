//! Which threads of a CTA execute a block. In one trip of the loop
//! holding it (the parent loop for a header, the kernel outside every
//! loop), where a loop nested in it stands as its header, a block runs
//! on the threads at its immediate dominator that take toward it a
//! branch below that dominator it is control dependent on (Ferrante,
//! Ottenstein and Warren, "The Program Dependence Graph and Its Use in
//! Optimization", TOPLAS 1987), or leave toward it a nested loop
//! holding one; when each such predicate is a comparison of affine
//! forms over the thread index alone, the block runs exactly on the
//! threads satisfying them, counted by enumeration over the block
//! shape. With the grid bound, a
//! comparison may hold the CTA index too: each thread is then in at
//! every CTA, at none, or at some, and the counts are bounds where CTAs
//! differ. A comparison with the counter of a loop holding the block,
//! its current trip's, decides a thread only where every trip agrees.
//! A predicate that involves anything else (a parameter, a loaded
//! value) is an unknown selector, and the block's counts stay bounds;
//! one the same for every thread empties or keeps whole warps.

use crate::analysis::affine::affine_analysis::AffineAnalysis;
use crate::analysis::affine::affine_expr::{AffineExpr, Axis, Var};
use crate::analysis::affine::lane_eval::{depends_on_lane, lanes, linear, thread, warps_of};
use crate::analysis::affine::sym_expr::SymExpr;
use crate::analysis::control_flow::dominators::{DominanceInfo, dominators, dominators_from};
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::uniformity_info::Uniform;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{FundamentalType, Instr, Kernel, Module, Opcode, Operand, Register, Stmt};
use crate::ptx::literal::parse_int;
use crate::ptx::modifier::{Cmp, Form, IntInstrType, IntKind, cp_async_size_operand};
use crate::ssa::Value;
use crate::support::intern::Symbol;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};

/// One condition `form cmp 0` over the thread index and, with the grid
/// bound, the CTA index: `cta` is the least and greatest value of the
/// form's `%ctaid` terms over the launch's CTAs, `None` when it has
/// none. `trip` names the loop whose counter, its current trip's, the
/// form holds instead, and the least and greatest value of its term
/// over the loop's trips.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constraint {
    pub form: AffineExpr,
    pub cmp: Cmp,
    pub cta: Option<(i128, i128)>,
    pub trip: Option<(LoopId, (i128, i128))>,
}

/// The condition under which a thread executes an instruction: the
/// selecting branches on its block's dominator chain and its own guard,
/// each a thread-index comparison, a combination of them, or `Unknown`
/// when affine analysis does not read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pred {
    True,
    False,
    Unknown,
    /// A condition that cannot be evaluated but is the same for the
    /// threads `Uniform` names: for every thread of the CTA, a comparison
    /// affine analysis reads of parameters, CTA indices and loop counters
    /// only; or one the uniformity analysis finds uniform. `Atom` names
    /// which.
    Uniform(Uniform, Atom),
    Cmp(Constraint),
    /// `elect.sync`: the lowest lane of each warp among the threads
    /// executing it, which the inner predicate selects.
    Elect(Box<Pred>),
    Not(Box<Pred>),
    And(Box<Pred>, Box<Pred>),
    Or(Box<Pred>, Box<Pred>),
}

/// Which condition a `Pred::Uniform` is, and whether it is negated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub key: AtomKey,
    pub negated: bool,
}

/// What a `Pred::Uniform` reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomKey {
    /// The comparison of `setp` statement `op`, or the first predicate
    /// another statement writes.
    Def(usize),
    /// A predicate value no statement writes first: a φ, an undefined
    /// value, or a second output.
    Value(Value),
    /// `brx.idx` statement `stmt` sends the threads toward `target`.
    Jump { stmt: usize, target: BlockId },
    /// The threads leave the loop headed by `header` toward `target`.
    Leave { header: BlockId, target: BlockId },
    /// A comparison with the counter of a loop that does not hold its use.
    OutOfLoop(Box<Constraint>),
}

/// Whether a thread satisfies a predicate; `Maybe` when an unknown
/// selector decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tri {
    Yes,
    No,
    Maybe,
}

/// How many threads a set holds: those it certainly does and those it
/// possibly does, equal when the set is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counted {
    pub certain: u32,
    pub possible: u32,
}

/// The CTA and the trip of a loop one evaluation is at; a comparison
/// with `%ctaid` or that loop's counter not given one reads the range it
/// holds for it.
#[derive(Debug, Clone, Copy, Default)]
struct Env {
    cta: Option<[i64; 3]>,
    trip: Option<(LoopId, i128)>,
}

/// Which lane an `elect.sync` elected, for one evaluation: the lowest
/// active lane as a representative, a given lane, or none.
#[derive(Debug, Clone, Copy)]
enum Leader {
    Lowest,
    Lane(i64),
    Nobody,
}

impl std::ops::Not for Pred {
    type Output = Pred;

    fn not(self) -> Pred {
        match self {
            Pred::True => Pred::False,
            Pred::False => Pred::True,
            Pred::Unknown => Pred::Unknown,
            Pred::Uniform(u, a) => Pred::Uniform(
                u,
                Atom {
                    negated: !a.negated,
                    ..a
                },
            ),
            Pred::Cmp(c) => Pred::Cmp(Constraint {
                form: c.form,
                cmp: c.cmp.inverse(),
                cta: c.cta,
                trip: c.trip,
            }),
            Pred::Not(p) => *p,
            Pred::And(a, b) => (!*a).or(!*b),
            Pred::Or(a, b) => (!*a).and(!*b),
            p => Pred::Not(Box::new(p)),
        }
    }
}

impl Pred {
    /// Condition `key`, unread but the same for the threads `u` names.
    fn uniform(u: Uniform, key: AtomKey) -> Pred {
        Pred::Uniform(
            u,
            Atom {
                key,
                negated: false,
            },
        )
    }

    pub fn and(self, other: Pred) -> Pred {
        match (self, other) {
            (Pred::False, _) | (_, Pred::False) => Pred::False,
            (Pred::True, p) | (p, Pred::True) => p,
            (Pred::Unknown, Pred::Unknown) => Pred::Unknown,
            (Pred::Uniform(u, a), Pred::Uniform(v, b)) if a == b => Pred::Uniform(u.max(v), a),
            (a, b) => Pred::And(Box::new(a), Box::new(b)),
        }
    }

    pub fn or(self, other: Pred) -> Pred {
        match (self, other) {
            (Pred::True, _) | (_, Pred::True) => Pred::True,
            (Pred::False, p) | (p, Pred::False) => p,
            (Pred::Unknown, Pred::Unknown) => Pred::Unknown,
            (Pred::Uniform(u, a), Pred::Uniform(v, b)) if a == b => Pred::Uniform(u.max(v), a),
            (a, b) => Pred::Or(Box::new(a), Box::new(b)),
        }
    }

    pub fn eval(&self, tid: [i64; 3], shape: [u32; 3]) -> Tri {
        self.eval_as(tid, shape, Leader::Lowest, Env::default())
    }

    fn eval_as(&self, tid: [i64; 3], shape: [u32; 3], leader: Leader, env: Env) -> Tri {
        let eval = |p: &Pred| p.eval_as(tid, shape, leader, env);
        match self {
            Pred::True => Tri::Yes,
            Pred::False => Tri::No,
            Pred::Unknown | Pred::Uniform(..) => Tri::Maybe,
            Pred::Cmp(c) => {
                let trip = env
                    .trip
                    .filter(|&(l, _)| c.trip.is_some_and(|(m, _)| l == m));
                let Ok(trip) = trip
                    .map(|(l, k)| i64::try_from(k).map(|k| (l, k)))
                    .transpose()
                else {
                    return Tri::Maybe;
                };
                let value = |v: &Var| match (v, env.cta, trip) {
                    (Var::Ctaid(a), Some(c), _) => c[*a as usize],
                    (Var::Iteration(l), _, Some((m, k))) if *l == m => k,
                    _ => lanes(tid)(v),
                };
                let Some(x) = at(&c.form, &value) else {
                    return Tri::Maybe;
                };
                let (lo, hi) = match c.cta.filter(|_| env.cta.is_none()) {
                    Some((a, b)) => (x + a, x + b),
                    None => (x, x),
                };
                let (lo, hi) = match c.trip.filter(|_| trip.is_none()) {
                    Some((_, (a, b))) => (lo + a, hi + b),
                    None => (lo, hi),
                };
                decide(c.cmp, lo, hi)
            }
            Pred::Elect(active) => match leader {
                Leader::Lowest => elected(active, tid, shape, env),
                Leader::Lane(k) if linear(tid, shape) == k => Tri::Yes,
                _ => Tri::No,
            },
            Pred::Not(p) => match eval(p) {
                Tri::Yes => Tri::No,
                Tri::No => Tri::Yes,
                Tri::Maybe => Tri::Maybe,
            },
            Pred::And(a, b) => match (eval(a), eval(b)) {
                (Tri::No, _) | (_, Tri::No) => Tri::No,
                (Tri::Yes, Tri::Yes) => Tri::Yes,
                _ => Tri::Maybe,
            },
            Pred::Or(a, b) => match (eval(a), eval(b)) {
                (Tri::Yes, _) | (_, Tri::Yes) => Tri::Yes,
                (Tri::No, Tri::No) => Tri::No,
                _ => Tri::Maybe,
            },
        }
    }

    /// The comparisons `keep` picks, left to right.
    fn constraints(&self, keep: fn(&Constraint) -> bool) -> Vec<&Constraint> {
        let (mut out, mut stack) = (Vec::new(), vec![self]);
        while let Some(node) = stack.pop() {
            match node {
                Pred::Cmp(k) if keep(k) => out.push(k),
                Pred::Elect(p) | Pred::Not(p) => stack.push(p),
                Pred::And(a, b) | Pred::Or(a, b) => stack.extend([&**b, &**a]),
                _ => {}
            }
        }
        out
    }

    /// The comparisons it reads.
    pub fn comparisons(&self) -> Vec<&Constraint> {
        match self {
            Pred::Cmp(c) => vec![c],
            Pred::Not(p) => p.comparisons(),
            Pred::And(a, b) | Pred::Or(a, b) => [a.comparisons(), b.comparisons()].concat(),
            _ => Vec::new(),
        }
    }

    /// The atoms of its uniform unknowns, but those of the threads an
    /// election chose among.
    pub fn atoms(&self) -> Vec<&Atom> {
        match self {
            Pred::Uniform(_, a) => vec![a],
            Pred::Not(p) => p.atoms(),
            Pred::And(a, b) | Pred::Or(a, b) => [a.atoms(), b.atoms()].concat(),
            _ => Vec::new(),
        }
    }

    /// The threads an `elect.sync` in this predicate chose among.
    pub fn elect_active(&self) -> Option<&Pred> {
        match self {
            Pred::Elect(active) => Some(active),
            Pred::Not(p) => p.elect_active(),
            Pred::And(a, b) | Pred::Or(a, b) => a.elect_active().or_else(|| b.elect_active()),
            _ => None,
        }
    }

    /// Every selector is a thread-index condition.
    pub fn exact(&self) -> bool {
        match self {
            Pred::True | Pred::False => true,
            Pred::Cmp(c) => c.cta.is_none() && c.trip.is_none(),
            Pred::Unknown | Pred::Uniform(..) => false,
            Pred::Elect(p) | Pred::Not(p) => p.exact(),
            Pred::And(a, b) | Pred::Or(a, b) => a.exact() && b.exact(),
        }
    }

    /// Which lanes of a warp are in the set is known whenever any is:
    /// unknowns uniform in the CTA or a warp, combined in any way, or a
    /// comparison of the CTA index or a loop
    /// counter alone, only ever empties the set, which a conjunction or
    /// an election keeps and a disjunction or a negation does not.
    pub fn lane_exact(&self) -> bool {
        match self {
            _ if self.uniform_only() => true,
            Pred::True | Pred::False | Pred::Uniform(..) => true,
            Pred::Cmp(c) => {
                (c.cta.is_none() && c.trip.is_none()) || !c.form.terms.keys().any(depends_on_lane)
            }
            Pred::Unknown => false,
            Pred::Elect(p) => p.lane_exact(),
            Pred::Not(p) => p.exact(),
            Pred::And(a, b) => a.lane_exact() && b.lane_exact(),
            Pred::Or(a, b) => a.exact() && b.exact(),
        }
    }

    /// The predicate where it is used in blocks of the loops `inside`
    /// accepts: a comparison with another loop's counter is not its
    /// current trip's there, and selects as an unknown.
    fn in_loops(self, inside: &dyn Fn(LoopId) -> bool) -> Pred {
        let at = |p: Box<Pred>| Box::new(p.in_loops(inside));
        match self {
            Pred::Cmp(c) if c.trip.is_some_and(|(l, _)| !inside(l)) => {
                match c.form.terms.keys().any(depends_on_lane) {
                    true => Pred::Unknown,
                    false => Pred::uniform(Uniform::Cta, AtomKey::OutOfLoop(Box::new(c))),
                }
            }
            Pred::Elect(p) => Pred::Elect(at(p)),
            Pred::Not(p) => Pred::Not(at(p)),
            Pred::And(a, b) => Pred::And(at(a), at(b)),
            Pred::Or(a, b) => Pred::Or(at(a), at(b)),
            p => p,
        }
    }

    /// Only unknowns uniform in the CTA or a warp, under any `and`, `or`
    /// and `not`.
    fn uniform_only(&self) -> bool {
        match self {
            Pred::Uniform(..) => true,
            Pred::Not(p) => p.uniform_only(),
            Pred::And(a, b) | Pred::Or(a, b) => a.uniform_only() && b.uniform_only(),
            _ => false,
        }
    }

    fn has_condition(&self) -> bool {
        match self {
            Pred::True | Pred::Unknown | Pred::Uniform(..) => false,
            Pred::False | Pred::Cmp(_) | Pred::Elect(_) => true,
            Pred::Not(p) => p.has_condition(),
            Pred::And(a, b) | Pred::Or(a, b) => a.has_condition() || b.has_condition(),
        }
    }

    /// The thread-index conditions as text; an unknown selector under
    /// conjunctions alone, `hedged`, is left to the `<=` marker,
    /// elsewhere it prints as `?`, uniform ones combined as one. A
    /// disjunction is parenthesized when nested.
    fn text(&self, nested: bool, hedged: bool, name: &dyn Fn(&Var) -> String) -> Option<String> {
        match self {
            _ if self.uniform_only() => Some("?".to_owned()),
            Pred::True => None,
            Pred::False => Some("no thread".to_owned()),
            Pred::Unknown | Pred::Uniform(..) => Some("?".to_owned()),
            Pred::Cmp(c) => Some(c.render_named(name)),
            Pred::Elect(_) => Some("elected".to_owned()),
            Pred::Not(p) => Some(format!("not ({})", p.text(false, false, name)?)),
            Pred::And(a, b) => {
                let parts: Vec<String> = [a, b]
                    .into_iter()
                    .filter(|p| {
                        !hedged || !(matches!(p.as_ref(), Pred::Unknown) || p.uniform_only())
                    })
                    .filter_map(|p| p.text(true, hedged, name))
                    .collect();
                (!parts.is_empty()).then(|| parts.join(" and "))
            }
            Pred::Or(a, b) => {
                let (a, b) = (a.text(true, false, name)?, b.text(true, false, name)?);
                let text = format!("{a} or {b}");
                Some(if nested { format!("({text})") } else { text })
            }
        }
    }
}

/// Whether `tid` is the lowest lane of its warp that `active` admits, a
/// representative of the elected lane: yes when no lower lane may be
/// active, maybe when the leader is one of several lanes that may be.
fn elected(active: &Pred, tid: [i64; 3], shape: [u32; 3], env: Env) -> Tri {
    let [nx, ny, nz] = shape.map(i64::from);
    let threads = nx * ny * nz;
    if threads == 0 {
        return Tri::Maybe;
    }
    let linear = linear(tid, shape);
    let base = linear - linear % 32;
    let mut candidates = Vec::new();
    for l in base..(base + 32).min(threads) {
        match active.eval_as(thread(l, shape), shape, Leader::Lowest, env) {
            Tri::Yes if candidates.is_empty() => {
                return if l == linear { Tri::Yes } else { Tri::No };
            }
            Tri::Yes => {
                candidates.push(l);
                break;
            }
            Tri::Maybe => candidates.push(l),
            Tri::No => {}
        }
    }
    if candidates.contains(&linear) {
        Tri::Maybe
    } else {
        Tri::No
    }
}

/// `form` with each variable at `value`, in i128 so that a comparison
/// of 64-bit operands reads exactly; `None` past i128.
fn at(form: &AffineExpr, value: &dyn Fn(&Var) -> i64) -> Option<i128> {
    (form.terms.iter()).try_fold(i128::from(form.base.const_part()), |x, (v, c)| {
        let term = i128::from(c.as_const().unwrap_or(0)).checked_mul(i128::from(v.eval(value)))?;
        x.checked_add(term)
    })
}

/// Whether `x cmp 0` holds for every `x` in [lo, hi], for none, or for
/// some: the values that satisfy an `lt`, `le`, `gt` or `ge` and those
/// that do not are each an interval, so the ends decide.
fn decide(cmp: Cmp, lo: i128, hi: i128) -> Tri {
    let zero = lo == 0 && hi == 0;
    let apart = lo > 0 || hi < 0;
    let (all, none) = match cmp {
        Cmp::Eq => (zero, apart),
        Cmp::Ne => (apart, zero),
        _ => (
            holds(cmp, lo) && holds(cmp, hi),
            !holds(cmp, lo) && !holds(cmp, hi),
        ),
    };
    match (all, none) {
        (true, _) => Tri::Yes,
        (_, true) => Tri::No,
        _ => Tri::Maybe,
    }
}

fn holds(cmp: Cmp, x: i128) -> bool {
    match cmp {
        Cmp::Lt => x < 0,
        Cmp::Le => x <= 0,
        Cmp::Gt => x > 0,
        Cmp::Ge => x >= 0,
        Cmp::Eq => x == 0,
        Cmp::Ne => x != 0,
    }
}

/// The axes of the CTA index `v` reads.
fn cta_axes(v: &Var, used: &mut [bool; 3]) {
    match v {
        Var::Ctaid(Axis::X) => used[0] = true,
        Var::Ctaid(Axis::Y) => used[1] = true,
        Var::Ctaid(Axis::Z) => used[2] = true,
        Var::FloorDiv(x, _) | Var::Mod(x, _) => cta_axes(x, used),
        _ => {}
    }
}

/// A CTA index, or a floor or remainder of one.
fn on_cta(v: &Var) -> bool {
    match v {
        Var::Ctaid(_) => true,
        Var::FloorDiv(x, _) | Var::Mod(x, _) => on_cta(x),
        _ => false,
    }
}

/// Constant coefficients and base over the thread and CTA indices and
/// their floors and remainders.
fn on_index(a: &AffineExpr) -> bool {
    a.base.as_const().is_some()
        && (a.terms.iter())
            .all(|(v, c)| c.as_const().is_some() && (depends_on_lane(v) || on_cta(v)))
}

/// `form cmp 0` over the launch, `range` bounding `%tid` by its block
/// and `%ctaid` by its grid: `True` or `False` when the form's range
/// decides it, else the comparison with its `%ctaid` terms substituted
/// when they take one value, or with their range; `None` when
/// unbounded.
fn across_ctas(
    form: &AffineExpr,
    cmp: Cmp,
    range: &dyn Fn(&Var) -> Option<(i64, i64)>,
) -> Option<Pred> {
    let (lo, hi) = form.range(range)?;
    match decide(cmp, lo, hi) {
        Tri::Yes => return Some(Pred::True),
        Tri::No => return Some(Pred::False),
        Tri::Maybe => {}
    }
    let (ctas, lanes): (BTreeMap<_, _>, BTreeMap<_, _>) =
        (form.terms.clone().into_iter()).partition(|(v, _)| on_cta(v));
    let (lo, hi) = AffineExpr {
        terms: ctas,
        base: SymExpr::Const(0),
    }
    .range(range)?;
    let base = i128::from(form.base.as_const()?);
    Some(Pred::Cmp(match lo == hi {
        true => Constraint {
            form: AffineExpr {
                terms: lanes,
                base: SymExpr::Const(i64::try_from(base + lo).ok()?),
            },
            cmp,
            cta: None,
            trip: None,
        },
        false => Constraint {
            form: form.clone(),
            cmp,
            cta: Some((lo, hi)),
            trip: None,
        },
    }))
}

fn on_lane_only(a: &AffineExpr) -> bool {
    a.base.as_const().is_some()
        && a.terms.iter().all(|(v, c)| {
            c.as_const().is_some()
                && matches!(v, Var::Tid(_) | Var::FloorDiv(..) | Var::Mod(..))
                && depends_on_lane(v)
        })
}

/// Per warp of a block, the lanes that may be in a set and those that
/// are, as bit masks.
pub(crate) type LaneMasks = Vec<(u32, u32)>;

/// What two CTAs share when every lane has the same truth at both
/// (`Pred::cta_keys`).
type CtaKey = Vec<(Tri, i128)>;

/// The most CTAs `Pred::launch` enumerates, and the most lanes it
/// evaluates over its groups of CTAs.
pub(crate) const CTA_BUDGET: u64 = 1 << 24;
const LANE_BUDGET: u64 = 1 << 26;

/// The thread index's values on any block, PTX ISA §10.1, §10.2:
/// `%tid.x` and `%tid.y` below 1024, `%tid.z` below 64.
fn any_block(v: &Var) -> Option<(i64, i64)> {
    match v {
        Var::Tid(Axis::Z) => Some((0, 63)),
        Var::Tid(_) => Some((0, 1023)),
        _ => None,
    }
}

/// The CTA index's values on any grid, PTX ISA §10.6, §10.7: `%ctaid.x`
/// below 2^31 − 1, `%ctaid.y` and `%ctaid.z` below 65535.
fn any_grid(v: &Var) -> Option<(i64, i64)> {
    match v {
        Var::Ctaid(Axis::X) => Some((0, (1 << 31) - 2)),
        Var::Ctaid(_) => Some((0, 65534)),
        _ => None,
    }
}

impl Constraint {
    /// The value of the form's `%ctaid` terms at the CTA `c`.
    fn cta_part(&self, c: [i64; 3]) -> i128 {
        let index = |v: &Var| match v {
            Var::Ctaid(Axis::X) => c[0],
            Var::Ctaid(Axis::Y) => c[1],
            Var::Ctaid(Axis::Z) => c[2],
            _ => 0,
        };
        (self.form.terms.iter())
            .filter(|(v, _)| on_cta(v))
            .map(|(v, k)| i128::from(k.as_const().unwrap_or(0)) * i128::from(v.eval(&index)))
            .sum()
    }

    /// `form cmp 0` with the constant moved to the right: `⌊%tid.x/32⌋ < 4`.
    pub fn render(&self) -> String {
        self.render_named(&|v| v.to_string())
    }

    /// The same, naming each variable by `name`.
    pub fn render_named(&self, name: &dyn Fn(&Var) -> String) -> String {
        let c = self.form.base.as_const().unwrap_or(0);
        let lhs = AffineExpr {
            terms: self.form.terms.clone(),
            base: crate::analysis::affine::sym_expr::SymExpr::Const(0),
        };
        let op = match self.cmp {
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
        };
        format!("{} {op} {}", lhs.render(name), -i128::from(c))
    }
}

impl Pred {
    /// Whether the thread at `tid` may be in the set; an unknown
    /// selector includes every thread, as a bound.
    pub fn contains(&self, tid: [i64; 3], shape: [u32; 3]) -> bool {
        self.eval(tid, shape) != Tri::No
    }

    /// A comparison selecting the set reads the CTA index.
    pub fn varies_by_cta(&self) -> bool {
        !self.constraints(|k| k.cta.is_some()).is_empty()
    }

    /// The thread-index conditions as text, `⌊%tid.x/32⌋ < 4 and
    /// %tid.x == 0`; `None` when no such condition selects.
    pub fn render(&self) -> Option<String> {
        self.render_named(&|v| v.to_string())
    }

    /// The same, naming each variable by `name`.
    pub fn render_named(&self, name: &dyn Fn(&Var) -> String) -> Option<String> {
        self.has_condition()
            .then(|| self.text(false, true, name))
            .flatten()
    }

    /// How many threads of a block of this shape are in the set, when a
    /// thread-index condition selects them.
    pub fn count(&self, shape: [u32; 3]) -> Option<Counted> {
        Some(self.tally(shape, Env::default())?.0)
    }

    /// How many warps of a block of this shape have a lane in the set:
    /// PTX ISA §3.1 forms warps from consecutive thread ids.
    pub fn warps(&self, shape: [u32; 3]) -> Option<Counted> {
        Some(self.tally(shape, Env::default())?.1)
    }

    /// Per warp of a block of `shape`, the lanes that may be in the set
    /// and those that are, as bit masks, at CTA `cta` or else at any CTA;
    /// an elected lane is the lowest active one.
    pub(crate) fn lane_masks(&self, shape: [u32; 3], cta: Option<[i64; 3]>) -> LaneMasks {
        let env = Env { cta, trip: None };
        let lane = |(may, must): (u32, u32), t: i64| {
            let bit = 1 << (t % 32);
            match self.eval_as(thread(t, shape), shape, Leader::Lowest, env) {
                Tri::Yes => (may | bit, must | bit),
                Tri::Maybe => (may | bit, must),
                Tri::No => (may, must),
            }
        };
        (warps_of(shape))
            .map(|lanes| lanes.fold((0, 0), lane))
            .collect()
    }

    /// The threads and warps the set holds over a launch of `grid` CTAs
    /// of `shape`, when a thread-index condition selects them and the
    /// count is exact at every CTA. The CTAs are grouped by whether each
    /// comparison whose `%ctaid` terms vary holds on every lane of the
    /// block, on none, or on some at the value those terms take: two
    /// CTAs of a group give every lane the same truth, so one of each
    /// group is counted. None past `CTA_BUDGET` CTAs on the axes those
    /// comparisons read, or `LANE_BUDGET` lane evaluations.
    pub fn launch(&self, shape: [u32; 3], grid: [u32; 3]) -> Option<(u64, u64)> {
        self.weighted_launch(shape, grid, [false; 3], &|_| Some(1))
    }

    /// The same with CTA `c` counted `weight(c)` times, the weight
    /// reading the axes `read`; a CTA where the set holds no thread needs
    /// none.
    pub fn weighted_launch(
        &self,
        shape: [u32; 3],
        grid: [u32; 3],
        read: [bool; 3],
        weight: &dyn Fn([i64; 3]) -> Option<u64>,
    ) -> Option<(u64, u64)> {
        let (ctas, rest) = self.cta_keys(shape, grid, read)?;
        let threads: u64 = shape.iter().map(|&n| u64::from(n)).product();
        let mut groups = HashMap::new();
        for (c, key) in ctas {
            let group = match groups.entry(key) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let env = Env {
                        cta: Some(c),
                        trip: None,
                    };
                    let (t, w) = self.tally(shape, env)?;
                    if t.certain != t.possible || w.certain != w.possible {
                        return None;
                    }
                    e.insert((0u64, t.certain, w.certain))
                }
            };
            if group.1 > 0 {
                group.0 = group.0.checked_add(weight(c)?)?;
            }
            if groups.len() as u64 > LANE_BUDGET / threads {
                return None;
            }
        }
        let (mut sum_t, mut sum_w) = (0u64, 0u64);
        for (k, t, w) in groups.into_values() {
            sum_t = sum_t.checked_add(u64::from(t).checked_mul(k)?)?;
            sum_w = sum_w.checked_add(u64::from(w).checked_mul(k)?)?;
        }
        Some((sum_t.checked_mul(rest)?, sum_w.checked_mul(rest)?))
    }

    /// The CTAs of `grid` grouped by the lanes of a block of `shape` the
    /// set holds at each (`lane_masks`), with the CTAs each group holds;
    /// `None` when no comparison reads the CTA index, past `CTA_BUDGET`
    /// CTAs on the axes those read, or `LANE_BUDGET` lane evaluations.
    pub(crate) fn cta_classes(
        &self,
        shape: [u32; 3],
        grid: [u32; 3],
    ) -> Option<Vec<(LaneMasks, u64)>> {
        if !self.varies_by_cta() {
            return None;
        }
        let (ctas, rest) = self.cta_keys(shape, grid, [false; 3])?;
        let threads: u64 = shape.iter().map(|&n| u64::from(n)).product();
        let (mut keys, mut classes) = (HashMap::new(), Vec::<(LaneMasks, u64)>::new());
        for (c, key) in ctas {
            let i = *keys.entry(key).or_insert_with(|| {
                let masks = self.lane_masks(shape, Some(c));
                (classes.iter().position(|(m, _)| *m == masks)).unwrap_or_else(|| {
                    classes.push((masks, 0));
                    classes.len() - 1
                })
            });
            classes[i].1 += rest;
            if keys.len() as u64 > LANE_BUDGET / threads {
                return None;
            }
        }
        Some(classes)
    }

    /// Every CTA of `grid` on the axes that `read` or the comparisons
    /// whose `%ctaid` terms vary read, with a key two CTAs share when
    /// every lane of a block of `shape` has the same truth at both:
    /// whether each such comparison holds on every lane, on none, or on
    /// some at the value those terms take; and the CTAs of the other
    /// axes. None past `CTA_BUDGET` CTAs on those axes.
    fn cta_keys(
        &self,
        shape: [u32; 3],
        grid: [u32; 3],
        read: [bool; 3],
    ) -> Option<(impl Iterator<Item = ([i64; 3], CtaKey)> + '_, u64)> {
        let over = self.constraints(|k| k.cta.is_some());
        let mut used = read;
        for v in over.iter().flat_map(|k| k.form.terms.keys()) {
            cta_axes(v, &mut used);
        }
        let ctas = |read: bool| {
            (0..3)
                .filter(|&i| used[i] == read)
                .try_fold(1u64, |n, i| n.checked_mul(u64::from(grid[i])))
        };
        let (read, rest) = (ctas(true)?, ctas(false)?);
        if read > CTA_BUDGET {
            return None;
        }
        let threads: Vec<[i64; 3]> = (0..shape.iter().map(|&n| i64::from(n)).product())
            .map(|l| thread(l, shape))
            .collect();
        if threads.is_empty() {
            return None;
        }
        let lanes: Vec<(i128, i128)> = (over.iter())
            .map(|k| {
                (threads.iter().map(|&t| at(&k.form, &lanes(t))))
                    .try_fold((i128::MAX, i128::MIN), |(lo, hi), x| {
                        Some((lo.min(x?), hi.max(x?)))
                    })
            })
            .collect::<Option<_>>()?;
        let n = move |i: usize| if used[i] { i64::from(grid[i]) } else { 1 };
        let all = (0..n(2))
            .flat_map(move |z| (0..n(1)).flat_map(move |y| (0..n(0)).map(move |x| [x, y, z])));
        let keys = all.map(move |c| {
            let key = (over.iter().zip(&lanes))
                .map(|(k, &(lo, hi))| {
                    let u = k.cta_part(c);
                    match decide(k.cmp, lo + u, hi + u) {
                        Tri::Maybe => (Tri::Maybe, u),
                        t => (t, 0),
                    }
                })
                .collect();
            (c, key)
        });
        Some((keys, rest))
    }

    /// The loop whose counter the set's comparisons read, when one does.
    pub fn trip_loop(&self) -> Option<LoopId> {
        let over = self.constraints(|k| k.trip.is_some());
        let l = over.first()?.trip?.0;
        over.iter()
            .all(|k| k.trip.is_some_and(|(t, _)| t == l))
            .then_some(l)
    }

    /// The threads and the warps the set holds summed over trips 1 to
    /// `n` of loop `l`, each as the fewest and the most. Each comparison
    /// `c·k + g cmp 0` with its counter keeps its truth on a lane between
    /// the trips around −g/c, so between those of every lane and
    /// comparison one trip is counted for the run. None where no
    /// thread-index condition selects the set, or past `LANE_BUDGET`
    /// lane evaluations.
    pub fn over_trips(&self, shape: [u32; 3], l: LoopId, n: u64) -> Option<[(u64, u64); 2]> {
        let over = self.constraints(|k| k.trip.is_some());
        let threads: Vec<[i64; 3]> = (0..shape.iter().map(|&n| i64::from(n)).product())
            .map(|t| thread(t, shape))
            .collect();
        let end = i128::from(n) + 1;
        let mut cuts = vec![1, end];
        for k in over.iter().filter(|k| k.trip.is_some_and(|(m, _)| m == l)) {
            let c = i128::from(k.form.terms.get(&Var::Iteration(l))?.as_const()?);
            for &t in &threads {
                let (num, den) = if c < 0 {
                    (at(&k.form, &lanes(t))?, -c)
                } else {
                    (-at(&k.form, &lanes(t))?, c)
                };
                let (lo, hi) = (num.div_euclid(den), -(-num).div_euclid(den));
                cuts.extend([lo, lo + 1, hi, hi + 1].map(|x| x.clamp(1, end)));
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        if (cuts.len() as u64).saturating_mul(threads.len() as u64) > LANE_BUDGET {
            return None;
        }
        let mut sums = [(0u64, 0u64); 2];
        for run in cuts.windows(2) {
            let env = Env {
                cta: None,
                trip: Some((l, run[0])),
            };
            let (t, w) = self.tally(shape, env)?;
            let len = u64::try_from(run[1] - run[0]).ok()?;
            for (sum, c) in sums.iter_mut().zip([t, w]) {
                sum.0 = sum.0.checked_add(u64::from(c.certain).checked_mul(len)?)?;
                sum.1 = sum.1.checked_add(u64::from(c.possible).checked_mul(len)?)?;
            }
        }
        Some(sums)
    }

    /// The threads and the warps with a lane in the set, from the fewest
    /// and the most lanes of each warp in it. An elected lane is
    /// unspecified (PTX ISA §9.7.14.15 promises only that the same leader
    /// is elected every time), so every active lane is tried as the
    /// leader, and no leader when no lane is certainly active, and the
    /// range covers those choices.
    fn tally(&self, shape: [u32; 3], env: Env) -> Option<(Counted, Counted)> {
        if !self.has_condition() || shape.contains(&0) {
            return None;
        }
        let zero = Counted {
            certain: 0,
            possible: 0,
        };
        let (mut threads, mut warps) = (zero, zero);
        for lanes in warps_of(shape) {
            let lanes_as = |leader: Leader| {
                let (mut yes, mut maybe) = (0, 0);
                for l in lanes.clone() {
                    match self.eval_as(thread(l, shape), shape, leader, env) {
                        Tri::Yes => yes += 1,
                        Tri::Maybe => maybe += 1,
                        Tri::No => {}
                    }
                }
                (yes, maybe)
            };
            let (min, max) = match self.elect_active() {
                None => {
                    let (yes, maybe) = lanes_as(Leader::Lowest);
                    (yes, yes + maybe)
                }
                Some(active) => {
                    let state = |l| active.eval_as(thread(l, shape), shape, Leader::Lowest, env);
                    let certain = lanes.clone().any(|l| state(l) == Tri::Yes);
                    (lanes.clone().filter(|&l| state(l) != Tri::No))
                        .map(Leader::Lane)
                        .chain((!certain).then_some(Leader::Nobody))
                        .map(lanes_as)
                        .fold((u32::MAX, 0), |(min, max), (yes, maybe)| {
                            (min.min(yes), max.max(yes + maybe))
                        })
                }
            };
            threads.certain += min;
            threads.possible += max;
            warps.certain += u32::from(min > 0);
            warps.possible += u32::from(max > 0);
        }
        Some((threads, warps))
    }
}

/// Reads predicate registers back to the thread-index conditions that
/// define them, once per definition, and block selections once per
/// block.
struct PredicateReader<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    cfg: &'a ControlFlowGraph,
    forest: &'a LoopForest,
    affine: &'a AffineAnalysis<'a>,
    /// Per scope, the top level and then each loop: post-dominance over
    /// one trip, its edges [`kept`], and the branches each block is
    /// control dependent on there.
    trips: Vec<(DominanceInfo, Vec<Vec<BlockId>>)>,
    memo: HashMap<usize, Pred>,
    blocks: Vec<Option<Pred>>,
    relative: Vec<Pred>,
    deciders: HashMap<(BlockId, AtomKey), usize>,
    comparison_deciders: HashMap<(BlockId, Constraint), usize>,
    forms: HashMap<AtomKey, (AffineExpr, Cmp)>,
}

/// Whether edge `x → t` lies within one trip of scope `s`: not one into
/// an irreducible region, and at the top level any other, else one
/// between blocks of loop `s` other than a back edge.
fn kept(forest: &LoopForest, s: Option<LoopId>, x: BlockId, t: BlockId) -> bool {
    !forest.irreducible_edges.contains(&(x, t))
        && s.is_none_or(|l| {
            let l = forest.get(l);
            t != l.header
                && l.blocks.binary_search(&x).is_ok()
                && l.blocks.binary_search(&t).is_ok()
        })
}

/// The block standing for `x` in one trip of scope `s`: `x`, or the
/// header of the loop directly inside `s` that holds it.
fn level(forest: &LoopForest, s: Option<LoopId>, x: BlockId) -> BlockId {
    let (mut at, mut l) = (x, forest.block_loop[x.0 as usize]);
    while let Some(id) = l.filter(|&id| Some(id) != s) {
        (at, l) = (forest.get(id).header, forest.get(id).parent);
    }
    at
}

/// The scope one trip of which decides whether `b` runs: its innermost
/// loop, or that loop's parent for its header.
fn scope(forest: &LoopForest, b: BlockId) -> Option<LoopId> {
    match forest.block_loop[b.0 as usize] {
        Some(l) if forest.get(l).header == b => forest.get(l).parent,
        other => other,
    }
}

/// The most steps `reach` takes for one block, one per branch it reads:
/// a ladder of rungs that each lead both ways to the next doubles them
/// per rung.
const REACH_BUDGET: u32 = 32;

impl PredicateReader<'_> {
    fn block(&mut self, b: BlockId) -> Pred {
        if let Some(p) = &self.blocks[b.0 as usize] {
            return p.clone();
        }
        let forest = self.forest;
        let pred = match forest.doms.idom[b.0 as usize] {
            Some(d) => {
                let s = scope(forest, b);
                let d = level(forest, s, d);
                let around = |l: LoopId| forest.get(l).blocks.binary_search(&b).is_ok();
                let at = self.block(d).in_loops(&around);
                let mut left = REACH_BUDGET;
                let relative = self.reach(s, d, b, b, &mut left);
                self.relative[b.0 as usize] = relative.clone();
                at.and(relative)
            }
            None if b == ControlFlowGraph::ENTRY => Pred::True,
            None => Pred::False,
        };
        self.blocks[b.0 as usize] = Some(pred.clone());
        pred
    }

    fn trip(&self, s: Option<LoopId>) -> &(DominanceInfo, Vec<Vec<BlockId>>) {
        &self.trips[s.map_or(0, |l| l.0 as usize + 1)]
    }

    /// Which threads at `d` reach `c` in one trip of scope `s`, read for
    /// block `b`: every one where `c` post-dominates `d`, else those
    /// taking toward `c` a branch it is control dependent on that `d`
    /// dominates, or leaving toward it a loop holding such a branch;
    /// unknown where `c` reaches no exit, or past `left` steps.
    fn reach(
        &mut self,
        s: Option<LoopId>,
        d: BlockId,
        c: BlockId,
        b: BlockId,
        left: &mut u32,
    ) -> Pred {
        let Some(rest) = left.checked_sub(1) else {
            return Pred::Unknown;
        };
        *left = rest;
        let (pdoms, deciders) = self.trip(s);
        if pdoms.dominates(c, d) {
            return Pred::True;
        }
        if !pdoms.is_reachable(c) {
            return Pred::Unknown;
        }
        let mut levels: Vec<BlockId> = (deciders[c.0 as usize].iter())
            .map(|&x| level(self.forest, s, x))
            .filter(|&x| x != c)
            .collect();
        levels.sort_unstable();
        levels.dedup();
        let mut pred = Pred::False;
        for x in levels {
            if self.forest.doms.dominates(d, x) {
                let way = match self.forest.block_loop[x.0 as usize] == s {
                    true => self.toward(s, x, c, b),
                    false => self.leaving(x, c),
                };
                pred = pred.or(self.reach(s, d, x, b, left).and(way));
            }
        }
        pred
    }

    /// Which threads at loop header `h` leave its loop toward `c`: every
    /// thread alike in the CTA or a warp where its exits are.
    fn leaving(&self, h: BlockId, c: BlockId) -> Pred {
        let affine = self.affine;
        let key = AtomKey::Leave {
            header: h,
            target: c,
        };
        match self.forest.block_loop[h.0 as usize] {
            Some(l) => (affine.uniformity().exit_uniformity(l))
                .map_or(Pred::Unknown, |u| Pred::uniform(u, key)),
            None => Pred::Unknown,
        }
    }

    /// Which threads at `x` take its branch toward `c` in one trip of
    /// scope `s`, read for block `b`; at a `brx.idx`, all or none of the
    /// threads its index is one value for.
    fn toward(&mut self, s: Option<LoopId>, x: BlockId, c: BlockId, b: BlockId) -> Pred {
        let affine = self.affine;
        let Some((pos, instr)) = affine.data.last_instr(x) else {
            return Pred::Unknown;
        };
        // PTX ISA §9.7.13.4: `brx.idx` continues at the label its index selects.
        if let (Opcode::Brx, None, Some(&index)) = (
            instr.mnemonic,
            instr.predicate,
            self.module.operand_ids(instr.operands).first(),
        ) && let Operand::Register(r) = self.module.operand(index)
        {
            let index = affine.uniformity().register_uniformity(*r, pos);
            let key = AtomKey::Jump {
                stmt: pos,
                target: c,
            };
            let p = index.map_or(Pred::Unknown, |u| Pred::uniform(u, key));
            self.decided(b, &p, pos);
            return p;
        }
        let (&[taken, _], Some(guard)) = (
            &self.cfg.block(x).succs[..],
            instr.predicate.filter(|_| instr.mnemonic == Opcode::Bra),
        ) else {
            return Pred::Unknown;
        };
        let forest = self.forest;
        let inside = |l: LoopId| {
            let blocks = &forest.get(l).blocks;
            blocks.binary_search(&b).is_ok() && blocks.binary_search(&x).is_ok()
        };
        let p = self.of(guard.reg, pos, 0).in_loops(&inside);
        let p = if guard.negated { !p } else { p };
        let p = match kept(forest, s, x, taken) && self.trip(s).0.dominates(c, taken) {
            true => p,
            false => !p,
        };
        self.decided(b, &p, pos);
        p
    }

    /// Branch `pos` decides the atoms and comparisons of `p` for block `b`.
    fn decided(&mut self, b: BlockId, p: &Pred, pos: usize) {
        for a in p.atoms() {
            self.deciders.entry((b, a.key.clone())).or_insert(pos);
        }
        for c in p.comparisons() {
            (self.comparison_deciders)
                .entry((b, c.clone()))
                .or_insert(pos);
        }
    }

    fn of(&mut self, reg: Symbol, pos: usize, depth: u32) -> Pred {
        self.read(reg, pos, depth).0
    }

    /// The predicate `reg` holds at `pos`, uniform when it cannot be read
    /// but the uniformity analysis finds it so, and whether the depth
    /// budget cut the reading short (then it is not memoized).
    fn read(&mut self, reg: Symbol, pos: usize, depth: u32) -> (Pred, bool) {
        match self.read_definition(reg, pos, depth) {
            (Pred::Unknown, cut) => (self.uniform(reg, pos), cut),
            read => read,
        }
    }

    /// The value of `reg` at `pos` as a condition the same for the
    /// threads the uniformity analysis finds it one value for.
    fn uniform(&self, reg: Symbol, pos: usize) -> Pred {
        let Some(v) = self.affine.data.ssa.value_before(pos, Register::Named(reg)) else {
            return Pred::Unknown;
        };
        let second =
            |op: usize| matches!(&self.kernel.stmts[op], Stmt::Instr(i) if self.second(i, reg));
        let key = match v {
            Value::Result { op, .. } if !second(op) => AtomKey::Def(op),
            v => AtomKey::Value(v),
        };
        (self.affine.uniformity().value_uniformity(v, pos))
            .map_or(Pred::Unknown, |u| Pred::uniform(u, key))
    }

    /// Whether `reg` is the second destination of `instr`, `d|p`.
    fn second(&self, instr: &Instr, reg: Symbol) -> bool {
        let ops = self.module.operand_ids(instr.operands);
        ops.first()
            .and_then(|&id| match self.module.operand(id) {
                Operand::MultipleDestinations { children } => {
                    self.module.operand_ids(*children).get(1)
                }
                _ => None,
            })
            .is_some_and(|&id| matches!(self.module.operand(id), Operand::Register(r) if *r == reg))
    }

    fn read_definition(&mut self, reg: Symbol, pos: usize, depth: u32) -> (Pred, bool) {
        let Some(Value::Result { op: def, .. }) =
            self.affine.data.ssa.value_before(pos, Register::Named(reg))
        else {
            return (Pred::Unknown, false);
        };
        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return (Pred::Unknown, false);
        };
        let gates = self.affine.uniformity().gate_uniformity(reg, pos);
        if instr.predicate.is_some() || gates != Ok(Uniform::Cta) {
            return (Pred::Unknown, false);
        }
        if depth >= 8 {
            return (Pred::Unknown, true);
        }
        let (p, cut) = match self.memo.get(&def) {
            Some(p) => (p.clone(), false),
            None => {
                let (p, cut) = self.decode(def, depth + 1);
                if !cut {
                    self.memo.insert(def, p.clone());
                }
                (p, cut)
            }
        };
        // `d|p`: setp's second output is the negated comparison, elect's
        // second output is the predicate and its first the leader's lane.
        let p = match (instr.mnemonic, self.second(instr, reg)) {
            (Opcode::Setp, true) => !p,
            (Opcode::Elect, false) => Pred::Unknown,
            (_, true) if !matches!(p, Pred::Elect(_)) => Pred::Unknown,
            _ => p,
        };
        (p, cut)
    }

    fn decode(&mut self, def: usize, depth: u32) -> (Pred, bool) {
        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return (Pred::Unknown, false);
        };
        let ops = self.module.operand_ids(instr.operands);
        let reg = |id| match self.module.operand(id) {
            Operand::Register(r) => Some(*r),
            _ => None,
        };
        match (instr.mnemonic, Form::of(self.module, instr), ops) {
            (Opcode::Setp, _, _) => {
                let p = condition(self.module, self.kernel, self.affine, def, &mut self.forms);
                (p.unwrap_or(Pred::Unknown), false)
            }
            (Opcode::And | Opcode::Or, Form::Pred, [_, a, b]) => {
                let (Some(a), Some(b)) = (reg(*a), reg(*b)) else {
                    return (Pred::Unknown, false);
                };
                let ((a, ca), (b, cb)) = (self.read(a, def, depth), self.read(b, def, depth));
                let p = if instr.mnemonic == Opcode::And {
                    a.and(b)
                } else {
                    a.or(b)
                };
                (p, ca || cb)
            }
            // PTX ISA §4.5.3: an integer constant as a predicate is False
            // when zero and True otherwise.
            (Opcode::Mov, Form::Pred, [_, a]) => match self.module.operand(*a) {
                Operand::Immediate(t) => match parse_int(self.module.interner.resolve(*t)) {
                    Some(0) => (Pred::False, false),
                    Some(_) => (Pred::True, false),
                    None => (Pred::Unknown, false),
                },
                _ => (Pred::Unknown, false),
            },
            (Opcode::Not, Form::Pred, [_, a]) => match reg(*a) {
                Some(a) => {
                    let (p, cut) = self.read(a, def, depth);
                    (!p, cut)
                }
                None => (Pred::Unknown, false),
            },
            (Opcode::Elect, _, _) => {
                let here = self
                    .cfg
                    .blocks
                    .iter_enumerated()
                    .find(|(_, blk)| blk.start <= def && def < blk.end)
                    .map(|(id, _)| id);
                match here {
                    Some(b) => (Pred::Elect(Box::new(self.block(b))), false),
                    None => (Pred::Unknown, false),
                }
            }
            _ => (Pred::Unknown, false),
        }
    }
}

/// The thread set of every block, and the guard predicate of every
/// guarded instruction other than a branch, by statement index.
#[derive(Debug, Default)]
pub struct ThreadSets {
    pub blocks: Vec<Pred>,
    pub guards: HashMap<usize, Pred>,
    /// Per `cp.async` whose src-size or ignore-src is a register, the
    /// threads that read its source, memory operand 1: those whose
    /// ignore-src is false; with a src-size register, unknown.
    pub sources: HashMap<usize, Pred>,
    /// The edges no thread follows: the other way out of a branch
    /// every thread takes the same way.
    pub dead_edges: Vec<(BlockId, BlockId)>,
    /// Per block, the threads at its immediate dominator that run it.
    pub relative: Vec<Pred>,
    /// Per block and atom of its `relative` set, the branch deciding it.
    pub deciders: HashMap<(BlockId, AtomKey), usize>,
    /// The same per comparison.
    pub comparison_deciders: HashMap<(BlockId, Constraint), usize>,
    /// Per atom of a `setp` comparing no thread index, as `form cmp 0`.
    pub forms: HashMap<AtomKey, (AffineExpr, Cmp)>,
}

impl ThreadSets {
    /// The threads that execute statement `stmt` of block `b`: the
    /// block's set under the statement's guard.
    pub fn of(&self, b: BlockId, stmt: usize) -> Pred {
        let block = &self.blocks[b.0 as usize];
        match self.guards.get(&stmt) {
            Some(guard) => block.clone().and(guard.clone()),
            None => block.clone(),
        }
    }

    /// The guard on memory operand `operand` of statement `stmt`: its
    /// own, and on a `cp.async`'s source that its ignore-src is false.
    pub fn guard(&self, stmt: usize, operand: usize) -> Option<Pred> {
        let own = self.guards.get(&stmt).cloned();
        match (operand, self.sources.get(&stmt)) {
            (1, Some(read)) => Some(own.unwrap_or(Pred::True).and(read.clone())),
            _ => own,
        }
    }

    /// The threads that access memory operand `operand` of statement
    /// `stmt` of block `b`.
    pub fn of_access(&self, b: BlockId, stmt: usize, operand: usize) -> Pred {
        let block = &self.blocks[b.0 as usize];
        match self.guard(stmt, operand) {
            Some(guard) => block.clone().and(guard),
            None => block.clone(),
        }
    }

    /// Whether no thread executes statement `stmt` of block `b`.
    pub fn never(&self, b: BlockId, stmt: usize) -> bool {
        self.blocks[b.0 as usize] == Pred::False || self.guards.get(&stmt) == Some(&Pred::False)
    }
}

pub(crate) fn thread_sets(affine: &AffineAnalysis) -> ThreadSets {
    let KernelAnalyses {
        module,
        kernel,
        cfg,
        forest,
        ..
    } = *affine.data;
    let exit = BlockId(cfg.blocks.len() as u32);
    let trips = std::iter::once(None)
        .chain((0..forest.loops.len() as u32).map(|l| Some(LoopId(l))))
        .map(|s| {
            let reversed = cfg.reversed(&|x, t| kept(forest, s, x, t));
            let pdoms = dominators_from(&reversed, exit);
            let deciders = pdoms.frontiers(&reversed);
            (pdoms, deciders)
        })
        .collect();
    let mut reader = PredicateReader {
        module,
        kernel,
        cfg,
        forest,
        affine,
        trips,
        memo: HashMap::new(),
        blocks: vec![None; cfg.blocks.len()],
        relative: vec![Pred::True; cfg.blocks.len()],
        deciders: HashMap::new(),
        comparison_deciders: HashMap::new(),
        forms: HashMap::new(),
    };
    let blocks: Vec<Pred> = (0..cfg.blocks.len() as u32)
        .map(|b| reader.block(BlockId(b)))
        .collect();
    let mut guards = HashMap::new();
    for (pos, stmt) in kernel.stmts.iter().enumerate() {
        let Stmt::Instr(instr) = stmt else { continue };
        let Some(guard) = instr.predicate else {
            continue;
        };
        if instr.mnemonic == Opcode::Bra {
            continue;
        }
        let b = affine.data.block_of(pos);
        let p = (reader.of(guard.reg, pos, 0))
            .in_loops(&|l| forest.get(l).blocks.binary_search(&b).is_ok());
        guards.insert(pos, if guard.negated { !p } else { p });
    }
    let mut dead_edges = Vec::new();
    for (b, blk) in cfg.blocks.iter_enumerated() {
        if let Some((pos, instr)) = affine.data.last_instr(b)
            && let (Opcode::Bra, Some(guard), &[taken, fallthrough]) =
                (instr.mnemonic, instr.predicate, &blk.succs[..])
            && taken != fallthrough
        {
            let p = reader.of(guard.reg, pos, 0);
            match if guard.negated { !p } else { p } {
                Pred::True => dead_edges.push((b, fallthrough)),
                Pred::False => dead_edges.push((b, taken)),
                _ => {}
            }
        }
    }
    let mut sources = HashMap::new();
    for (pos, stmt) in kernel.stmts.iter().enumerate() {
        let Stmt::Instr(instr) = stmt else { continue };
        let Some(op) = cp_async_size_operand(module, instr) else {
            continue;
        };
        let Operand::Register(r) = module.operand(op) else {
            continue;
        };
        let b = affine.data.block_of(pos);
        let read = match kernel.register_type(module, *r) {
            Some(FundamentalType::Pred) => !(reader.of(*r, pos, 0))
                .in_loops(&|l| forest.get(l).blocks.binary_search(&b).is_ok()),
            _ => Pred::Unknown,
        };
        if read != Pred::True {
            sources.insert(pos, read);
        }
    }
    let doms = dominators(&cfg.without(&dead_edges));
    let mut blocks = blocks;
    for (b, pred) in blocks.iter_mut().enumerate() {
        if !doms.is_reachable(BlockId(b as u32)) {
            *pred = Pred::False;
        }
    }
    ThreadSets {
        blocks,
        guards,
        sources,
        dead_edges,
        relative: reader.relative,
        deciders: reader.deciders,
        comparison_deciders: reader.comparison_deciders,
        forms: reader.forms,
    }
}

/// The `setp` at `def` as `a − b cmp 0` over the thread index, its
/// operands read under affine analysis's bindings (`--bind`, the block
/// shape), when it compares two such forms as integers: an `lt`, `le`,
/// `gt` or `ge`
/// only when both lie in the range of its type, [0, 2^width) or
/// [−2^(width−1), 2^(width−1)), on every thread, an `eq` or `ne` only
/// when `a − b` lies in (−2^width, 2^width), so that it is zero
/// exactly when it is zero modulo 2^width. A form free of the thread
/// index whose comparison the uniformity analysis finds divergent is
/// none, unless it is a loop counter's compared within its type, one
/// value at each trip. Such a comparison free of the thread index and
/// of loop counters that reads as an unknown for the CTA, or as none,
/// is kept in `forms` as `form cmp 0`.
fn condition(
    module: &Module,
    kernel: &Kernel,
    affine: &AffineAnalysis,
    def: usize,
    forms: &mut HashMap<AtomKey, (AffineExpr, Cmp)>,
) -> Option<Pred> {
    let Stmt::Instr(setp) = &kernel.stmts[def] else {
        return None;
    };
    // PTX ISA §9.7.6.2, §9.7.7.2: a `setp` of no integer type compares
    // floats, a packed one per half.
    let Form::IntCompare {
        pred,
        width,
        bool_op: None,
    } = Form::of(module, setp)
    else {
        return None;
    };
    let (cmp, int) = (
        pred.cmp(),
        IntInstrType {
            kind: pred.reads(),
            width,
        },
    );
    let ops = module.operand_ids(setp.operands);
    if ops.len() != 3 {
        return None;
    }
    let bound = |op| {
        let x = affine.operand_at(op, def, 0).ok()?;
        Some(affine.bindings.map_or(x.clone(), |m| x.bind(m)))
    };
    let (a, b) = (bound(ops[1])?, bound(ops[2])?);
    let form = a.clone() - b.clone();
    // %ctaid below the bound grid, PTX ISA §10.6.
    let range = |v: &Var| match v {
        Var::Ctaid(_) | Var::Iteration(_) => affine.var_bounds(v),
        _ => any_block(v),
    };
    // The input contract (PLAN.md): an operand of unknown range does not
    // wrap its type; an unsigned one may still be negative, as in C's
    // `(unsigned)i < n`.
    let fits = |x: &AffineExpr, (min, end): (i128, i128)| {
        (x.range(&range)).is_none_or(|(lo, hi)| min <= lo && hi < end)
    };
    let nonnegative = |x: &AffineExpr| {
        let any = |v: &Var| range(v).or_else(|| any_grid(v));
        (x.range(&any)).is_some_and(|(lo, _)| lo >= 0)
    };
    let comparable = |a: &AffineExpr, b: &AffineExpr| match int.kind {
        IntKind::Bits => {
            let w = 1i128 << int.width;
            fits(&(a.clone() - b.clone()), (1 - w, w))
        }
        IntKind::Signed => fits(a, int.values()) && fits(b, int.values()),
        IntKind::Unsigned => {
            nonnegative(a) && nonnegative(b) && fits(a, int.values()) && fits(b, int.values())
        }
    };
    if let Some(d) = form.as_const()
        && comparable(&a, &b)
    {
        return Some(if holds(cmp, d.into()) {
            Pred::True
        } else {
            Pred::False
        });
    }
    let lanes = form.terms.keys().any(depends_on_lane);
    let trip = per_trip(affine, &form, def).filter(|_| comparable(&a, &b));
    let def_uniform = affine.uniformity().def_uniformity(def);
    let kept = (!lanes && trip.is_none() && comparable(&a, &b)).then(|| (form.clone(), cmp));
    let p = if !lanes && trip.is_none() && def_uniform != Ok(Uniform::Cta) {
        None
    } else if on_lane_only(&form) && comparable(&a, &b) {
        Some(Pred::Cmp(Constraint {
            form,
            cmp,
            cta: None,
            trip: None,
        }))
    } else if let Some(trip) = trip {
        Some(Pred::Cmp(Constraint {
            form,
            cmp,
            cta: None,
            trip: Some(trip),
        }))
    } else if on_index(&form)
        && comparable(&a, &b)
        && let Some(p) = across_ctas(&form, cmp, &|v| {
            affine.var_bounds(v).or_else(|| any_block(v))
        })
    {
        Some(p)
    } else if !lanes {
        Some(Pred::uniform(Uniform::Cta, AtomKey::Def(def)))
    } else {
        None
    };
    if let Some(kept) = kept
        && matches!(p, None | Some(Pred::Uniform(..)))
    {
        forms.insert(AtomKey::Def(def), kept);
    }
    p
}

/// A form of the thread index and one loop's counter, read in that loop
/// so its current trip's, the loop's trip count a constant n: the loop
/// and the range of the counter's term over trips 1 to n.
fn per_trip(
    affine: &AffineAnalysis,
    form: &AffineExpr,
    def: usize,
) -> Option<(LoopId, (i128, i128))> {
    let mut rest = form.clone();
    let mut counters = (form.terms.iter()).filter(|(v, _)| matches!(v, Var::Iteration(_)));
    let (Some((v @ Var::Iteration(l), c)), None) = (counters.next(), counters.next()) else {
        return None;
    };
    rest.terms.remove(v);
    let c = i128::from(c.as_const().filter(|&c| c != 0)?);
    let (_, n) = affine.var_bounds(v)?;
    let inside = affine
        .data
        .forest
        .get(*l)
        .blocks
        .binary_search(&affine.data.block_of(def))
        .is_ok();
    (on_lane_only(&rest) && inside)
        .then_some((*l, (c.min(c * i128::from(n)), c.max(c * i128::from(n)))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::affine::affine_expr::Axis;
    use crate::analysis::affine::sym_expr::SymExpr;
    use crate::analysis::uniformity_info::UniformityInfo;

    fn exact(n: u32) -> Counted {
        Counted {
            certain: n,
            possible: n,
        }
    }

    /// The comparison of statement `def`, the same for every thread.
    fn cta(def: usize) -> Pred {
        Pred::uniform(Uniform::Cta, AtomKey::Def(def))
    }

    fn warps_below_4() -> Constraint {
        let warp = AffineExpr::var(Var::FloorDiv(Box::new(Var::Tid(Axis::X)), 32));
        Constraint {
            form: warp + AffineExpr::constant(-4),
            cmp: Cmp::Lt,
            cta: None,
            trip: None,
        }
    }

    /// A comparison with a loop counter over trips 1 to 512 decides a
    /// thread only where every trip agrees, `k − 513 < 0`, not `k < 511`;
    /// it is never exact, lane-exact without lanes, and outside its loop
    /// an unknown.
    #[test]
    fn a_counter_comparison_is_read_per_trip() {
        let counter = |rest: AffineExpr| {
            Pred::Cmp(Constraint {
                form: AffineExpr::var(Var::Iteration(LoopId(0))) + rest,
                cmp: Cmp::Lt,
                cta: None,
                trip: Some((LoopId(0), (1, 512))),
            })
        };
        let c = AffineExpr::constant;
        let below = counter(c(-511));
        assert!(!below.exact() && below.lane_exact());
        let some = Counted {
            certain: 0,
            possible: 32,
        };
        assert_eq!(below.count([32, 1, 1]), Some(some));
        assert_eq!(counter(c(-513)).count([32, 1, 1]), Some(exact(32)));
        let lanes = counter(AffineExpr::var(Var::Tid(Axis::X)) + c(-511));
        assert!(!lanes.lane_exact());
        let Pred::Cmp(k) = below.clone() else {
            panic!("a comparison");
        };
        assert_eq!(
            below.clone().in_loops(&|_| false),
            Pred::uniform(Uniform::Cta, AtomKey::OutOfLoop(Box::new(k)))
        );
        assert_eq!(below.clone().in_loops(&|_| true), below);
    }

    /// Over a loop's trips, the threads of each trip summed: by runs of
    /// trips between the lanes' breakpoints, as enumerating every trip
    /// gives, for comparisons of the counter alone and with lanes, either
    /// sign of step, every comparison, and two comparisons at once; one
    /// with another loop's counter is read over its range.
    #[test]
    fn a_set_over_trips_sums_each_trip_s_threads() {
        let l = LoopId(0);
        let cmp = |step: i64, lanes: AffineExpr, cmp: Cmp, n: i128| {
            let (a, b) = (i128::from(step), i128::from(step) * n);
            Pred::Cmp(Constraint {
                form: AffineExpr::var(Var::Iteration(l)).scale(SymExpr::Const(step)) + lanes,
                cmp,
                cta: None,
                trip: Some((l, (a.min(b), a.max(b)))),
            })
        };
        let c = AffineExpr::constant;
        let lane = |m: i64| AffineExpr::var(Var::Mod(Box::new(Var::Tid(Axis::X)), m));
        let brute = |set: &Pred, shape: [u32; 3], n: u64| {
            (1..=i128::from(n)).try_fold([(0u64, 0u64); 2], |mut sums, k| {
                let env = Env {
                    cta: None,
                    trip: Some((l, k)),
                };
                let (t, w) = set.tally(shape, env)?;
                for (sum, c) in sums.iter_mut().zip([t, w]) {
                    sum.0 += u64::from(c.certain);
                    sum.1 += u64::from(c.possible);
                }
                Some(sums)
            })
        };
        let shape = [128, 1, 1];
        for n in [1u64, 7, 40] {
            let n128 = i128::from(n);
            for op in [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne] {
                for (step, lanes) in [(1, c(-5)), (64, lane(64) + c(-300)), (-3, lane(32) + c(40))]
                {
                    let set = cmp(step, lanes, op, n128);
                    assert_eq!(
                        set.over_trips(shape, l, n),
                        brute(&set, shape, n),
                        "{op:?} {step} {n}"
                    );
                }
            }
            let both =
                cmp(1, c(-3), Cmp::Ge, n128).and(cmp(64, lane(64) + c(-1000), Cmp::Lt, n128));
            assert_eq!(both.over_trips(shape, l, n), brute(&both, shape, n));
        }
        let k_below_511 = cmp(1, c(-511), Cmp::Lt, 512);
        assert_eq!(
            k_below_511.over_trips([256, 1, 1], l, 512),
            Some([(510 * 256, 510 * 256), (510 * 8, 510 * 8)])
        );
        let elsewhere = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Iteration(LoopId(1))) + c(-3),
            cmp: Cmp::Lt,
            cta: None,
            trip: Some((LoopId(1), (1, 8))),
        });
        assert_eq!(
            elsewhere.over_trips([32, 1, 1], l, 4),
            Some([(0, 128), (0, 4)])
        );
    }

    #[test]
    fn a_constraint_counts_and_renders() {
        let c = warps_below_4();
        assert_eq!(c.render(), "⌊%tid.x/32⌋ < 4");
        let min = Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)) + AffineExpr::constant(i64::MIN),
            ..warps_below_4()
        };
        assert_eq!(min.render(), "%tid.x < 9223372036854775808");
        let set = Pred::Cmp(c);
        assert!(set.exact());
        assert_eq!(set.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(set.count([64, 1, 1]), Some(exact(64)));
        assert_eq!(Pred::True.count([256, 1, 1]), None);
        let lane0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        assert_eq!(lane0.count([128, 2, 1]), Some(exact(2)));
    }

    #[test]
    fn an_unknown_selector_keeps_the_index_conditions_as_a_bound() {
        let set = Pred::Cmp(warps_below_4()).and(Pred::Unknown);
        assert!(!set.exact());
        assert_eq!(set.render().as_deref(), Some("⌊%tid.x/32⌋ < 4"));
        assert_eq!(
            set.count([256, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 128
            })
        );
        assert!(set.contains([0, 0, 0], [256, 1, 1]));
        assert!(!set.contains([200, 0, 0], [256, 1, 1]));
        let unknown = Pred::True.and(Pred::Unknown);
        assert_eq!(unknown, Pred::Unknown);
        assert!(unknown.contains([5, 0, 0], [256, 1, 1]));
        assert_eq!(unknown.render(), None);
        assert_eq!(unknown.count([256, 1, 1]), None);
    }

    #[test]
    fn or_and_not_evaluate_three_valued_and_render() {
        let lane0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        let either = lane0.clone().or(Pred::Cmp(warps_below_4()));
        assert_eq!(either.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(
            either.render().as_deref(),
            Some("%tid.x == 0 or ⌊%tid.x/32⌋ < 4")
        );
        let neither = !either.clone();
        assert!(neither.exact());
        assert_eq!(neither.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(
            neither.render().as_deref(),
            Some("%tid.x != 0 and ⌊%tid.x/32⌋ >= 4")
        );
        assert_eq!((!lane0.clone()).eval([0, 0, 0], [64, 1, 1]), Tri::No);
        assert_eq!(!!lane0.clone(), lane0);
        let hedged = lane0.clone().or(Pred::Unknown);
        assert!(!hedged.exact());
        assert_eq!(hedged.eval([0, 0, 0], [64, 1, 1]), Tri::Yes);
        assert_eq!(hedged.eval([1, 0, 0], [64, 1, 1]), Tri::Maybe);
        assert_eq!(
            hedged.count([64, 1, 1]),
            Some(Counted {
                certain: 1,
                possible: 64
            })
        );
        assert_eq!(hedged.render().as_deref(), Some("%tid.x == 0 or ?"));
        let partly = |p: Pred| p.and(Pred::Unknown);
        assert_eq!(
            (!partly(lane0.clone())).render().as_deref(),
            Some("%tid.x != 0 or ?")
        );
        assert_eq!(
            partly(lane0.clone()).or(!lane0.clone()).render().as_deref(),
            Some("%tid.x == 0 and ? or %tid.x != 0")
        );
        assert_eq!(!Pred::Unknown, Pred::Unknown);
        assert_eq!(!Pred::True, Pred::False);
        assert_eq!((!Pred::True).eval([0, 0, 0], [64, 1, 1]), Tri::No);
        assert_eq!(Pred::False.or(lane0.clone()), lane0);
        assert_eq!(Pred::Unknown.and(Pred::False), Pred::False);
        assert_eq!(Pred::False.count([64, 1, 1]), Some(exact(0)));
        assert_eq!(Pred::False.render().as_deref(), Some("no thread"));
    }

    #[test]
    fn elect_is_one_lane_per_warp_of_the_active_threads() {
        let all = Pred::Elect(Box::new(Pred::True));
        assert_eq!(all.count([128, 1, 1]), Some(exact(4)));
        assert_eq!(all.eval([32, 0, 0], [128, 1, 1]), Tri::Yes);
        assert_eq!(all.eval([33, 0, 0], [128, 1, 1]), Tri::No);
        assert_eq!(all.render().as_deref(), Some("elected"));
        assert_eq!(all.eval([0, 0, 0], [0, 0, 0]), Tri::Maybe);
        let upper = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)) + AffineExpr::constant(-40),
            cmp: Cmp::Ge,
            cta: None,
            trip: None,
        });
        let of_upper = Pred::Elect(Box::new(upper.clone()));
        assert_eq!(of_upper.count([128, 1, 1]), Some(exact(3)));
        let warp0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::FloorDiv(Box::new(Var::Tid(Axis::X)), 32)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        // A warp-uniform condition on the elected lane is exact.
        let leader_of_warp0 = warp0.and(Pred::Elect(Box::new(Pred::True)));
        assert_eq!(leader_of_warp0.count([128, 1, 1]), Some(exact(1)));
        assert_eq!(
            leader_of_warp0.render().as_deref(),
            Some("⌊%tid.x/32⌋ == 0 and elected")
        );
        // A lane-varying one is not: the leader may or may not be lane 0.
        let lane0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        let leader_is_lane0 = lane0.and(Pred::Elect(Box::new(Pred::True)));
        assert_eq!(
            leader_is_lane0.count([128, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 1
            })
        );
        let others = !Pred::Elect(Box::new(Pred::True));
        assert_eq!(others.count([128, 1, 1]), Some(exact(124)));
        let hedged = Pred::Elect(Box::new(upper.or(Pred::Unknown)));
        assert!(!hedged.exact());
        // Warp 0 may have no active lane, so no leader.
        assert_eq!(
            hedged.count([128, 1, 1]),
            Some(Counted {
                certain: 3,
                possible: 4
            })
        );
        let maybe = Pred::Elect(Box::new(cta(0)));
        let range = |certain, possible| Some(Counted { certain, possible });
        assert_eq!(maybe.count([64, 1, 1]), range(0, 2));
        assert_eq!((!maybe).count([64, 1, 1]), range(62, 64));
    }

    /// `a·%tid.x + b·%ctaid.x + base cmp 0`, `%ctaid.x` below `ctas`.
    fn over_ctas(a: i64, b: i64, base: i64, cmp: Cmp, ctas: i64) -> Pred {
        let (lo, hi) = (0, i128::from(b) * i128::from(ctas - 1));
        Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)).scale(SymExpr::Const(a))
                + AffineExpr::var(Var::Ctaid(Axis::X)).scale(SymExpr::Const(b))
                + AffineExpr::constant(base),
            cmp,
            cta: Some((lo.min(hi), lo.max(hi))),
            trip: None,
        })
    }

    #[test]
    fn a_launch_sums_each_cta_s_threads_and_warps() {
        // t < 2c over CTAs 0..4: 0, 2, 4 and 6 threads in 0, 1, 1, 1 warps.
        let below = over_ctas(1, -2, 0, Cmp::Lt, 4);
        assert_eq!(below.launch([32, 1, 1], [4, 1, 1]), Some((12, 3)));
        assert_eq!(
            over_ctas(1, -2, 0, Cmp::Eq, 4).launch([32, 1, 1], [4, 1, 1]),
            Some((4, 4))
        );
        // An axis no comparison reads multiplies.
        assert_eq!(
            below.launch([32, 1, 1], [4, 1 << 25, 1]),
            Some((12 << 25, 3 << 25))
        );
        // A set free of %ctaid: its count at every CTA.
        let lanes = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)) + AffineExpr::constant(-16),
            cmp: Cmp::Lt,
            cta: None,
            trip: None,
        });
        assert_eq!(lanes.launch([32, 1, 1], [3, 2, 1]), Some((96, 6)));
        // No condition, one not decided at a CTA, or no block shape: no count.
        assert_eq!(Pred::True.launch([32, 1, 1], [4, 1, 1]), None);
        assert_eq!(below.launch([0, 0, 0], [4, 1, 1]), None);
        assert_eq!(
            below.clone().and(cta(0)).launch([32, 1, 1], [4, 1, 1]),
            None
        );
        // An election of one lane per warp counts; one among lanes that a
        // comparison then splits does not.
        let first_two = over_ctas(0, 1, -2, Cmp::Lt, 4);
        let elected = Pred::Elect(Box::new(Pred::True));
        assert_eq!(
            first_two.and(elected.clone()).launch([64, 1, 1], [4, 1, 1]),
            Some((4, 4))
        );
        assert_eq!(below.and(elected).launch([32, 1, 1], [4, 1, 1]), None);
        // Past the budgets: 2^24 CTAs on the axis read, or 130,944 groups of
        // a lane-dependent comparison times 1024 lanes.
        let one = over_ctas(1, -1, 0, Cmp::Lt, (1 << 24) + 1);
        assert_eq!(one.launch([32, 1, 1], [(1 << 24) + 1, 1, 1]), None);
        let spread = over_ctas(128, -1, 0, Cmp::Lt, 1 << 17);
        assert_eq!(spread.launch([1024, 1, 1], [1 << 17, 1, 1]), None);
    }

    /// A CTA or a trip keeps a product past i64 that the base brings back:
    /// `2^62·c + t < 2^62 + 1` over 3 CTAs holds 32, 1 and no thread, and
    /// `2^62·k + t − 2^63 + 1 < 0` holds every lane at k = 1 and, as
    /// `t + 1 < 0`, none at k = 2.
    #[test]
    fn a_cta_or_a_trip_sums_past_i64() {
        let wide = over_ctas(1, 1 << 62, -(1 << 62) - 1, Cmp::Lt, 3);
        assert_eq!(wide.launch([32, 1, 1], [3, 1, 1]), Some((33, 2)));
        let l = LoopId(0);
        let tid = || AffineExpr::var(Var::Tid(Axis::X));
        let trip = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Iteration(l)).scale(SymExpr::Const(1 << 62))
                + tid()
                + AffineExpr::constant(i64::MIN + 1),
            cmp: Cmp::Lt,
            cta: None,
            trip: Some((l, (1 << 62, 3 << 62))),
        });
        let at = |k| Env {
            cta: None,
            trip: Some((l, k)),
        };
        assert_eq!(trip.tally([32, 1, 1], at(1)), Some((exact(32), exact(1))));
        assert_eq!(trip.tally([32, 1, 1], at(2)), Some((exact(0), exact(0))));
    }

    /// Weighted, CTA c counts `weight(c)` times: t < 2c over 4 CTAs with
    /// weight c holds 0·0 + 2·1 + 4·2 + 6·3 threads; a weight on the y
    /// axis, which no comparison reads, enumerates it too; a weight that
    /// does not evaluate leaves no count, unless only at CTA 0, where the
    /// set holds no thread, and stops at the first CTA that holds one.
    #[test]
    fn a_weighted_launch_counts_each_cta_its_weight_times() {
        let below = over_ctas(1, -2, 0, Cmp::Lt, 4);
        let by_x = |c: [i64; 3]| u64::try_from(c[0]).ok();
        let launch = |grid, read, weight: &dyn Fn([i64; 3]) -> Option<u64>| {
            below.weighted_launch([32, 1, 1], grid, read, weight)
        };
        assert_eq!(
            launch([4, 1, 1], [true, false, false], &by_x),
            Some((28, 6))
        );
        let by_y = |c: [i64; 3]| u64::try_from(c[1] + 1).ok();
        assert_eq!(
            launch([4, 2, 1], [false, true, false], &by_y),
            Some((36, 9))
        );
        assert_eq!(launch([4, 1, 1], [true, false, false], &|_| None), None);
        let past_zero = |c: [i64; 3]| (c[0] > 0).then_some(1);
        assert_eq!(
            launch([4, 1, 1], [true, false, false], &past_zero),
            Some((12, 3))
        );
        let calls = std::cell::Cell::new(0);
        let fails = |_: [i64; 3]| {
            calls.set(calls.get() + 1);
            None
        };
        assert_eq!(launch([1 << 16, 1, 1], [true, false, false], &fails), None);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_comparison_over_ctas_holds_on_a_lane_at_every_cta_none_or_some() {
        // %tid.x − 2·%ctaid.x with %ctaid.x in 0..=3: 2·%ctaid.x spans 0..=6.
        let at = |cmp: Cmp| {
            Pred::Cmp(Constraint {
                form: AffineExpr::var(Var::Tid(Axis::X))
                    + AffineExpr::var(Var::Ctaid(Axis::X)).scale(SymExpr::Const(-2)),
                cmp,
                cta: Some((-6, 0)),
                trip: None,
            })
        };
        let range = |certain, possible| Some(Counted { certain, possible });
        assert_eq!(at(Cmp::Lt).count([32, 1, 1]), range(0, 6));
        assert_eq!(at(Cmp::Le).count([32, 1, 1]), range(1, 7));
        assert_eq!(at(Cmp::Gt).count([32, 1, 1]), range(25, 31));
        assert_eq!(at(Cmp::Ge).count([32, 1, 1]), range(26, 32));
        assert_eq!(at(Cmp::Eq).count([32, 1, 1]), range(0, 7));
        assert_eq!(at(Cmp::Ne).count([32, 1, 1]), range(25, 32));
        assert_eq!((!at(Cmp::Lt)).eval([3, 0, 0], [32, 1, 1]), Tri::Maybe);
        assert_eq!((!at(Cmp::Lt)).eval([7, 0, 0], [32, 1, 1]), Tri::Yes);
        assert!(!at(Cmp::Lt).exact() && !at(Cmp::Lt).lane_exact());
        // Of the CTA index alone: whole warps, so the lanes stay known.
        let cta = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Ctaid(Axis::X)) + AffineExpr::constant(-2),
            cmp: Cmp::Lt,
            cta: Some((0, 3)),
            trip: None,
        });
        assert_eq!(cta.count([64, 1, 1]), range(0, 64));
        assert!(!cta.exact() && cta.lane_exact());
    }

    #[test]
    fn warps_count_the_groups_of_32_with_a_lane_in_the_set() {
        let warp0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::FloorDiv(Box::new(Var::Tid(Axis::X)), 32)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        let lane0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        assert_eq!(
            Pred::Cmp(warps_below_4()).warps([256, 1, 1]),
            Some(exact(4))
        );
        assert_eq!(lane0.warps([128, 1, 1]), Some(exact(1)));
        assert_eq!((!lane0.clone()).warps([64, 1, 1]), Some(exact(2)));
        assert_eq!((!lane0.clone()).count([64, 1, 1]), Some(exact(63)));
        // 100 threads span four warps, the last one partial.
        let below_100 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)) + AffineExpr::constant(-100),
            cmp: Cmp::Lt,
            cta: None,
            trip: None,
        });
        assert_eq!(below_100.warps([128, 1, 1]), Some(exact(4)));
        // Every warp with an active lane has a leader, whichever lane.
        assert_eq!(
            Pred::Elect(Box::new(Pred::True)).warps([128, 1, 1]),
            Some(exact(4))
        );
        assert_eq!(
            warp0
                .and(Pred::Elect(Box::new(Pred::True)))
                .warps([128, 1, 1]),
            Some(exact(1))
        );
        // Whether warp 0's leader is lane 0 is not known.
        assert_eq!(
            lane0
                .clone()
                .and(Pred::Elect(Box::new(Pred::True)))
                .warps([128, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 1
            })
        );
        // Lane 0 is certainly in; the other lanes of every warp may be.
        let hedged = lane0.or(Pred::Unknown);
        assert_eq!(
            hedged.warps([64, 1, 1]),
            Some(Counted {
                certain: 1,
                possible: 2
            })
        );
        assert_eq!(Pred::True.warps([64, 1, 1]), None);
    }

    #[test]
    fn a_cta_uniform_unknown_is_lane_exact_but_not_exact() {
        let uniform = Pred::Cmp(warps_below_4()).and(cta(0));
        assert!(!uniform.exact() && uniform.lane_exact());
        assert_eq!(uniform.render().as_deref(), Some("⌊%tid.x/32⌋ < 4"));
        assert_eq!(
            uniform.count([256, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 128
            })
        );
        let unknown = Pred::Cmp(warps_below_4()).and(Pred::Unknown);
        assert!(!unknown.exact() && !unknown.lane_exact());
        // Under a disjunction or a negation the uniform unknown decides
        // between two lane sets.
        let lanes = || Pred::Cmp(warps_below_4());
        assert!(!cta(0).or(lanes()).lane_exact());
        assert!(!(!(cta(0).and(lanes()))).lane_exact());
        assert!(Pred::Elect(Box::new(cta(0).and(lanes()))).lane_exact());
        // The negation of a disjunction is the conjunction of its parts'
        // negations: Triton's leader branch `@(%p or ⌊%tid.x/32⌋ ≠ 0) bra`
        // leaves warp 0's lanes, or none.
        let leader = !(cta(0).or(lanes()));
        assert!(matches!(leader, Pred::And(..)) && leader.lane_exact());
    }

    /// Uniform unknowns combined in any way, as comparisons with another
    /// loop's counter are where that loop does not hold them, are the same
    /// for the threads of a warp: lane-exact, and one `?`, which a
    /// conjunction leaves to the `<=` marker.
    #[test]
    fn uniform_unknowns_combined_are_one_unknown() {
        let counter = |cmp| {
            Pred::Cmp(Constraint {
                form: AffineExpr::var(Var::Iteration(LoopId(0))) + AffineExpr::constant(-3),
                cmp,
                cta: None,
                trip: Some((LoopId(0), (1, 8))),
            })
        };
        let either = counter(Cmp::Lt).or(counter(Cmp::Gt)).in_loops(&|_| false);
        let both = counter(Cmp::Lt).and(counter(Cmp::Gt));
        let neither = (!both).in_loops(&|_| false);
        assert!(matches!(either, Pred::Or(..)) && either.lane_exact());
        assert!(matches!(neither, Pred::Or(..)) && neither.lane_exact());
        let below = Pred::Cmp(warps_below_4()).and(either.clone());
        assert!(below.lane_exact());
        assert_eq!(below.render().as_deref(), Some("⌊%tid.x/32⌋ < 4"));
        let lane0 = Pred::Cmp(Constraint {
            form: AffineExpr::var(Var::Tid(Axis::X)),
            cmp: Cmp::Eq,
            cta: None,
            trip: None,
        });
        let hedged = lane0.or(either);
        assert_eq!(hedged.render().as_deref(), Some("%tid.x == 0 or ?"));
    }

    /// A uniform unknown keeps which condition it is: its negation flips
    /// it, and only the same condition at the same polarity merges, at the
    /// larger level; a condition and its negation stay both.
    #[test]
    fn a_uniform_unknown_keeps_its_condition_and_polarity() {
        let (u, v) = (cta(0), cta(1));
        let not_u = !u.clone();
        assert!(matches!(&not_u, Pred::Uniform(Uniform::Cta, a) if a.negated));
        assert_eq!(!not_u.clone(), u);
        assert_eq!(u.clone().and(u.clone()), u);
        assert_eq!(u.clone().or(u.clone()), u);
        let warp = Pred::uniform(Uniform::Warp, AtomKey::Def(0));
        assert_eq!(u.clone().and(warp.clone()), warp);
        let neither = u.clone().and(not_u.clone());
        assert!(matches!(neither, Pred::And(..)));
        assert_eq!(neither.eval([0, 0, 0], [64, 1, 1]), Tri::Maybe);
        assert!(matches!(u.and(v), Pred::And(..)));
    }

    /// The two arms of a branch on a comparison read through its `setp`
    /// hold that comparison and its negation, relative to the block
    /// before them, which the branch decides: one of parameters for the
    /// CTA, one of a loaded value for a warp, each kept as traced.
    #[test]
    fn the_arms_of_a_uniform_branch_hold_its_setp_s_comparison() {
        for (def, level, form) in [
            ("cvt.u32.u64 %r2, %rd1;", Uniform::Cta, "param_0 - 7"),
            ("ld.global.u32 %r2, [%rd1];", Uniform::Warp, "%r2 - 7"),
        ] {
            let (m, sets) = sets_of(&format!(
                "{def}\nsetp.eq.u32 %p1, %r2, 7;\n@%p1 bra $L__Y;\n\
                 add.s32 %r4, %r1, 1;\nbra.uni $L__E;\n$L__Y:\nadd.s32 %r5, %r1, 2;\n$L__E:"
            ));
            let k = &m.kernels[0];
            let setp = (0..k.stmts.len())
                .find(|&s| matches!(&k.stmts[s], Stmt::Instr(i) if i.mnemonic == Opcode::Setp))
                .expect("a setp");
            let taken = Pred::uniform(level, AtomKey::Def(setp));
            let other = !taken.clone();
            assert_eq!((&sets.blocks[1], &sets.blocks[2]), (&other, &taken));
            assert_eq!((&sets.relative[1], &sets.relative[2]), (&other, &taken));
            assert_eq!(sets.relative[3], Pred::True);
            let bra = (setp..k.stmts.len())
                .find(|&s| matches!(&k.stmts[s], Stmt::Instr(i) if i.mnemonic == Opcode::Bra))
                .expect("a branch");
            for b in [1, 2] {
                let decider = sets.deciders.get(&(BlockId(b), AtomKey::Def(setp)));
                assert_eq!(decider, Some(&bra));
            }
            let traced = &sets.forms[&AtomKey::Def(setp)];
            let name = |v: &Var| match v {
                Var::Unknown(_) => "%r2".to_owned(),
                v => v.to_string(),
            };
            assert_eq!(
                (traced.0.render(name), traced.1),
                (form.to_owned(), Cmp::Eq)
            );
        }
    }

    /// Kernel `body`'s module and thread sets, on 64 threads; `%rd1`
    /// holds parameter 0 and `%r1` the thread index.
    fn sets_of(body: &str) -> (Module, ThreadSets) {
        use crate::analysis::control_flow::loop_forest;
        use crate::ptx::cfg::build_cfg;
        use crate::ptx::parse::parser::parse;
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 64, 1, 1\n{{\n\
             .reg .pred %p<8>;\n.reg .b32 %r<8>;\n.reg .b64 %rd<4>;\n\
             ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n{body}\nret;\n}}\n"
        );
        let m = parse(&src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let data = KernelAnalyses::new(&m, k, &cfg, &forest);
        let uniformity = UniformityInfo::new(&data);
        let sets = thread_sets(&AffineAnalysis::new(&data, &uniformity, None));
        (m, sets)
    }

    /// A block runs on the threads its control dependences send to it:
    /// below two branches of the thread index, those either one sends,
    /// `%tid.x < 48 and (%tid.x < 8 or %tid.x >= 40)`; after a loop of
    /// one exit, every thread that entered it, though the latch compares
    /// a parameter.
    #[test]
    fn a_block_runs_on_the_threads_its_control_dependences_send_to_it() {
        let (_, either) = sets_of(
            "setp.ge.u32 %p1, %r1, 48;\n@%p1 bra $L__E;\n\
             setp.lt.u32 %p2, %r1, 8;\n@%p2 bra $L__X;\n\
             setp.lt.u32 %p3, %r1, 40;\n@%p3 bra $L__E;\n\
             $L__X:\nadd.s32 %r4, %r1, 1;\n$L__E:",
        );
        assert_eq!(either.blocks[3].count([64, 1, 1]), Some(exact(16)));
        let (_, after) = sets_of(
            "cvt.u32.u64 %r2, %rd1;\nsetp.ge.u32 %p1, %r1, 32;\n@%p1 bra $L__E;\n\
             mov.u32 %r3, 0;\n$L__H:\nadd.s32 %r3, %r3, 1;\nbra.uni $L__L;\n\
             $L__L:\nsetp.lt.u32 %p2, %r3, %r2;\n@%p2 bra $L__H;\n\
             add.s32 %r4, %r3, 1;\n$L__E:",
        );
        assert!(after.blocks[4].exact());
        assert_eq!(after.blocks[4].count([64, 1, 1]), Some(exact(32)));
    }

    /// A `brx.idx` on a value each warp loads from one address, at the
    /// warp index its lane 0 shuffles to the others, sends all of a warp's
    /// threads to one target: each target runs on whole warps, which ones
    /// unknown. Without the shuffle `⌊%tid.x/32⌋` reads as divergent, and
    /// so do the targets.
    #[test]
    fn a_jump_table_on_a_warp_s_index_selects_whole_warps() {
        let table = |warp: &str| {
            format!(
                "shr.u32 %r2, %r1, 5;\n{warp}\n\
                 mul.wide.u32 %rd2, %r3, 4;\nadd.s64 %rd3, %rd1, %rd2;\nld.global.u32 %r4, [%rd3];\n\
                 $L__T: .branchtargets $L__A, $L__B;\nbrx.idx %r4, $L__T;\n\
                 $L__A:\nadd.s32 %r5, %r1, 1;\nbra.uni $L__E;\n$L__B:\nadd.s32 %r6, %r1, 2;\n$L__E:"
            )
        };
        let (_, shuffled) = sets_of(&table("shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;"));
        let target = |p: &Pred| match p {
            Pred::Uniform(Uniform::Warp, a) if !a.negated => match a.key {
                AtomKey::Jump { target, .. } => Some(target),
                _ => None,
            },
            _ => None,
        };
        assert_eq!(target(&shuffled.blocks[2]), Some(BlockId(2)));
        assert_eq!(target(&shuffled.blocks[3]), Some(BlockId(3)));
        let (_, divided) = sets_of(&table("mov.u32 %r3, %r2;"));
        assert_eq!(divided.blocks[2], Pred::Unknown);
    }

    /// Three-valued logic does not see that an unknown and its negation
    /// cover every thread: the block both arms of an unknown branch reach
    /// by the same test of the thread index counts `<= 16` threads where
    /// 16 run it.
    #[test]
    fn a_test_both_arms_of_an_unknown_branch_repeat_is_a_bound() {
        let (_, sets) = sets_of(
            "ld.global.u32 %r4, [%rd1];\nsetp.ne.u32 %p1, %r4, 0;\n@%p1 bra $L__Y;\n\
             setp.lt.u32 %p2, %r1, 16;\n@%p2 bra $L__C;\nbra.uni $L__E;\n\
             $L__Y:\nsetp.lt.u32 %p3, %r1, 16;\n@%p3 bra $L__C;\nbra.uni $L__E;\n\
             $L__C:\nst.global.u32 [%rd1+4], %r1;\n$L__E:",
        );
        let both = Counted {
            certain: 0,
            possible: 16,
        };
        assert_eq!(sets.blocks[5].count([64, 1, 1]), Some(both));
    }

    /// On a ladder whose rungs each branch both ways to the next on a bit
    /// of the thread index, the steps `reach` takes double per rung: an
    /// early rung is exact, one past `REACH_BUDGET` steps a bound.
    #[test]
    fn a_ladder_past_the_reach_budget_is_an_unknown_selector() {
        let rungs = 12;
        let mut body = String::from("setp.lt.u32 %p2, %r1, 32;\n@%p2 bra $L__Y0;\n");
        for i in 0..rungs {
            let next = |side: &str| match i + 1 == rungs {
                true => "$L__E".to_owned(),
                false => format!("$L__{side}{}", i + 1),
            };
            for side in ["X", "Y"] {
                body += &format!(
                    "$L__{side}{i}:\nand.b32 %r4, %r1, {};\nsetp.ne.u32 %p1, %r4, 0;\n\
                     @%p1 bra {};\nbra.uni {};\n",
                    1 << (i % 6),
                    next("X"),
                    next("Y"),
                );
            }
        }
        body += "$L__E:";
        let (m, sets) = sets_of(&body);
        let cfg = crate::ptx::cfg::build_cfg(&m, &m.kernels[0]);
        let rung = |name: &str| {
            let (b, _) = (cfg.blocks.iter_enumerated())
                .find(|(_, b)| b.label.is_some_and(|l| m.interner.resolve(l) == name))
                .expect("a rung");
            &sets.blocks[b.0 as usize]
        };
        assert!(rung("$L__X1").exact());
        assert!(!rung("$L__X11").exact());
    }

    /// Per `cp.async` of `body`, in order: the threads that read its
    /// source as stored, and its guards on memory operands 0 and 1.
    /// `%r1` holds the thread index.
    fn copies(body: &str) -> Vec<(Option<Pred>, Option<Pred>, Option<Pred>)> {
        let (m, sets) = sets_of(body);
        let k = &m.kernels[0];
        let copy =
            |s: usize| matches!(&k.stmts[s], Stmt::Instr(i) if m.opcode(i).starts_with("cp.async"));
        (0..k.stmts.len())
            .filter(|&s| copy(s))
            .map(|s| {
                (
                    sets.sources.get(&s).cloned(),
                    sets.guard(s, 0),
                    sets.guard(s, 1),
                )
            })
            .collect()
    }

    /// A `cp.async` reads its source where its ignore-src is false (PTX
    /// ISA §9.7.9.26.3.1), read like a guard: the fourth operand, or
    /// under `.L2::cache_hint` the one before the policy. A src-size
    /// register is unknown; a constant false ignore-src stores nothing,
    /// a true one no thread. A guarded copy's source is under both.
    #[test]
    fn a_cp_async_reads_its_source_where_ignore_src_is_false() {
        let text = |p: &Option<Pred>| p.as_ref().and_then(Pred::render);
        let copy = |def: &str, form: &str, tail: &str| {
            format!("{def}\ncp.async.cg.shared.global{form} [%r2], [%rd1], 16{tail};")
        };
        let ignore = "setp.lt.u32 %p1, %r1, 16;\nnot.pred %p2, %p1;";
        let below = Some("%tid.x < 16".to_owned());
        for body in [
            copy(ignore, "", ", %p2"),
            copy(ignore, ".L2::cache_hint", ", %p2, %rd2"),
        ] {
            let [(read, own, source)] = &copies(&body)[..] else {
                panic!("one copy");
            };
            assert_eq!(
                (text(read), own, text(source)),
                (below.clone(), &None, below.clone())
            );
        }
        for body in [
            copy("", "", ""),
            copy("", "", ", 8"),
            copy("", ".L2::cache_hint", ", %rd2"),
            copy("mov.pred %p2, 0;", "", ", %p2"),
        ] {
            assert_eq!(copies(&body), [(None, None, None)], "{body}");
        }
        let unknown = Some(Pred::Unknown);
        let size = copies(&copy("", "", ", %r3"));
        assert_eq!(size, [(unknown.clone(), None, unknown)]);
        let never = Some(Pred::False);
        let all = copies(&copy("mov.pred %p2, -1;", "", ", %p2"));
        assert_eq!(all, [(never.clone(), None, never)]);
        let guarded = format!(
            "{ignore}\nsetp.lt.u32 %p3, %r1, 48;\n\
             @%p3 cp.async.ca.shared.global [%r2], [%rd1], 4, %p2;"
        );
        let [(_, own, source)] = &copies(&guarded)[..] else {
            panic!("one copy");
        };
        assert_eq!(text(own).as_deref(), Some("%tid.x < 48"));
        assert_eq!(text(source).as_deref(), Some("%tid.x < 48 and %tid.x < 16"));
    }

    /// An ignore-src comparing a loop counter reads per trip at a copy in
    /// the loop, as a guard does, and as the same for every thread after
    /// it.
    #[test]
    fn an_ignore_src_on_a_loop_counter_is_read_in_its_loop() {
        let body = "mov.u32 %r5, 0;\n$L__L:\nadd.s32 %r5, %r5, 1;\nsetp.lt.u32 %p6, %r5, 3;\n\
                    cp.async.cg.shared.global [%r2], [%rd1], 16, %p6;\n\
                    setp.lt.u32 %p7, %r5, 4;\n@%p7 bra $L__L;\n\
                    cp.async.cg.shared.global [%r2], [%rd1], 16, %p6;";
        let [(Some(inside), ..), (after, ..)] = &copies(body)[..] else {
            panic!("two copies, the first read");
        };
        assert!(
            matches!(inside, Pred::Cmp(c) if c.trip.is_some()),
            "{inside:?}"
        );
        let out_of_loop = |a: &Atom| a.negated && matches!(a.key, AtomKey::OutOfLoop(_));
        assert!(
            matches!(after, Some(Pred::Uniform(Uniform::Cta, a)) if out_of_loop(a)),
            "{after:?}"
        );
    }
}
