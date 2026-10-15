//! Which threads of a CTA execute a block. A conditional branch on the
//! block's dominator chain fixes its predicate on entry when the block
//! is reachable from only one of the two successors without passing
//! the branch again (its last evaluation decided); when each such
//! predicate is a comparison of affine forms over the thread index
//! alone, the block runs exactly on the threads satisfying all of them,
//! counted by enumeration over the block shape. With the grid bound, a
//! comparison may hold the CTA index too: each thread is then in at
//! every CTA, at none, or at some, and the counts are bounds where CTAs
//! differ. A comparison with the counter of a loop holding the block,
//! its current trip's, decides a thread only where every trip agrees.
//! A predicate that involves anything else (a parameter, a loaded
//! value) is an unknown selector, and the block's counts stay bounds;
//! one the same for every thread empties or keeps whole warps. So is a branch off the chain that decides whether
//! the block runs: a path from a chain branch that reaches the block and
//! also leaves its scope around it.

use crate::analysis::control_flow::dominators;
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::instruction_counts::classify::cp_async_size_operand;
use crate::analysis::scalar::affine::{Affine, Axis, Var};
use crate::analysis::scalar::lane_eval::{depends_on_lane, eval};
use crate::analysis::scalar::symexpr::SymExpr;
use crate::analysis::scalar::trace::AffineValueTracer;
use crate::analysis::uniformity::Uniformity;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{FundamentalType, Kernel, Module, Opcode, Operand, Register, Stmt};
use crate::ptx::literal::parse_int;
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub form: Affine,
    pub cmp: String,
    pub cta: Option<(i128, i128)>,
    pub trip: Option<(LoopId, (i128, i128))>,
}

/// The condition under which a thread executes an instruction: the
/// selecting branches on its block's dominator chain and its own guard,
/// each a thread-index comparison, a combination of them, or `Unknown`
/// when the tracer does not read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pred {
    True,
    False,
    Unknown,
    /// A condition that cannot be evaluated but is the same for every
    /// thread of the CTA: a comparison the tracer reads of parameters,
    /// CTA indices and loop counters only, or one the uniformity
    /// analysis finds uniform.
    Uniform,
    Cmp(Constraint),
    /// `elect.sync`: the lowest lane of each warp among the threads
    /// executing it, which the inner predicate selects.
    Elect(Box<Pred>),
    Not(Box<Pred>),
    And(Box<Pred>, Box<Pred>),
    Or(Box<Pred>, Box<Pred>),
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
            Pred::Uniform => Pred::Uniform,
            Pred::Cmp(c) => Pred::Cmp(Constraint {
                form: c.form,
                cmp: negate(&c.cmp).to_owned(),
                cta: c.cta,
                trip: c.trip,
            }),
            Pred::Not(p) => *p,
            p => Pred::Not(Box::new(p)),
        }
    }
}

impl Pred {
    pub fn and(self, other: Pred) -> Pred {
        match (self, other) {
            (Pred::False, _) | (_, Pred::False) => Pred::False,
            (Pred::True, p) | (p, Pred::True) => p,
            (Pred::Unknown, Pred::Unknown) => Pred::Unknown,
            (Pred::Uniform, Pred::Uniform) => Pred::Uniform,
            (a, b) => Pred::And(Box::new(a), Box::new(b)),
        }
    }

    pub fn or(self, other: Pred) -> Pred {
        match (self, other) {
            (Pred::True, _) | (_, Pred::True) => Pred::True,
            (Pred::False, p) | (p, Pred::False) => p,
            (Pred::Unknown, Pred::Unknown) => Pred::Unknown,
            (Pred::Uniform, Pred::Uniform) => Pred::Uniform,
            (a, b) => Pred::Or(Box::new(a), Box::new(b)),
        }
    }

    pub fn eval(&self, tid: [i64; 3], shape: [u32; 3]) -> Tri {
        self.eval_as(tid, shape, Leader::Lowest)
    }

    fn eval_as(&self, tid: [i64; 3], shape: [u32; 3], leader: Leader) -> Tri {
        match self {
            Pred::True => Tri::Yes,
            Pred::False => Tri::No,
            Pred::Unknown | Pred::Uniform => Tri::Maybe,
            Pred::Cmp(c) => {
                let x = at(&c.form, tid);
                let (lo, hi) = c.cta.map_or((x, x), |(a, b)| (x + a, x + b));
                let (lo, hi) = c.trip.map_or((lo, hi), |(_, (a, b))| (lo + a, hi + b));
                decide(&c.cmp, lo, hi)
            }
            Pred::Elect(active) => match leader {
                Leader::Lowest => elected(active, tid, shape),
                Leader::Lane(k) if linear(tid, shape) == k => Tri::Yes,
                _ => Tri::No,
            },
            Pred::Not(p) => match p.eval_as(tid, shape, leader) {
                Tri::Yes => Tri::No,
                Tri::No => Tri::Yes,
                Tri::Maybe => Tri::Maybe,
            },
            Pred::And(a, b) => match (a.eval_as(tid, shape, leader), b.eval_as(tid, shape, leader))
            {
                (Tri::No, _) | (_, Tri::No) => Tri::No,
                (Tri::Yes, Tri::Yes) => Tri::Yes,
                _ => Tri::Maybe,
            },
            Pred::Or(a, b) => {
                match (a.eval_as(tid, shape, leader), b.eval_as(tid, shape, leader)) {
                    (Tri::Yes, _) | (_, Tri::Yes) => Tri::Yes,
                    (Tri::No, Tri::No) => Tri::No,
                    _ => Tri::Maybe,
                }
            }
        }
    }

    /// The predicate at the CTA `c`: each comparison's `%ctaid` terms
    /// at their value there.
    fn at_cta(&self, c: [i64; 3]) -> Pred {
        let at = |p: &Pred| Box::new(p.at_cta(c));
        match self {
            Pred::Cmp(k) if k.cta.is_some() => {
                let base = i128::from(k.form.base.const_part()) + k.cta_part(c);
                let Ok(base) = i64::try_from(base) else {
                    return Pred::Unknown;
                };
                Pred::Cmp(Constraint {
                    form: Affine {
                        terms: (k.form.terms.iter())
                            .filter(|(v, _)| !on_cta(v))
                            .map(|(v, c)| (v.clone(), c.clone()))
                            .collect(),
                        base: SymExpr::Const(base),
                    },
                    cmp: k.cmp.clone(),
                    cta: None,
                    trip: None,
                })
            }
            Pred::Elect(p) => Pred::Elect(at(p)),
            Pred::Not(p) => Pred::Not(at(p)),
            Pred::And(a, b) => Pred::And(at(a), at(b)),
            Pred::Or(a, b) => Pred::Or(at(a), at(b)),
            p => p.clone(),
        }
    }

    /// The predicate at trip `k` of loop `l`: each comparison with its
    /// counter at the counter's value there.
    fn at_trip(&self, l: LoopId, k: i128) -> Option<Pred> {
        let at = |p: &Pred| p.at_trip(l, k).map(Box::new);
        Some(match self {
            Pred::Cmp(c) if c.trip.is_some_and(|(t, _)| t == l) => {
                let mut form = c.form.clone();
                let step = i128::from(form.terms.remove(&Var::Iter(l))?.as_const()?);
                let base = i128::from(form.base.as_const()?).checked_add(step.checked_mul(k)?)?;
                form.base = SymExpr::Const(i64::try_from(base).ok()?);
                Pred::Cmp(Constraint {
                    form,
                    trip: None,
                    ..c.clone()
                })
            }
            Pred::Elect(p) => Pred::Elect(at(p)?),
            Pred::Not(p) => Pred::Not(at(p)?),
            Pred::And(a, b) => Pred::And(at(a)?, at(b)?),
            Pred::Or(a, b) => Pred::Or(at(a)?, at(b)?),
            p => p.clone(),
        })
    }

    /// The comparisons with loop counters.
    fn over_trips<'a>(&'a self, out: &mut Vec<&'a Constraint>) {
        match self {
            Pred::Cmp(k) if k.trip.is_some() => out.push(k),
            Pred::Elect(p) | Pred::Not(p) => p.over_trips(out),
            Pred::And(a, b) | Pred::Or(a, b) => {
                a.over_trips(out);
                b.over_trips(out);
            }
            _ => {}
        }
    }

    /// The comparisons whose `%ctaid` terms vary over the grid.
    fn over_ctas<'a>(&'a self, out: &mut Vec<&'a Constraint>) {
        match self {
            Pred::Cmp(k) if k.cta.is_some() => out.push(k),
            Pred::Elect(p) | Pred::Not(p) => p.over_ctas(out),
            Pred::And(a, b) | Pred::Or(a, b) => {
                a.over_ctas(out);
                b.over_ctas(out);
            }
            _ => {}
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

    pub fn exact(&self) -> bool {
        match self {
            Pred::True | Pred::False => true,
            Pred::Cmp(c) => c.cta.is_none() && c.trip.is_none(),
            Pred::Unknown | Pred::Uniform => false,
            Pred::Elect(p) | Pred::Not(p) => p.exact(),
            Pred::And(a, b) | Pred::Or(a, b) => a.exact() && b.exact(),
        }
    }

    /// Which lanes of a warp are in the set is known whenever any is:
    /// a uniform unknown, or a comparison of the CTA index or a loop
    /// counter alone, only ever empties the set, which a conjunction or
    /// an election keeps and a disjunction or a negation does not.
    pub fn lane_exact(&self) -> bool {
        match self {
            Pred::True | Pred::False | Pred::Uniform => true,
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
                    false => Pred::Uniform,
                }
            }
            Pred::Elect(p) => Pred::Elect(at(p)),
            Pred::Not(p) => Pred::Not(at(p)),
            Pred::And(a, b) => Pred::And(at(a), at(b)),
            Pred::Or(a, b) => Pred::Or(at(a), at(b)),
            p => p,
        }
    }

    fn has_condition(&self) -> bool {
        match self {
            Pred::True | Pred::Unknown | Pred::Uniform => false,
            Pred::False | Pred::Cmp(_) | Pred::Elect(_) => true,
            Pred::Not(p) => p.has_condition(),
            Pred::And(a, b) | Pred::Or(a, b) => a.has_condition() || b.has_condition(),
        }
    }

    /// The thread-index conditions as text; an unknown selector under
    /// a conjunction is left to the `<=` marker, elsewhere it prints
    /// as `?`. A disjunction is parenthesized when nested.
    fn render(&self, nested: bool, name: &dyn Fn(&Var) -> String) -> Option<String> {
        match self {
            Pred::True => None,
            Pred::False => Some("no thread".to_owned()),
            Pred::Unknown | Pred::Uniform => Some("?".to_owned()),
            Pred::Cmp(c) => Some(c.render_named(name)),
            Pred::Elect(_) => Some("elected".to_owned()),
            Pred::Not(p) => Some(format!("not ({})", p.render(false, name)?)),
            Pred::And(a, b) => {
                let parts: Vec<String> = [a, b]
                    .into_iter()
                    .filter(|p| !matches!(p.as_ref(), Pred::Unknown | Pred::Uniform))
                    .filter_map(|p| p.render(true, name))
                    .collect();
                (!parts.is_empty()).then(|| parts.join(" and "))
            }
            Pred::Or(a, b) => {
                let text = format!("{} or {}", a.render(true, name)?, b.render(true, name)?);
                Some(if nested { format!("({text})") } else { text })
            }
        }
    }
}

fn linear(tid: [i64; 3], shape: [u32; 3]) -> i64 {
    let [nx, ny, _] = shape.map(i64::from);
    tid[0] + nx * (tid[1] + ny * tid[2])
}

fn lane_of(linear: i64, shape: [u32; 3]) -> [i64; 3] {
    let [nx, ny, _] = shape.map(i64::from);
    [linear % nx, (linear / nx) % ny, linear / (nx * ny)]
}

/// Whether `tid` is the lowest lane of its warp that `active` admits, a
/// representative of the elected lane: yes when no lower lane may be
/// active, maybe when the leader is one of several lanes that may be.
fn elected(active: &Pred, tid: [i64; 3], shape: [u32; 3]) -> Tri {
    let [nx, ny, nz] = shape.map(i64::from);
    let threads = nx * ny * nz;
    if threads == 0 {
        return Tri::Maybe;
    }
    let linear = linear(tid, shape);
    let base = linear - linear % 32;
    let mut candidates = Vec::new();
    for l in base..(base + 32).min(threads) {
        match active.eval(lane_of(l, shape), shape) {
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

/// The threads of a CTA that execute a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadSet {
    pub pred: Pred,
}

fn negate(cmp: &str) -> &str {
    match cmp {
        "lt" => "ge",
        "ge" => "lt",
        "le" => "gt",
        "gt" => "le",
        "eq" => "ne",
        "ne" => "eq",
        other => other,
    }
}

/// `form` at thread `tid` in i128: an i64 base plus i64 coefficients
/// times thread indices below 1024 keeps every partial sum far inside
/// it, so a comparison of 64-bit operands reads exactly.
fn at(form: &Affine, tid: [i64; 3]) -> i128 {
    (form.terms.iter()).fold(i128::from(form.base.const_part()), |x, (v, c)| {
        x + i128::from(c.as_const().unwrap_or(0)) * i128::from(eval(v, tid))
    })
}

/// Whether `x cmp 0` holds for every `x` in [lo, hi], for none, or for
/// some: the values that satisfy an `lt`, `le`, `gt` or `ge` and those
/// that do not are each an interval, so the ends decide.
fn decide(cmp: &str, lo: i128, hi: i128) -> Tri {
    let zero = lo == 0 && hi == 0;
    let apart = lo > 0 || hi < 0;
    let (all, none) = match cmp {
        "eq" => (zero, apart),
        "ne" => (apart, zero),
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

fn holds(cmp: &str, x: i128) -> bool {
    match cmp {
        "lt" => x < 0,
        "le" => x <= 0,
        "gt" => x > 0,
        "ge" => x >= 0,
        "eq" => x == 0,
        "ne" => x != 0,
        _ => false,
    }
}

/// `v`, a CTA index or a floor or remainder of one, at the CTA `c`.
fn cta_index(v: &Var, c: [i64; 3]) -> i64 {
    match v {
        Var::Ctaid(Axis::X) => c[0],
        Var::Ctaid(Axis::Y) => c[1],
        Var::Ctaid(Axis::Z) => c[2],
        Var::Div(x, d) => cta_index(x, c).div_euclid(*d),
        Var::Mod(x, m) => cta_index(x, c).rem_euclid(*m),
        _ => 0,
    }
}

/// The axes of the CTA index `v` reads.
fn cta_axes(v: &Var, used: &mut [bool; 3]) {
    match v {
        Var::Ctaid(Axis::X) => used[0] = true,
        Var::Ctaid(Axis::Y) => used[1] = true,
        Var::Ctaid(Axis::Z) => used[2] = true,
        Var::Div(x, _) | Var::Mod(x, _) => cta_axes(x, used),
        _ => {}
    }
}

/// A CTA index, or a floor or remainder of one.
fn on_cta(v: &Var) -> bool {
    match v {
        Var::Ctaid(_) => true,
        Var::Div(x, _) | Var::Mod(x, _) => on_cta(x),
        _ => false,
    }
}

/// Constant coefficients and base over the thread and CTA indices and
/// their floors and remainders.
fn on_index(a: &Affine) -> bool {
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
    form: &Affine,
    cmp: &str,
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
    let (lo, hi) = Affine {
        terms: ctas,
        base: SymExpr::Const(0),
    }
    .range(range)?;
    let base = i128::from(form.base.as_const()?);
    Some(Pred::Cmp(match lo == hi {
        true => Constraint {
            form: Affine {
                terms: lanes,
                base: SymExpr::Const(i64::try_from(base + lo).ok()?),
            },
            cmp: cmp.to_owned(),
            cta: None,
            trip: None,
        },
        false => Constraint {
            form: form.clone(),
            cmp: cmp.to_owned(),
            cta: Some((lo, hi)),
            trip: None,
        },
    }))
}

fn on_lane_only(a: &Affine) -> bool {
    a.base.as_const().is_some()
        && a.terms.iter().all(|(v, c)| {
            c.as_const().is_some()
                && matches!(v, Var::Tid(_) | Var::Div(..) | Var::Mod(..))
                && depends_on_lane(v)
        })
}

/// The most CTAs `ThreadSet::launch` enumerates, and the most lanes it
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

impl Constraint {
    /// The value of the form's `%ctaid` terms at the CTA `c`.
    fn cta_part(&self, c: [i64; 3]) -> i128 {
        (self.form.terms.iter())
            .filter(|(v, _)| on_cta(v))
            .map(|(v, k)| i128::from(k.as_const().unwrap_or(0)) * i128::from(cta_index(v, c)))
            .sum()
    }

    /// `form cmp 0` with the constant moved to the right: `⌊%tid.x/32⌋ < 4`.
    pub fn render(&self) -> String {
        self.render_named(&|v| v.to_string())
    }

    /// The same, naming each variable by `name`.
    pub fn render_named(&self, name: &dyn Fn(&Var) -> String) -> String {
        let c = self.form.base.as_const().unwrap_or(0);
        let lhs = Affine {
            terms: self.form.terms.clone(),
            base: crate::analysis::scalar::symexpr::SymExpr::Const(0),
        };
        let op = match self.cmp.as_str() {
            "lt" => "<",
            "le" => "<=",
            "gt" => ">",
            "ge" => ">=",
            "eq" => "==",
            "ne" => "!=",
            other => other,
        };
        format!("{} {op} {}", lhs.render(name), -i128::from(c))
    }
}

impl ThreadSet {
    /// Whether the thread at `tid` may be in the set; an unknown
    /// selector includes every thread, as a bound.
    pub fn contains(&self, tid: [i64; 3], shape: [u32; 3]) -> bool {
        self.pred.eval(tid, shape) != Tri::No
    }

    /// Every selector is a thread-index condition.
    pub fn exact(&self) -> bool {
        self.pred.exact()
    }

    pub fn lane_exact(&self) -> bool {
        self.pred.lane_exact()
    }

    /// A comparison selecting the set reads the CTA index.
    pub fn varies_by_cta(&self) -> bool {
        let mut over = Vec::new();
        self.pred.over_ctas(&mut over);
        !over.is_empty()
    }

    /// The thread-index conditions as text, `⌊%tid.x/32⌋ < 4 and
    /// %tid.x == 0`; `None` when no such condition selects.
    pub fn render(&self) -> Option<String> {
        self.render_named(&|v| v.to_string())
    }

    /// The same, naming each variable by `name`.
    pub fn render_named(&self, name: &dyn Fn(&Var) -> String) -> Option<String> {
        self.pred
            .has_condition()
            .then(|| self.pred.render(false, name))
            .flatten()
    }

    /// How many threads of a block of this shape are in the set, when a
    /// thread-index condition selects them.
    pub fn count(&self, shape: [u32; 3]) -> Option<Counted> {
        self.tally(shape, |min, max| (min, max))
    }

    /// How many warps of a block of this shape have a lane in the set:
    /// PTX ISA §3.1 forms warps from consecutive thread ids.
    pub fn warps(&self, shape: [u32; 3]) -> Option<Counted> {
        self.tally(shape, |min, max| (u32::from(min > 0), u32::from(max > 0)))
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
        let mut over = Vec::new();
        self.pred.over_ctas(&mut over);
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
            .map(|l| lane_of(l, shape))
            .collect();
        if threads.is_empty() {
            return None;
        }
        let lanes: Vec<(i128, i128)> = (over.iter())
            .map(|k| {
                (threads.iter().map(|&t| at(&k.form, t)))
                    .fold((i128::MAX, i128::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)))
            })
            .collect();
        let n = |i: usize| if used[i] { i64::from(grid[i]) } else { 1 };
        let mut groups = HashMap::new();
        for z in 0..n(2) {
            for y in 0..n(1) {
                for x in 0..n(0) {
                    let c = [x, y, z];
                    let key: Vec<(Tri, i128)> = (over.iter().zip(&lanes))
                        .map(|(k, &(lo, hi))| {
                            let u = k.cta_part(c);
                            match decide(&k.cmp, lo + u, hi + u) {
                                Tri::Maybe => (Tri::Maybe, u),
                                t => (t, 0),
                            }
                        })
                        .collect();
                    let group = match groups.entry(key) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => {
                            let at = ThreadSet {
                                pred: self.pred.at_cta(c),
                            };
                            let (t, w) = (at.count(shape)?, at.warps(shape)?);
                            if t.certain != t.possible || w.certain != w.possible {
                                return None;
                            }
                            e.insert((0u64, t.certain, w.certain))
                        }
                    };
                    if group.1 > 0 {
                        group.0 = group.0.checked_add(weight(c)?)?;
                    }
                    if groups.len() as u64 > LANE_BUDGET / threads.len() as u64 {
                        return None;
                    }
                }
            }
        }
        let (mut sum_t, mut sum_w) = (0u64, 0u64);
        for (k, t, w) in groups.into_values() {
            sum_t = sum_t.checked_add(u64::from(t).checked_mul(k)?)?;
            sum_w = sum_w.checked_add(u64::from(w).checked_mul(k)?)?;
        }
        Some((sum_t.checked_mul(rest)?, sum_w.checked_mul(rest)?))
    }

    /// The loop whose counter the set's comparisons read, when one does.
    pub fn trip_loop(&self) -> Option<LoopId> {
        let mut over = Vec::new();
        self.pred.over_trips(&mut over);
        let l = over.first()?.trip?.0;
        over.iter()
            .all(|k| k.trip.is_some_and(|(t, _)| t == l))
            .then_some(l)
    }

    /// The threads and warps the set holds summed over trips 1 to `n` of
    /// loop `l`, whose counter its comparisons read. Each comparison
    /// `c·k + g cmp 0` keeps its truth on a lane between the trips
    /// around −g/c, so between those of every lane and comparison one
    /// trip is counted for the run. None where the set is not exact at a
    /// trip, or past `LANE_BUDGET` lane evaluations.
    pub fn over_trips(&self, shape: [u32; 3], l: LoopId, n: u64) -> Option<(u64, u64)> {
        let mut over = Vec::new();
        self.pred.over_trips(&mut over);
        let threads: Vec<[i64; 3]> = (0..shape.iter().map(|&n| i64::from(n)).product())
            .map(|t| lane_of(t, shape))
            .collect();
        let end = i128::from(n) + 1;
        let mut cuts = vec![1, end];
        for k in &over {
            let c = i128::from(k.form.terms.get(&Var::Iter(l))?.as_const()?);
            for &t in &threads {
                let (num, den) = if c < 0 {
                    (at(&k.form, t), -c)
                } else {
                    (-at(&k.form, t), c)
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
        let (mut sum_t, mut sum_w) = (0u64, 0u64);
        for run in cuts.windows(2) {
            let at = ThreadSet {
                pred: self.pred.at_trip(l, run[0])?,
            };
            let (t, w) = (at.count(shape)?, at.warps(shape)?);
            if t.certain != t.possible || w.certain != w.possible {
                return None;
            }
            let len = u64::try_from(run[1] - run[0]).ok()?;
            sum_t = sum_t.checked_add(u64::from(t.certain).checked_mul(len)?)?;
            sum_w = sum_w.checked_add(u64::from(w.certain).checked_mul(len)?)?;
        }
        Some((sum_t, sum_w))
    }

    /// Per warp, the fewest and the most lanes in the set, folded by
    /// `per_warp`. An elected lane is unspecified (PTX ISA §9.7.14.15
    /// promises only that the same leader is elected every time), so
    /// every active lane is tried as the leader, and no leader when no
    /// lane is certainly active, and the range covers those choices.
    fn tally(&self, shape: [u32; 3], per_warp: impl Fn(u32, u32) -> (u32, u32)) -> Option<Counted> {
        if !self.pred.has_condition() {
            return None;
        }
        let [nx, ny, nz] = shape.map(i64::from);
        let threads = nx * ny * nz;
        if threads == 0 {
            return None;
        }
        let mut certain = 0;
        let mut possible = 0;
        for base in (0..threads).step_by(32) {
            let lanes = base..(base + 32).min(threads);
            let lanes_as = |leader: Leader| {
                let (mut yes, mut maybe) = (0, 0);
                for l in lanes.clone() {
                    match self.pred.eval_as(lane_of(l, shape), shape, leader) {
                        Tri::Yes => yes += 1,
                        Tri::Maybe => maybe += 1,
                        Tri::No => {}
                    }
                }
                (yes, maybe)
            };
            let (min, max) = match self.pred.elect_active() {
                None => {
                    let (yes, maybe) = lanes_as(Leader::Lowest);
                    (yes, yes + maybe)
                }
                Some(active) => {
                    let state = |l| active.eval(lane_of(l, shape), shape);
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
            let (min, max) = per_warp(min, max);
            certain += min;
            possible += max;
        }
        Some(Counted { certain, possible })
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
    tracer: &'a AffineValueTracer<'a>,
    uniformity: &'a Uniformity,
    memo: HashMap<usize, Pred>,
    blocks: Vec<Option<Pred>>,
    bypass: Vec<Option<bool>>,
}

impl PredicateReader<'_> {
    fn block(&mut self, b: BlockId) -> Pred {
        if let Some(p) = &self.blocks[b.0 as usize] {
            return p.clone();
        }
        let mut pred = Pred::True;
        let mut cur = self.forest.doms.idom[b.0 as usize];
        while let Some(d) = cur {
            let blk = self.cfg.block(d);
            // The dominator's conditional branch, if it selects for b.
            let last = self.kernel.stmts[blk.start..blk.end]
                .iter()
                .enumerate()
                .rev()
                .find_map(|(i, s)| match s {
                    Stmt::Instr(instr) => Some((blk.start + i, instr)),
                    _ => None,
                });
            if let Some((pos, instr)) = last
                && instr.mnemonic == Opcode::Bra
                && let Some(guard) = instr.predicate
                && let [taken, fallthrough] = blk.succs[..]
            {
                let via = |s: BlockId| s == b || self.tracer.path_avoiding(s, b, d);
                let selected = match (via(taken), via(fallthrough)) {
                    (true, false) => Some(true),
                    (false, true) => Some(false),
                    _ => None,
                };
                if let Some(taken_edge) = selected {
                    let forest = self.forest;
                    let inside = |l: LoopId| {
                        let blocks = &forest.get(l).blocks;
                        blocks.binary_search(&b).is_ok() && blocks.binary_search(&d).is_ok()
                    };
                    let p = self.of(guard.reg, pos, 0).in_loops(&inside);
                    pred = pred.and(if taken_edge != guard.negated { p } else { !p });
                }
            }
            if d == ControlFlowGraph::ENTRY {
                break;
            }
            cur = self.forest.doms.idom[d.0 as usize];
        }
        if self.bypassed(b) {
            pred = pred.and(Pred::Unknown);
        }
        self.blocks[b.0 as usize] = Some(pred.clone());
        pred
    }

    /// Whether a condition off b's dominator chain decides whether b
    /// runs: so for b's immediate dominator, or one of its successors
    /// reaches b without passing it and leaves b's scope around b.
    fn bypassed(&mut self, b: BlockId) -> bool {
        if let Some(x) = self.bypass[b.0 as usize] {
            return x;
        }
        let x = match self.forest.doms.idom[b.0 as usize] {
            Some(d) => self.skips(d, b) || self.bypassed(d),
            None => false,
        };
        self.bypass[b.0 as usize] = Some(x);
        x
    }

    /// A successor of `d` reaches `c` without passing `d`, and also an exit
    /// of c's scope without passing `c`: a latch or exit target of c's
    /// innermost loop (its parent's for a loop header), or a kernel exit.
    fn skips(&self, d: BlockId, c: BlockId) -> bool {
        let innermost = self.forest.block_loop[c.0 as usize].map(|l| self.forest.get(l));
        let scope = match innermost {
            Some(l) if l.header == c => l.parent.map(|p| self.forest.get(p)),
            other => other,
        };
        let inside = |x: BlockId| scope.is_none_or(|s| s.blocks.binary_search(&x).is_ok());
        let exits: Vec<BlockId> = match scope {
            Some(s) => (s.blocks.iter())
                .flat_map(|&x| self.cfg.block(x).succs.iter().copied())
                .filter(|&t| !inside(t))
                .chain(s.latches.iter().copied())
                .collect(),
            None => (0..self.cfg.blocks.len() as u32)
                .map(BlockId)
                .filter(|&x| self.cfg.block(x).succs.is_empty())
                .collect(),
        };
        let around =
            |s: BlockId, t: BlockId| t != c && (s == t || self.tracer.path_avoiding(s, t, c));
        (self.cfg.block(d).succs.iter()).any(|&s| {
            s != c && self.tracer.path_avoiding(s, c, d) && exits.iter().any(|&t| around(s, t))
        })
    }

    fn of(&mut self, reg: Symbol, pos: usize, depth: u32) -> Pred {
        self.read(reg, pos, depth).0
    }

    /// The predicate `reg` holds at `pos`, uniform when it cannot be read
    /// but the uniformity analysis finds it so, and whether the depth
    /// budget cut the reading short (then it is not memoized).
    fn read(&mut self, reg: Symbol, pos: usize, depth: u32) -> (Pred, bool) {
        match self.read_definition(reg, pos, depth) {
            (Pred::Unknown, cut) if self.uniformity.is_uniform(self.tracer, reg, pos).is_ok() => {
                (Pred::Uniform, cut)
            }
            read => read,
        }
    }

    fn read_definition(&mut self, reg: Symbol, pos: usize, depth: u32) -> (Pred, bool) {
        let Some(Value::Result { op: def, .. }) =
            self.tracer.ssa.value_before(pos, Register::Named(reg))
        else {
            return (Pred::Unknown, false);
        };
        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return (Pred::Unknown, false);
        };
        if instr.predicate.is_some()
            || self
                .uniformity
                .gates_uniform(self.tracer, reg, pos)
                .is_err()
        {
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
        let ops = self.module.operand_ids(instr.operands);
        let second = ops
            .first()
            .and_then(|&id| match self.module.operand(id) {
                Operand::MultipleDestinations { children } => {
                    self.module.operand_ids(*children).get(1)
                }
                _ => None,
            })
            .is_some_and(
                |&id| matches!(self.module.operand(id), Operand::Register(r) if *r == reg),
            );
        let p = match (instr.mnemonic, second) {
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
        let mods: Vec<&str> = self
            .module
            .modifiers(instr)
            .iter()
            .map(|&m| self.module.interner.resolve(m))
            .collect();
        let ops = self.module.operand_ids(instr.operands);
        let reg = |id| match self.module.operand(id) {
            Operand::Register(r) => Some(*r),
            _ => None,
        };
        match (instr.mnemonic, mods.as_slice(), ops) {
            (Opcode::Setp, _, _) => (
                condition(self.module, self.kernel, self.tracer, self.uniformity, def)
                    .unwrap_or(Pred::Unknown),
                false,
            ),
            (Opcode::And | Opcode::Or, ["pred"], [_, a, b]) => {
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
            (Opcode::Mov, ["pred"], [_, a]) => match self.module.operand(*a) {
                Operand::Immediate(t) => match parse_int(self.module.interner.resolve(*t)) {
                    Some(0) => (Pred::False, false),
                    Some(_) => (Pred::True, false),
                    None => (Pred::Unknown, false),
                },
                _ => (Pred::Unknown, false),
            },
            (Opcode::Not, ["pred"], [_, a]) => match reg(*a) {
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
    pub blocks: Vec<ThreadSet>,
    pub guards: HashMap<usize, Pred>,
    /// Per `cp.async` whose src-size or ignore-src is a register, the
    /// threads that read its source, memory operand 1: those whose
    /// ignore-src is false; with a src-size register, unknown.
    pub sources: HashMap<usize, Pred>,
    /// The edges no thread follows: the other way out of a branch
    /// every thread takes the same way.
    pub dead_edges: Vec<(BlockId, BlockId)>,
}

impl ThreadSets {
    /// The threads that execute statement `stmt` of block `b`: the
    /// block's set under the statement's guard.
    pub fn of(&self, b: BlockId, stmt: usize) -> ThreadSet {
        let block = &self.blocks[b.0 as usize];
        match self.guards.get(&stmt) {
            Some(guard) => ThreadSet {
                pred: block.pred.clone().and(guard.clone()),
            },
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
    pub fn of_access(&self, b: BlockId, stmt: usize, operand: usize) -> ThreadSet {
        let block = &self.blocks[b.0 as usize];
        match self.guard(stmt, operand) {
            Some(guard) => ThreadSet {
                pred: block.pred.clone().and(guard),
            },
            None => block.clone(),
        }
    }

    /// Whether no thread executes statement `stmt` of block `b`.
    pub fn never(&self, b: BlockId, stmt: usize) -> bool {
        self.blocks[b.0 as usize].pred == Pred::False
            || self.guards.get(&stmt) == Some(&Pred::False)
    }
}

pub(crate) fn thread_sets(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    tracer: &AffineValueTracer,
    uniformity: &Uniformity,
) -> ThreadSets {
    let mut reader = PredicateReader {
        module,
        kernel,
        cfg,
        forest,
        tracer,
        uniformity,
        memo: HashMap::new(),
        blocks: vec![None; cfg.blocks.len()],
        bypass: vec![None; cfg.blocks.len()],
    };
    let blocks: Vec<ThreadSet> = (0..cfg.blocks.len() as u32)
        .map(|b| ThreadSet {
            pred: reader.block(BlockId(b)),
        })
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
        let b = tracer.block_of(pos);
        let p = (reader.of(guard.reg, pos, 0))
            .in_loops(&|l| forest.get(l).blocks.binary_search(&b).is_ok());
        guards.insert(pos, if guard.negated { !p } else { p });
    }
    let mut dead_edges = Vec::new();
    for (b, blk) in cfg.blocks.iter_enumerated() {
        if let Some((pos, instr)) = tracer.last_instr(b)
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
        let b = tracer.block_of(pos);
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
    for (b, set) in blocks.iter_mut().enumerate() {
        if !doms.is_reachable(BlockId(b as u32)) {
            set.pred = Pred::False;
        }
    }
    ThreadSets {
        blocks,
        guards,
        sources,
        dead_edges,
    }
}

/// The `setp` at `def` as `a − b cmp 0` over the thread index, its
/// operands read strictly (a write under a guard is refused, not taken
/// as the value) under the tracer's bindings (`--bind`, the block
/// shape), when it compares two such forms as integers: an `lt`, `le`,
/// `gt` or `ge`
/// only when both lie in the range of its type, [0, 2^width) or
/// [−2^(width−1), 2^(width−1)), on every thread, an `eq` or `ne` only
/// when `a − b` lies in (−2^width, 2^width), so that it is zero
/// exactly when it is zero modulo 2^width. A form free of the thread
/// index whose comparison the uniformity analysis finds divergent is
/// none, unless it is a loop counter's compared within its type, one
/// value at each trip.
fn condition(
    module: &Module,
    kernel: &Kernel,
    tracer: &AffineValueTracer,
    uniformity: &Uniformity,
    def: usize,
) -> Option<Pred> {
    let Stmt::Instr(setp) = &kernel.stmts[def] else {
        return None;
    };
    let mods: Vec<&str> = module
        .modifiers(setp)
        .iter()
        .map(|&m| module.interner.resolve(m))
        .collect();
    let cmp = mods.first()?.to_string();
    let ty = *mods.last()?;
    // PTX ISA §9.7.6.2, §9.7.7.2: a floating-point `setp` compares floats, a packed one per half.
    if !matches!(cmp.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne")
        || ty.starts_with('f')
        || ty.starts_with("bf")
        || mods.iter().any(|m| matches!(*m, "and" | "or" | "xor"))
    {
        return None;
    }
    let ops = module.operand_ids(setp.operands);
    if ops.len() != 3 {
        return None;
    }
    let bound = |op| {
        let x = (tracer.strictly(|| tracer.trace_operand(op, def, None, 0))).ok()?;
        Some(tracer.bindings.map_or(x.clone(), |m| x.bind(m)))
    };
    let (a, b) = (bound(ops[1])?, bound(ops[2])?);
    let form = a.clone() - b.clone();
    // %ctaid below the bound grid, PTX ISA §10.6.
    let range = |v: &Var| match v {
        Var::Ctaid(_) | Var::Iter(_) => tracer.bound(v),
        _ => any_block(v),
    };
    let fits = |x: &Affine, min: i128, end: i128| {
        (x.range(&range)).is_some_and(|(lo, hi)| min <= lo && hi < end)
    };
    let (sign, width) = ty.split_at_checked(1).unwrap_or_default();
    let comparable = |a: &Affine, b: &Affine| {
        width
            .parse::<u32>()
            .is_ok_and(|w| match (cmp.as_str(), sign) {
                ("eq" | "ne", _) => fits(&(a.clone() - b.clone()), 1 - (1 << w), 1 << w),
                (_, "u") => fits(a, 0, 1 << w) && fits(b, 0, 1 << w),
                (_, "s") => {
                    fits(a, -1 << (w - 1), 1 << (w - 1)) && fits(b, -1 << (w - 1), 1 << (w - 1))
                }
                _ => false,
            })
    };
    let constant = |op| {
        (tracer
            .strictly(|| tracer.trace_operand(op, def, None, 0))
            .ok())
        .and_then(|x| tracer.constant(&x))
        .map(|c| Affine::invariant(SymExpr::Const(c)))
    };
    if let (Some(x), Some(y)) = (constant(ops[1]), constant(ops[2]))
        && comparable(&x, &y)
    {
        let d = (x - y).base.as_const()?;
        return Some(if holds(&cmp, d.into()) {
            Pred::True
        } else {
            Pred::False
        });
    }
    let lanes = form.terms.keys().any(depends_on_lane);
    let trip = per_trip(tracer, &form, def).filter(|_| comparable(&a, &b));
    if !lanes && trip.is_none() && !uniformity.is_uniform_def(def) {
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
        && let Some(p) = across_ctas(&form, &cmp, &|v| tracer.bound(v).or_else(|| any_block(v)))
    {
        Some(p)
    } else if !lanes {
        Some(Pred::Uniform)
    } else {
        None
    }
}

/// A form of the thread index and one loop's counter, read in that loop
/// so its current trip's, the loop's trip count a constant n: the loop
/// and the range of the counter's term over trips 1 to n.
fn per_trip(
    tracer: &AffineValueTracer,
    form: &Affine,
    def: usize,
) -> Option<(LoopId, (i128, i128))> {
    let mut rest = form.clone();
    let mut counters = (form.terms.iter()).filter(|(v, _)| matches!(v, Var::Iter(_)));
    let (Some((v @ Var::Iter(l), c)), None) = (counters.next(), counters.next()) else {
        return None;
    };
    rest.terms.remove(v);
    let c = i128::from(c.as_const().filter(|&c| c != 0)?);
    let (_, n) = tracer.bound(v)?;
    let inside = tracer
        .forest
        .get(*l)
        .blocks
        .binary_search(&tracer.block_of(def))
        .is_ok();
    (on_lane_only(&rest) && inside)
        .then_some((*l, (c.min(c * i128::from(n)), c.max(c * i128::from(n)))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scalar::affine::Axis;
    use crate::analysis::scalar::symexpr::SymExpr;

    fn exact(n: u32) -> Counted {
        Counted {
            certain: n,
            possible: n,
        }
    }

    fn warps_below_4() -> Constraint {
        let warp = Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 32));
        Constraint {
            form: warp + Affine::invariant(SymExpr::Const(-4)),
            cmp: "lt".to_owned(),
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
        let counter = |rest: Affine| {
            Pred::Cmp(Constraint {
                form: Affine::var(Var::Iter(LoopId(0))) + rest,
                cmp: "lt".to_owned(),
                cta: None,
                trip: Some((LoopId(0), (1, 512))),
            })
        };
        let c = |n: i64| Affine::invariant(SymExpr::Const(n));
        let below = counter(c(-511));
        let set = ThreadSet {
            pred: below.clone(),
        };
        assert!(!set.exact() && set.lane_exact());
        let some = Counted {
            certain: 0,
            possible: 32,
        };
        assert_eq!(set.count([32, 1, 1]), Some(some));
        let always = ThreadSet {
            pred: counter(c(-513)),
        };
        assert_eq!(always.count([32, 1, 1]), Some(exact(32)));
        let lanes = ThreadSet {
            pred: counter(Affine::var(Var::Tid(Axis::X)) + c(-511)),
        };
        assert!(!lanes.lane_exact());
        assert_eq!(below.clone().in_loops(&|_| false), Pred::Uniform);
        assert_eq!(below.clone().in_loops(&|_| true), below);
    }

    /// Over a loop's trips, the threads of each trip summed: by runs of
    /// trips between the lanes' breakpoints, as enumerating every trip
    /// gives, for comparisons of the counter alone and with lanes, either
    /// sign of step, every comparison, and two comparisons at once.
    #[test]
    fn a_set_over_trips_sums_each_trip_s_threads() {
        let l = LoopId(0);
        let cmp = |step: i64, lanes: Affine, cmp: &str, n: i128| {
            let (a, b) = (i128::from(step), i128::from(step) * n);
            Pred::Cmp(Constraint {
                form: Affine::var(Var::Iter(l)).scale(SymExpr::Const(step)) + lanes,
                cmp: cmp.to_owned(),
                cta: None,
                trip: Some((l, (a.min(b), a.max(b)))),
            })
        };
        let c = |x: i64| Affine::invariant(SymExpr::Const(x));
        let lane = |m: i64| Affine::var(Var::Mod(Box::new(Var::Tid(Axis::X)), m));
        let brute = |set: &ThreadSet, shape: [u32; 3], n: u64| {
            (1..=i128::from(n)).try_fold((0u64, 0u64), |(t, w), k| {
                let at = ThreadSet {
                    pred: set.pred.at_trip(l, k)?,
                };
                Some((
                    t + u64::from(at.count(shape)?.certain),
                    w + u64::from(at.warps(shape)?.certain),
                ))
            })
        };
        let shape = [128, 1, 1];
        for n in [1u64, 7, 40] {
            let n128 = i128::from(n);
            for op in ["lt", "le", "gt", "ge", "eq", "ne"] {
                for (step, lanes) in [(1, c(-5)), (64, lane(64) + c(-300)), (-3, lane(32) + c(40))]
                {
                    let set = ThreadSet {
                        pred: cmp(step, lanes, op, n128),
                    };
                    assert_eq!(
                        set.over_trips(shape, l, n),
                        brute(&set, shape, n),
                        "{op} {step} {n}"
                    );
                }
            }
            let both = ThreadSet {
                pred: cmp(1, c(-3), "ge", n128).and(cmp(64, lane(64) + c(-1000), "lt", n128)),
            };
            assert_eq!(both.over_trips(shape, l, n), brute(&both, shape, n));
        }
        let k_below_511 = ThreadSet {
            pred: cmp(1, c(-511), "lt", 512),
        };
        assert_eq!(
            k_below_511.over_trips([256, 1, 1], l, 512),
            Some((510 * 256, 510 * 8))
        );
    }

    #[test]
    fn a_constraint_counts_and_renders() {
        let c = warps_below_4();
        assert_eq!(c.render(), "⌊%tid.x/32⌋ < 4");
        let min = Constraint {
            form: Affine::var(Var::Tid(Axis::X)) + Affine::invariant(SymExpr::Const(i64::MIN)),
            ..warps_below_4()
        };
        assert_eq!(min.render(), "%tid.x < 9223372036854775808");
        let set = ThreadSet { pred: Pred::Cmp(c) };
        assert!(set.exact());
        assert_eq!(set.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(set.count([64, 1, 1]), Some(exact(64)));
        assert_eq!(ThreadSet { pred: Pred::True }.count([256, 1, 1]), None);
        let lane0 = ThreadSet {
            pred: Pred::Cmp(Constraint {
                form: Affine::var(Var::Tid(Axis::X)),
                cmp: "eq".to_owned(),
                cta: None,
                trip: None,
            }),
        };
        assert_eq!(lane0.count([128, 2, 1]), Some(exact(2)));
    }

    #[test]
    fn an_unknown_selector_keeps_the_index_conditions_as_a_bound() {
        let set = ThreadSet {
            pred: Pred::Cmp(warps_below_4()).and(Pred::Unknown),
        };
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
        let unknown = ThreadSet {
            pred: Pred::True.and(Pred::Unknown),
        };
        assert_eq!(unknown.pred, Pred::Unknown);
        assert!(unknown.contains([5, 0, 0], [256, 1, 1]));
        assert_eq!(unknown.render(), None);
        assert_eq!(unknown.count([256, 1, 1]), None);
    }

    #[test]
    fn or_and_not_evaluate_three_valued_and_render() {
        let lane0 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)),
            cmp: "eq".to_owned(),
            cta: None,
            trip: None,
        });
        let either = ThreadSet {
            pred: lane0.clone().or(Pred::Cmp(warps_below_4())),
        };
        assert_eq!(either.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(
            either.render().as_deref(),
            Some("%tid.x == 0 or ⌊%tid.x/32⌋ < 4")
        );
        let neither = ThreadSet {
            pred: !either.pred.clone(),
        };
        assert!(neither.exact());
        assert_eq!(neither.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(
            neither.render().as_deref(),
            Some("not (%tid.x == 0 or ⌊%tid.x/32⌋ < 4)")
        );
        assert_eq!((!lane0.clone()).eval([0, 0, 0], [64, 1, 1]), Tri::No);
        assert_eq!(!!lane0.clone(), lane0);
        let hedged = ThreadSet {
            pred: lane0.clone().or(Pred::Unknown),
        };
        assert!(!hedged.exact());
        assert_eq!(hedged.pred.eval([0, 0, 0], [64, 1, 1]), Tri::Yes);
        assert_eq!(hedged.pred.eval([1, 0, 0], [64, 1, 1]), Tri::Maybe);
        assert_eq!(
            hedged.count([64, 1, 1]),
            Some(Counted {
                certain: 1,
                possible: 64
            })
        );
        assert_eq!(hedged.render().as_deref(), Some("%tid.x == 0 or ?"));
        assert_eq!(!Pred::Unknown, Pred::Unknown);
        assert_eq!(!Pred::True, Pred::False);
        assert_eq!((!Pred::True).eval([0, 0, 0], [64, 1, 1]), Tri::No);
        assert_eq!(Pred::False.or(lane0.clone()), lane0);
        assert_eq!(Pred::Unknown.and(Pred::False), Pred::False);
        let none = ThreadSet { pred: Pred::False };
        assert_eq!(none.count([64, 1, 1]), Some(exact(0)));
        assert_eq!(none.render().as_deref(), Some("no thread"));
    }

    #[test]
    fn elect_is_one_lane_per_warp_of_the_active_threads() {
        let all = ThreadSet {
            pred: Pred::Elect(Box::new(Pred::True)),
        };
        assert_eq!(all.count([128, 1, 1]), Some(exact(4)));
        assert_eq!(all.pred.eval([32, 0, 0], [128, 1, 1]), Tri::Yes);
        assert_eq!(all.pred.eval([33, 0, 0], [128, 1, 1]), Tri::No);
        assert_eq!(all.render().as_deref(), Some("elected"));
        assert_eq!(all.pred.eval([0, 0, 0], [0, 0, 0]), Tri::Maybe);
        let upper = Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)) + Affine::invariant(SymExpr::Const(-40)),
            cmp: "ge".to_owned(),
            cta: None,
            trip: None,
        });
        let of_upper = ThreadSet {
            pred: Pred::Elect(Box::new(upper.clone())),
        };
        assert_eq!(of_upper.count([128, 1, 1]), Some(exact(3)));
        let warp0 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 32)),
            cmp: "eq".to_owned(),
            cta: None,
            trip: None,
        });
        // A warp-uniform condition on the elected lane is exact.
        let leader_of_warp0 = ThreadSet {
            pred: warp0.and(Pred::Elect(Box::new(Pred::True))),
        };
        assert_eq!(leader_of_warp0.count([128, 1, 1]), Some(exact(1)));
        assert_eq!(
            leader_of_warp0.render().as_deref(),
            Some("⌊%tid.x/32⌋ == 0 and elected")
        );
        // A lane-varying one is not: the leader may or may not be lane 0.
        let lane0 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)),
            cmp: "eq".to_owned(),
            cta: None,
            trip: None,
        });
        let leader_is_lane0 = ThreadSet {
            pred: lane0.and(Pred::Elect(Box::new(Pred::True))),
        };
        assert_eq!(
            leader_is_lane0.count([128, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 1
            })
        );
        let others = ThreadSet {
            pred: !Pred::Elect(Box::new(Pred::True)),
        };
        assert_eq!(others.count([128, 1, 1]), Some(exact(124)));
        let hedged = ThreadSet {
            pred: Pred::Elect(Box::new(upper.or(Pred::Unknown))),
        };
        assert!(!hedged.exact());
        // Warp 0 may have no active lane, so no leader.
        assert_eq!(
            hedged.count([128, 1, 1]),
            Some(Counted {
                certain: 3,
                possible: 4
            })
        );
        let maybe = Pred::Elect(Box::new(Pred::Uniform));
        let range = |certain, possible| Some(Counted { certain, possible });
        assert_eq!(set_of(maybe.clone()).count([64, 1, 1]), range(0, 2));
        assert_eq!(set_of(!maybe).count([64, 1, 1]), range(62, 64));
    }

    fn set_of(pred: Pred) -> ThreadSet {
        ThreadSet { pred }
    }

    /// `a·%tid.x + b·%ctaid.x + base cmp 0`, `%ctaid.x` below `ctas`.
    fn over_ctas(a: i64, b: i64, base: i64, cmp: &str, ctas: i64) -> Pred {
        let (lo, hi) = (0, i128::from(b) * i128::from(ctas - 1));
        Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)).scale(SymExpr::Const(a))
                + Affine::var(Var::Ctaid(Axis::X)).scale(SymExpr::Const(b))
                + Affine::invariant(SymExpr::Const(base)),
            cmp: cmp.to_owned(),
            cta: Some((lo.min(hi), lo.max(hi))),
            trip: None,
        })
    }

    #[test]
    fn a_launch_sums_each_cta_s_threads_and_warps() {
        let set = |pred| ThreadSet { pred };
        // t < 2c over CTAs 0..4: 0, 2, 4 and 6 threads in 0, 1, 1, 1 warps.
        let below = over_ctas(1, -2, 0, "lt", 4);
        assert_eq!(
            set(below.clone()).launch([32, 1, 1], [4, 1, 1]),
            Some((12, 3))
        );
        assert_eq!(
            set(over_ctas(1, -2, 0, "eq", 4)).launch([32, 1, 1], [4, 1, 1]),
            Some((4, 4))
        );
        // An axis no comparison reads multiplies.
        assert_eq!(
            set(below.clone()).launch([32, 1, 1], [4, 1 << 25, 1]),
            Some((12 << 25, 3 << 25))
        );
        // A set free of %ctaid: its count at every CTA.
        let lanes = Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)) + Affine::invariant(SymExpr::Const(-16)),
            cmp: "lt".to_owned(),
            cta: None,
            trip: None,
        });
        assert_eq!(
            set(lanes.clone()).launch([32, 1, 1], [3, 2, 1]),
            Some((96, 6))
        );
        // No condition, one not decided at a CTA, or no block shape: no count.
        assert_eq!(set(Pred::True).launch([32, 1, 1], [4, 1, 1]), None);
        assert_eq!(set(below.clone()).launch([0, 0, 0], [4, 1, 1]), None);
        assert_eq!(
            set(below.clone().and(Pred::Uniform)).launch([32, 1, 1], [4, 1, 1]),
            None
        );
        // An election of one lane per warp counts; one among lanes that a
        // comparison then splits does not.
        let first_two = over_ctas(0, 1, -2, "lt", 4);
        let elected = Pred::Elect(Box::new(Pred::True));
        assert_eq!(
            set(first_two.and(elected.clone())).launch([64, 1, 1], [4, 1, 1]),
            Some((4, 4))
        );
        assert_eq!(set(below.and(elected)).launch([32, 1, 1], [4, 1, 1]), None);
        // Past the budgets: 2^24 CTAs on the axis read, or 130,944 groups of
        // a lane-dependent comparison times 1024 lanes.
        let one = over_ctas(1, -1, 0, "lt", (1 << 24) + 1);
        assert_eq!(set(one).launch([32, 1, 1], [(1 << 24) + 1, 1, 1]), None);
        let spread = over_ctas(128, -1, 0, "lt", 1 << 17);
        assert_eq!(set(spread).launch([1024, 1, 1], [1 << 17, 1, 1]), None);
    }

    /// Weighted, CTA c counts `weight(c)` times: t < 2c over 4 CTAs with
    /// weight c holds 0·0 + 2·1 + 4·2 + 6·3 threads; a weight on the y
    /// axis, which no comparison reads, enumerates it too; a weight that
    /// does not evaluate leaves no count, unless only at CTA 0, where the
    /// set holds no thread, and stops at the first CTA that holds one.
    #[test]
    fn a_weighted_launch_counts_each_cta_its_weight_times() {
        let below = ThreadSet {
            pred: over_ctas(1, -2, 0, "lt", 4),
        };
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
        let at = |cmp: &str| ThreadSet {
            pred: Pred::Cmp(Constraint {
                form: Affine::var(Var::Tid(Axis::X))
                    + Affine::var(Var::Ctaid(Axis::X)).scale(SymExpr::Const(-2)),
                cmp: cmp.to_owned(),
                cta: Some((-6, 0)),
                trip: None,
            }),
        };
        let range = |certain, possible| Some(Counted { certain, possible });
        assert_eq!(at("lt").count([32, 1, 1]), range(0, 6));
        assert_eq!(at("le").count([32, 1, 1]), range(1, 7));
        assert_eq!(at("gt").count([32, 1, 1]), range(25, 31));
        assert_eq!(at("ge").count([32, 1, 1]), range(26, 32));
        assert_eq!(at("eq").count([32, 1, 1]), range(0, 7));
        assert_eq!(at("ne").count([32, 1, 1]), range(25, 32));
        assert_eq!((!at("lt").pred).eval([3, 0, 0], [32, 1, 1]), Tri::Maybe);
        assert_eq!((!at("lt").pred).eval([7, 0, 0], [32, 1, 1]), Tri::Yes);
        assert!(!at("lt").exact() && !at("lt").lane_exact());
        // Of the CTA index alone: whole warps, so the lanes stay known.
        let cta = ThreadSet {
            pred: Pred::Cmp(Constraint {
                form: Affine::var(Var::Ctaid(Axis::X)) + Affine::invariant(SymExpr::Const(-2)),
                cmp: "lt".to_owned(),
                cta: Some((0, 3)),
                trip: None,
            }),
        };
        assert_eq!(cta.count([64, 1, 1]), range(0, 64));
        assert!(!cta.exact() && cta.lane_exact());
    }

    #[test]
    fn warps_count_the_groups_of_32_with_a_lane_in_the_set() {
        let warp0 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 32)),
            cmp: "eq".to_owned(),
            cta: None,
            trip: None,
        });
        let lane0 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)),
            cmp: "eq".to_owned(),
            cta: None,
            trip: None,
        });
        let set = |pred| ThreadSet { pred };
        assert_eq!(
            set(Pred::Cmp(warps_below_4())).warps([256, 1, 1]),
            Some(exact(4))
        );
        assert_eq!(set(lane0.clone()).warps([128, 1, 1]), Some(exact(1)));
        assert_eq!(set(!lane0.clone()).warps([64, 1, 1]), Some(exact(2)));
        assert_eq!(set(!lane0.clone()).count([64, 1, 1]), Some(exact(63)));
        // 100 threads span four warps, the last one partial.
        let below_100 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Tid(Axis::X)) + Affine::invariant(SymExpr::Const(-100)),
            cmp: "lt".to_owned(),
            cta: None,
            trip: None,
        });
        assert_eq!(set(below_100).warps([128, 1, 1]), Some(exact(4)));
        // Every warp with an active lane has a leader, whichever lane.
        assert_eq!(
            set(Pred::Elect(Box::new(Pred::True))).warps([128, 1, 1]),
            Some(exact(4))
        );
        assert_eq!(
            set(warp0.and(Pred::Elect(Box::new(Pred::True)))).warps([128, 1, 1]),
            Some(exact(1))
        );
        // Whether warp 0's leader is lane 0 is not known.
        assert_eq!(
            set(lane0.clone().and(Pred::Elect(Box::new(Pred::True)))).warps([128, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 1
            })
        );
        // Lane 0 is certainly in; the other lanes of every warp may be.
        let hedged = set(lane0.or(Pred::Unknown));
        assert_eq!(
            hedged.warps([64, 1, 1]),
            Some(Counted {
                certain: 1,
                possible: 2
            })
        );
        assert_eq!(set(Pred::True).warps([64, 1, 1]), None);
    }

    #[test]
    fn a_uniform_unknown_is_lane_exact_but_not_exact() {
        let set = |pred: Pred| ThreadSet { pred };
        let uniform = set(Pred::Cmp(warps_below_4()).and(Pred::Uniform));
        assert!(!uniform.exact() && uniform.lane_exact());
        assert_eq!(uniform.render().as_deref(), Some("⌊%tid.x/32⌋ < 4"));
        assert_eq!(
            uniform.count([256, 1, 1]),
            Some(Counted {
                certain: 0,
                possible: 128
            })
        );
        let unknown = set(Pred::Cmp(warps_below_4()).and(Pred::Unknown));
        assert!(!unknown.exact() && !unknown.lane_exact());
        // Under a disjunction or a negation the uniform unknown decides
        // between two lane sets.
        let lanes = || Pred::Cmp(warps_below_4());
        assert!(!set(Pred::Uniform.or(lanes())).lane_exact());
        assert!(!set(!(Pred::Uniform.and(lanes()))).lane_exact());
        assert!(set(Pred::Elect(Box::new(Pred::Uniform.and(lanes())))).lane_exact());
    }

    /// Per `cp.async` of `body`, in order: the threads that read its
    /// source as stored, and its guards on memory operands 0 and 1.
    /// `%r1` holds the thread index.
    fn copies(body: &str) -> Vec<(Option<Pred>, Option<Pred>, Option<Pred>)> {
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
        let t = AffineValueTracer::new(&m, k, &cfg, &forest);
        let sets = thread_sets(&m, k, &cfg, &forest, &t, t.uniformity());
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
        let text = |p: &Option<Pred>| p.clone().and_then(|pred| ThreadSet { pred }.render());
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
        assert_eq!(after, &Some(Pred::Uniform));
    }
}
