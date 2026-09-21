//! Which threads of a CTA execute a block. A conditional branch on the
//! block's dominator chain fixes its predicate on entry when the block
//! is reachable from only one of the two successors without passing
//! the branch again (its last evaluation decided); when each such
//! predicate is a comparison of affine forms over the thread index
//! alone, the block runs exactly on the threads satisfying all of them,
//! counted by enumeration over the block shape. A predicate that
//! involves anything else (a CTA index, a parameter, a loaded value)
//! is an unknown selector, and the block's counts stay bounds.

use crate::analysis::control_flow::loops::LoopForest;
use crate::analysis::scalar::affine::{Affine, Var};
use crate::analysis::scalar::lane_eval::eval_lane;
use crate::analysis::scalar::trace::{AffineValueTracer, ReachingDefinition};
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{Kernel, Module, Operand, Stmt};
use crate::support::intern::Symbol;
use std::collections::HashMap;

/// One condition `form cmp 0` over the thread index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub form: Affine,
    pub cmp: String,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            Pred::Cmp(c) => Pred::Cmp(Constraint {
                form: c.form,
                cmp: negate(&c.cmp).to_owned(),
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
            (a, b) => Pred::And(Box::new(a), Box::new(b)),
        }
    }

    pub fn or(self, other: Pred) -> Pred {
        match (self, other) {
            (Pred::True, _) | (_, Pred::True) => Pred::True,
            (Pred::False, p) | (p, Pred::False) => p,
            (Pred::Unknown, Pred::Unknown) => Pred::Unknown,
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
            Pred::Unknown => Tri::Maybe,
            Pred::Cmp(c) => {
                if holds(&c.cmp, eval_lane(&c.form, tid)) {
                    Tri::Yes
                } else {
                    Tri::No
                }
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
            Pred::True | Pred::False | Pred::Cmp(_) => true,
            Pred::Unknown => false,
            Pred::Elect(p) | Pred::Not(p) => p.exact(),
            Pred::And(a, b) | Pred::Or(a, b) => a.exact() && b.exact(),
        }
    }

    fn has_condition(&self) -> bool {
        match self {
            Pred::True | Pred::Unknown => false,
            Pred::False | Pred::Cmp(_) | Pred::Elect(_) => true,
            Pred::Not(p) => p.has_condition(),
            Pred::And(a, b) | Pred::Or(a, b) => a.has_condition() || b.has_condition(),
        }
    }

    /// The thread-index conditions as text; an unknown selector under
    /// a conjunction is left to the `<=` marker, elsewhere it prints
    /// as `?`. A disjunction is parenthesized when nested.
    fn render(&self, nested: bool) -> Option<String> {
        match self {
            Pred::True => None,
            Pred::False => Some("no thread".to_owned()),
            Pred::Unknown => Some("?".to_owned()),
            Pred::Cmp(c) => Some(c.render()),
            Pred::Elect(_) => Some("elected".to_owned()),
            Pred::Not(p) => Some(format!("not ({})", p.render(false)?)),
            Pred::And(a, b) => {
                let parts: Vec<String> = [a, b]
                    .into_iter()
                    .filter(|p| !matches!(p.as_ref(), Pred::Unknown))
                    .filter_map(|p| p.render(true))
                    .collect();
                (!parts.is_empty()).then(|| parts.join(" and "))
            }
            Pred::Or(a, b) => {
                let text = format!("{} or {}", a.render(true)?, b.render(true)?);
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

fn holds(cmp: &str, x: i64) -> bool {
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

fn on_lane_only(a: &Affine) -> bool {
    a.base.as_const().is_some()
        && a.terms.iter().all(|(v, c)| {
            c.as_const().is_some()
                && matches!(v, Var::Tid(_) | Var::Div(..) | Var::Mod(..))
                && crate::analysis::scalar::lane_eval::depends_on_lane(v)
        })
}

impl Constraint {
    /// `form cmp 0` with the constant moved to the right: `⌊%tid.x/32⌋ < 4`.
    pub fn render(&self) -> String {
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
        format!("{lhs} {op} {}", -c)
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

    /// The thread-index conditions as text, `⌊%tid.x/32⌋ < 4 and
    /// %tid.x == 0`; `None` when no such condition selects.
    pub fn render(&self) -> Option<String> {
        self.pred
            .has_condition()
            .then(|| self.pred.render(false))
            .flatten()
    }

    /// How many threads of a block of this shape are in the set, when a
    /// thread-index condition selects them. An elected lane is
    /// unspecified (PTX ISA §9.7.14.15 promises only that the same
    /// leader is elected every time), so every active lane is tried as
    /// the leader and the count ranges over those choices.
    pub fn count(&self, shape: [u32; 3]) -> Option<Counted> {
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
            let tally = |leader: Leader| {
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
            let candidates: Vec<i64> = match self.pred.elect_active() {
                Some(active) => lanes
                    .clone()
                    .filter(|&l| active.eval(lane_of(l, shape), shape) != Tri::No)
                    .collect(),
                None => Vec::new(),
            };
            let (min, max) = if self.pred.elect_active().is_none() {
                let (yes, maybe) = tally(Leader::Lowest);
                (yes, yes + maybe)
            } else if candidates.is_empty() {
                let (yes, maybe) = tally(Leader::Nobody);
                (yes, yes + maybe)
            } else {
                candidates
                    .iter()
                    .map(|&k| tally(Leader::Lane(k)))
                    .fold((u32::MAX, 0), |(min, max), (yes, maybe)| {
                        (min.min(yes), max.max(yes + maybe))
                    })
            };
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
    memo: HashMap<usize, Pred>,
    blocks: Vec<Option<Pred>>,
}

impl PredicateReader<'_> {
    fn block(&mut self, b: BlockId) -> Pred {
        if let Some(p) = &self.blocks[b.0 as usize] {
            return p.clone();
        }
        let sym_bra = self.module.interner.get("bra");
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
                && Some(instr.mnemonic) == sym_bra
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
                    let p = self.of(guard.reg, pos, 0);
                    pred = pred.and(if taken_edge != guard.negated { p } else { !p });
                }
            }
            if d == ControlFlowGraph::ENTRY {
                break;
            }
            cur = self.forest.doms.idom[d.0 as usize];
        }
        self.blocks[b.0 as usize] = Some(pred.clone());
        pred
    }

    fn of(&mut self, reg: Symbol, pos: usize, depth: u32) -> Pred {
        self.read(reg, pos, depth).0
    }

    /// The predicate `reg` holds at `pos`, and whether the depth budget
    /// cut the reading short (then it is not memoized).
    fn read(&mut self, reg: Symbol, pos: usize, depth: u32) -> (Pred, bool) {
        let ReachingDefinition::Def(def) = self.tracer.reach_def(reg, pos, None) else {
            return (Pred::Unknown, false);
        };
        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return (Pred::Unknown, false);
        };
        if instr.predicate.is_some() {
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
        let p = match (self.module.interner.resolve(instr.mnemonic), second) {
            ("setp", true) => !p,
            ("elect", false) => Pred::Unknown,
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
        match (
            self.module.interner.resolve(instr.mnemonic),
            mods.as_slice(),
            ops,
        ) {
            ("setp", _, _) => (
                condition(self.module, self.kernel, self.tracer, def)
                    .map_or(Pred::Unknown, |(form, cmp)| {
                        Pred::Cmp(Constraint { form, cmp })
                    }),
                false,
            ),
            ("and" | "or", ["pred"], [_, a, b]) => {
                let (Some(a), Some(b)) = (reg(*a), reg(*b)) else {
                    return (Pred::Unknown, false);
                };
                let ((a, ca), (b, cb)) = (self.read(a, def, depth), self.read(b, def, depth));
                let p = if self.module.interner.resolve(instr.mnemonic) == "and" {
                    a.and(b)
                } else {
                    a.or(b)
                };
                (p, ca || cb)
            }
            ("not", ["pred"], [_, a]) => match reg(*a) {
                Some(a) => {
                    let (p, cut) = self.read(a, def, depth);
                    (!p, cut)
                }
                None => (Pred::Unknown, false),
            },
            ("elect", _, _) => {
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
}

pub(crate) fn thread_sets(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    tracer: &AffineValueTracer,
) -> ThreadSets {
    let sym_bra = module.interner.get("bra");
    let mut reader = PredicateReader {
        module,
        kernel,
        cfg,
        forest,
        tracer,
        memo: HashMap::new(),
        blocks: vec![None; cfg.blocks.len()],
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
        if Some(instr.mnemonic) == sym_bra {
            continue;
        }
        let p = reader.of(guard.reg, pos, 0);
        guards.insert(pos, if guard.negated { !p } else { p });
    }
    ThreadSets { blocks, guards }
}

/// The `setp` at `def` as `a − b cmp 0` over the thread index, when it
/// compares two such forms.
fn condition(
    module: &Module,
    kernel: &Kernel,
    tracer: &AffineValueTracer,
    def: usize,
) -> Option<(Affine, String)> {
    let Stmt::Instr(setp) = &kernel.stmts[def] else {
        return None;
    };
    let mods: Vec<&str> = module
        .modifiers(setp)
        .iter()
        .map(|&m| module.interner.resolve(m))
        .collect();
    let cmp = mods.first()?.to_string();
    if !matches!(cmp.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne")
        || mods
            .iter()
            .any(|m| *m == "and" || *m == "or" || *m == "xor")
    {
        return None;
    }
    let ops = module.operand_ids(setp.operands);
    if ops.len() != 3 {
        return None;
    }
    let a = tracer.trace_operand(ops[1], def, None, 0).ok()?;
    let b = tracer.trace_operand(ops[2], def, None, 0).ok()?;
    let form = a - b;
    on_lane_only(&form).then_some((form, cmp))
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
        }
    }

    #[test]
    fn a_constraint_counts_and_renders() {
        let c = warps_below_4();
        assert_eq!(c.render(), "⌊%tid.x/32⌋ < 4");
        let set = ThreadSet { pred: Pred::Cmp(c) };
        assert!(set.exact());
        assert_eq!(set.count([256, 1, 1]), Some(exact(128)));
        assert_eq!(set.count([64, 1, 1]), Some(exact(64)));
        assert_eq!(ThreadSet { pred: Pred::True }.count([256, 1, 1]), None);
        let lane0 = ThreadSet {
            pred: Pred::Cmp(Constraint {
                form: Affine::var(Var::Tid(Axis::X)),
                cmp: "eq".to_owned(),
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
        });
        let of_upper = ThreadSet {
            pred: Pred::Elect(Box::new(upper.clone())),
        };
        assert_eq!(of_upper.count([128, 1, 1]), Some(exact(3)));
        let warp0 = Pred::Cmp(Constraint {
            form: Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 32)),
            cmp: "eq".to_owned(),
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
        assert_eq!(hedged.count([128, 1, 1]), Some(exact(4)));
    }
}
