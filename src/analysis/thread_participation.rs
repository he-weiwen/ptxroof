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
use crate::ptx::ir::{Kernel, Module, Stmt};

/// One condition `form cmp 0` over the thread index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub form: Affine,
    pub cmp: String,
}

/// The condition under which a thread executes a block: the selecting
/// branches on its dominator chain, each a thread-index comparison
/// when the tracer reads it and `Unknown` when it does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pred {
    True,
    Unknown,
    Cmp(Constraint),
    And(Box<Pred>, Box<Pred>),
}

impl Pred {
    pub fn and(self, other: Pred) -> Pred {
        match (self, other) {
            (Pred::True, p) | (p, Pred::True) => p,
            (Pred::Unknown, Pred::Unknown) => Pred::Unknown,
            (a, b) => Pred::And(Box::new(a), Box::new(b)),
        }
    }

    fn may_hold(&self, tid: [i64; 3]) -> bool {
        match self {
            Pred::True | Pred::Unknown => true,
            Pred::Cmp(c) => holds(&c.cmp, eval_lane(&c.form, tid)),
            Pred::And(a, b) => a.may_hold(tid) && b.may_hold(tid),
        }
    }

    fn exact(&self) -> bool {
        match self {
            Pred::True | Pred::Cmp(_) => true,
            Pred::Unknown => false,
            Pred::And(a, b) => a.exact() && b.exact(),
        }
    }

    fn constraints<'a>(&'a self, out: &mut Vec<&'a Constraint>) {
        match self {
            Pred::True | Pred::Unknown => {}
            Pred::Cmp(c) => out.push(c),
            Pred::And(a, b) => {
                a.constraints(out);
                b.constraints(out);
            }
        }
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
    pub fn contains(&self, tid: [i64; 3]) -> bool {
        self.pred.may_hold(tid)
    }

    /// Every selecting branch is a thread-index condition.
    pub fn exact(&self) -> bool {
        self.pred.exact()
    }

    /// The thread-index conditions as text, `⌊%tid.x/32⌋ < 4 and
    /// %tid.x == 0`; `None` when no such branch selects.
    pub fn render(&self) -> Option<String> {
        let mut cs = Vec::new();
        self.pred.constraints(&mut cs);
        if cs.is_empty() {
            return None;
        }
        Some(
            cs.iter()
                .map(|c| c.render())
                .collect::<Vec<_>>()
                .join(" and "),
        )
    }

    /// How many threads of a block of this shape may be in the set,
    /// when a thread-index branch selects them.
    pub fn count(&self, shape: [u32; 3]) -> Option<u32> {
        self.render()?;
        let [nx, ny, nz] = shape.map(i64::from);
        if nx * ny * nz == 0 {
            return None;
        }
        let mut n = 0;
        for t in 0..nx * ny * nz {
            let tid = [t % nx, (t / nx) % ny, t / (nx * ny)];
            if self.contains(tid) {
                n += 1;
            }
        }
        Some(n)
    }
}

/// The thread set of every block.
pub(crate) fn block_thread_sets(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    tracer: &AffineValueTracer,
) -> Vec<ThreadSet> {
    let sym = |text: &str| module.interner.get(text);
    let sym_bra = sym("bra");
    let sym_setp = sym("setp");
    (0..cfg.blocks.len() as u32)
        .map(BlockId)
        .map(|b| {
            let mut pred = Pred::True;
            let mut cur = forest.doms.idom[b.0 as usize];
            while let Some(d) = cur {
                let blk = cfg.block(d);
                // The dominator's conditional branch, if it selects for b.
                let last = kernel.stmts[blk.start..blk.end]
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
                    let via = |s: BlockId| s == b || tracer.path_avoiding(s, b, d);
                    let selected = match (via(taken), via(fallthrough)) {
                        (true, false) => Some(true),
                        (false, true) => Some(false),
                        _ => None,
                    };
                    if let Some(taken_edge) = selected {
                        let value = taken_edge != guard.negated;
                        pred = pred.and(
                            match condition(module, kernel, tracer, pos, guard.reg, sym_setp) {
                                Some((form, cmp)) => Pred::Cmp(Constraint {
                                    form,
                                    cmp: if value { cmp } else { negate(&cmp).to_owned() },
                                }),
                                None => Pred::Unknown,
                            },
                        );
                    }
                }
                if d == ControlFlowGraph::ENTRY {
                    break;
                }
                cur = forest.doms.idom[d.0 as usize];
            }
            ThreadSet { pred }
        })
        .collect()
}

/// The predicate at `pos` as `a − b cmp 0` over the thread index, when
/// its setp compares two such forms.
fn condition(
    module: &Module,
    kernel: &Kernel,
    tracer: &AffineValueTracer,
    pos: usize,
    pred: crate::support::intern::Symbol,
    sym_setp: Option<crate::support::intern::Symbol>,
) -> Option<(Affine, String)> {
    let ReachingDefinition::Def(def) = tracer.reach_def(pred, pos, None) else {
        return None;
    };
    let Stmt::Instr(setp) = &kernel.stmts[def] else {
        return None;
    };
    if Some(setp.mnemonic) != sym_setp {
        return None;
    }
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
        assert_eq!(set.count([256, 1, 1]), Some(128));
        assert_eq!(set.count([64, 1, 1]), Some(64));
        assert_eq!(ThreadSet { pred: Pred::True }.count([256, 1, 1]), None);
        let lane0 = ThreadSet {
            pred: Pred::Cmp(Constraint {
                form: Affine::var(Var::Tid(Axis::X)),
                cmp: "eq".to_owned(),
            }),
        };
        assert_eq!(lane0.count([128, 2, 1]), Some(2));
    }

    #[test]
    fn an_unknown_selector_keeps_the_index_conditions_as_a_bound() {
        let set = ThreadSet {
            pred: Pred::Cmp(warps_below_4()).and(Pred::Unknown),
        };
        assert!(!set.exact());
        assert_eq!(set.render().as_deref(), Some("⌊%tid.x/32⌋ < 4"));
        assert_eq!(set.count([256, 1, 1]), Some(128));
        assert!(set.contains([0, 0, 0]));
        assert!(!set.contains([200, 0, 0]));
        let unknown = ThreadSet {
            pred: Pred::True.and(Pred::Unknown),
        };
        assert_eq!(unknown.pred, Pred::Unknown);
        assert!(unknown.contains([5, 0, 0]));
        assert_eq!(unknown.render(), None);
        assert_eq!(unknown.count([256, 1, 1]), None);
    }
}
