//! Which threads of a CTA execute a block. A conditional branch on the
//! block's dominator chain fixes its predicate on entry when the block
//! is reachable from only one of the two successors without passing
//! the branch again (its last evaluation decided); when each such
//! predicate is a comparison of affine forms over the thread index
//! alone, the block runs exactly on the threads satisfying all of them,
//! counted by enumeration over the block shape. A predicate that
//! involves anything else (a CTA index, a parameter, a loaded value)
//! leaves the set unknown, and the block's counts stay bounds.

use crate::analysis::control_flow::loops::LoopForest;
use crate::analysis::control_flow::{BlockId, Cfg};
use crate::analysis::scalar::affine::{Affine, Var};
use crate::analysis::scalar::lane_eval::eval_lane;
use crate::analysis::scalar::trace::{Reach, Tracer};
use crate::ptx::ir::{Kernel, Module, Stmt};

/// One condition `form cmp 0` over the thread index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub form: Affine,
    pub cmp: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadSet {
    /// No branch on the dominator chain selects threads.
    All,
    /// The threads satisfying every constraint: exactly those when every
    /// selecting branch was a thread-index condition, at most those
    /// when some other branch (a bounds check, a loaded flag) also
    /// selects.
    Some {
        constraints: Vec<Constraint>,
        exact: bool,
    },
    /// Only branches that are not thread-index conditions select.
    Unknown,
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
    /// Whether the thread at `tid` is in the set; an unknown set
    /// includes every thread, as a bound.
    pub fn contains(&self, tid: [i64; 3]) -> bool {
        match self {
            ThreadSet::All | ThreadSet::Unknown => true,
            ThreadSet::Some { constraints, .. } => constraints
                .iter()
                .all(|c| holds(&c.cmp, eval_lane(&c.form, tid))),
        }
    }

    /// How many threads of a block of this shape satisfy the set.
    pub fn count(&self, shape: [u32; 3]) -> Option<u32> {
        let ThreadSet::Some { constraints, .. } = self else {
            return None;
        };
        let [nx, ny, nz] = shape.map(i64::from);
        if nx * ny * nz == 0 {
            return None;
        }
        let mut n = 0;
        for t in 0..nx * ny * nz {
            let tid = [t % nx, (t / nx) % ny, t / (nx * ny)];
            if constraints
                .iter()
                .all(|c| holds(&c.cmp, eval_lane(&c.form, tid)))
            {
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
    cfg: &Cfg,
    forest: &LoopForest,
    tracer: &Tracer,
) -> Vec<ThreadSet> {
    let sym = |text: &str| module.interner.get(text);
    let sym_bra = sym("bra");
    let sym_setp = sym("setp");
    (0..cfg.blocks.len() as u32)
        .map(BlockId)
        .map(|b| {
            let mut constraints = Vec::new();
            let mut exact = true;
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
                    && let Some(pred) = instr.predicate
                    && let [taken, fallthrough] = blk.succs[..]
                {
                    let via = |s: BlockId| s == b || tracer.path_avoiding(s, b, d);
                    let selected = match (via(taken), via(fallthrough)) {
                        (true, false) => Some(true),
                        (false, true) => Some(false),
                        _ => None,
                    };
                    if let Some(taken_edge) = selected {
                        let value = taken_edge != pred.negated;
                        match condition(module, kernel, tracer, pos, pred.reg, sym_setp) {
                            Some((form, cmp)) => constraints.push(Constraint {
                                form,
                                cmp: if value { cmp } else { negate(&cmp).to_owned() },
                            }),
                            None => exact = false,
                        }
                    }
                }
                if d == Cfg::ENTRY {
                    break;
                }
                cur = forest.doms.idom[d.0 as usize];
            }
            match (constraints.is_empty(), exact) {
                (true, true) => ThreadSet::All,
                (true, false) => ThreadSet::Unknown,
                (false, exact) => ThreadSet::Some { constraints, exact },
            }
        })
        .collect()
}

/// The predicate at `pos` as `a − b cmp 0` over the thread index, when
/// its setp compares two such forms.
fn condition(
    module: &Module,
    kernel: &Kernel,
    tracer: &Tracer,
    pos: usize,
    pred: crate::support::intern::Symbol,
    sym_setp: Option<crate::support::intern::Symbol>,
) -> Option<(Affine, String)> {
    let Reach::Def(def) = tracer.reach_def(pred, pos, None) else {
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

    #[test]
    fn a_constraint_counts_and_renders() {
        let warp = Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 32));
        let c = Constraint {
            form: warp + Affine::invariant(SymExpr::Const(-4)),
            cmp: "lt".to_owned(),
        };
        assert_eq!(c.render(), "⌊%tid.x/32⌋ < 4");
        let set = ThreadSet::Some {
            constraints: vec![c],
            exact: true,
        };
        assert_eq!(set.count([256, 1, 1]), Some(128));
        assert_eq!(set.count([64, 1, 1]), Some(64));
        assert_eq!(ThreadSet::All.count([256, 1, 1]), None);
        let lane0 = ThreadSet::Some {
            constraints: vec![Constraint {
                form: Affine::var(Var::Tid(Axis::X)),
                cmp: "eq".to_owned(),
            }],
            exact: true,
        };
        assert_eq!(lane0.count([128, 2, 1]), Some(2));
    }
}
