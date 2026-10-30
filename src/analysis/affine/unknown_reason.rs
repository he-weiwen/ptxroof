//! Why the affine analysis has no affine expression for a value.

use crate::analysis::affine::affine_expr::Var;
use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::ptx::ir::SpecialReg;

/// Why an SSA value has no affine expression: a fact about its
/// definition, which `render` puts in words for a reader.
#[derive(Clone, PartialEq)]
pub(crate) enum UnknownReason {
    /// A load from memory, of no kernel parameter, at statement `def`.
    Load { def: usize },
    /// An atomic operation.
    Atomic,
    /// A special register that is not one value for the CTA.
    SpecialRegister(SpecialReg),
    /// Register `0`, which a loop header's φ carries from one trip of an
    /// enclosing loop to the next.
    LoopCarried(String),
    /// `reason`, behind an instruction the analysis does not read.
    Behind {
        reason: Box<UnknownReason>,
        opcode: String,
    },
    /// Operation `what` of a form with the variables `vars`, in order.
    NotAffine { what: &'static str, vars: Vec<Var> },
    /// Variable `var`, holding an unknown value, read as one value for
    /// threads that `why` says may not share it; `own` is why affine
    /// analysis has no form for the value.
    NotOneValue {
        var: Box<Var>,
        why: String,
        own: Option<Box<UnknownReason>>,
    },
    /// A sentence that reads the same wherever the value is read.
    Other(String),
}

impl UnknownReason {
    /// The reason as a sentence for a read in loop `scope`, a latch's
    /// condition where `latch`: a load inside or outside that loop, a
    /// variable other than its counter, one holding an unknown value
    /// first, as `name` prints it.
    pub(crate) fn render(
        &self,
        data: &KernelAnalyses,
        scope: Option<LoopId>,
        latch: bool,
        name: &dyn Fn(&Var) -> String,
    ) -> String {
        let prefix = if latch { "latch condition " } else { "" };
        match self {
            UnknownReason::Load { def } => {
                let place = match scope.is_some_and(|l| data.in_loop(l, *def)) {
                    true => "inside the loop",
                    false => "from memory",
                };
                format!("{prefix}depends on a value loaded {place}")
            }
            UnknownReason::Atomic => format!("{prefix}depends on an atomic operation"),
            UnknownReason::SpecialRegister(reg) => {
                format!("{prefix}depends on special register {}", reg.name())
            }
            UnknownReason::LoopCarried(reg) => {
                format!("{prefix}depends on {reg}, carried around an enclosing loop")
            }
            UnknownReason::Behind { reason, opcode } => {
                format!(
                    "{}, behind `{opcode}`",
                    reason.render(data, scope, latch, name)
                )
            }
            UnknownReason::NotAffine { what, vars } => {
                if let Some(u) = vars.iter().find(|v| v.unknown().is_some()) {
                    return format!("{prefix}depends on {}, in a {what}", u.render(&name));
                }
                for v in vars {
                    match v {
                        Var::Iteration(l) if Some(*l) == scope => {}
                        Var::Iteration(_) => {
                            return format!("{prefix}depends on an enclosing loop's counter");
                        }
                        other => {
                            return format!(
                                "{prefix}depends on special register {other}, in a {what}"
                            );
                        }
                    }
                }
                format!("{what} of an induction variable")
            }
            UnknownReason::NotOneValue { var, why, own } => match own {
                Some(own) => format!(
                    "{} {why}: {}",
                    var.render(&name),
                    own.render(data, scope, false, name)
                ),
                None => format!("{} {why}", var.render(&name)),
            },
            UnknownReason::Other(s) => s.clone(),
        }
    }
}

impl From<String> for UnknownReason {
    fn from(s: String) -> Self {
        UnknownReason::Other(s)
    }
}

impl From<&str> for UnknownReason {
    fn from(s: &str) -> Self {
        UnknownReason::Other(s.to_owned())
    }
}
