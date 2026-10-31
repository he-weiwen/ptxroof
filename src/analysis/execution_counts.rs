//! How many threads and warps run a block or a statement, and how many
//! times: per thread, per CTA or over the launch, exact or a bound.

use crate::analysis::affine::sym_expr::SymExpr;
use crate::analysis::control_flow::loops::LoopId;

pub(crate) struct CtaSize {
    pub(crate) threads: u64,
    pub(crate) warps: u64,
    pub(crate) exact: bool,
}

/// How far a count is summed: over one thread, over one CTA of the
/// launch shape, or over the launch, whose CTA count may not fit in u64.
#[derive(Clone, Copy)]
pub(crate) enum Per<'a> {
    Thread,
    Cta(&'a CtaSize),
    Launch(&'a CtaSize, Option<u64>),
}

impl<'a> Per<'a> {
    pub(crate) fn cta(self) -> Option<&'a CtaSize> {
        match self {
            Per::Thread => None,
            Per::Cta(c) | Per::Launch(c, _) => Some(c),
        }
    }

    pub(crate) fn ctas(self) -> Option<u64> {
        match self {
            Per::Launch(_, n) => n,
            _ => None,
        }
    }
}

/// What a count over a block multiplies by (`KernelReportBuilder::multiplier`).
pub(crate) struct Multiplier<'a> {
    pub(crate) per: Per<'a>,
    pub(crate) below: Option<LoopId>,
    pub(crate) at_most: bool,
    /// Over the launch, a loop around the block has no trip count.
    pub(crate) unread: bool,
    pub(crate) split: Option<Vec<LoopId>>,
    /// `None` where the caller counts none: aggregates under `unread`.
    pub(crate) trips: Option<SymExpr>,
    pub(crate) ctas: Option<u64>,
}

/// The threads and warps that run a set (`KernelReportBuilder::running`),
/// whether they are a bound, whether the count explains the set's
/// threads, and the trips it runs.
#[derive(Clone)]
pub(crate) struct Participation {
    pub(crate) threads: u64,
    pub(crate) warps: Option<(u64, bool)>,
    pub(crate) at_most: bool,
    pub(crate) explained: bool,
    pub(crate) trips: Option<SymExpr>,
}

/// What a statement's memory operand, or the statement, contributes
/// (`KernelReportBuilder::statement`).
pub(crate) enum Resolution {
    Skipped,
    Unknown,
    Counted(Participation),
}

impl Participation {
    /// Its threads and warps as the i64 counts the totals sum, when they fit.
    pub(crate) fn narrow(&self) -> Option<(i64, Option<(i64, bool)>)> {
        let warps = match self.warps {
            Some((w, bound)) => Some((i64::try_from(w).ok()?, bound)),
            None => None,
        };
        Some((i64::try_from(self.threads).ok()?, warps))
    }
}
