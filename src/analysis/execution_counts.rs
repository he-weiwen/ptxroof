//! How many threads and warps run a block or a statement, and how many
//! times: per thread, per CTA or over the launch, exact or a bound.

use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::trip_counts::TripCountResults;
use crate::analysis::control_flow::dominators;
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier};
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::thread_participation::{CTA_BUDGET, Pred, ThreadSets};
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Kernel, SpecialReg, SpecialVector, Stmt};
use std::collections::HashMap;

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
    fn cta(self) -> Option<&'a CtaSize> {
        match self {
            Per::Thread => None,
            Per::Cta(c) | Per::Launch(c, _) => Some(c),
        }
    }

    fn ctas(self) -> Option<u64> {
        match self {
            Per::Launch(_, n) => n,
            _ => None,
        }
    }
}

/// What a count over a block multiplies by (`ExecutionCounts::multiplier`).
pub(crate) struct Multiplier<'a> {
    per: Per<'a>,
    below: Option<LoopId>,
    at_most: bool,
    /// Over the launch, a loop around the block has no trip count.
    pub(crate) unread: bool,
    split: Option<Vec<LoopId>>,
    /// `None` where the caller counts none: aggregates under `unread`.
    pub(crate) trips: Option<SymExpr>,
    ctas: Option<u64>,
}

/// The threads and warps that run a set (`ExecutionCounts::running`),
/// whether they are a bound, whether the count explains the set's
/// threads, and the trips it runs.
#[derive(Clone)]
pub(crate) struct Participation {
    pub(crate) threads: u64,
    pub(crate) warps: Option<(u64, bool)>,
    pub(crate) at_most: bool,
    explained: bool,
    pub(crate) trips: Option<SymExpr>,
}

/// What a statement's memory operand, or the statement, contributes
/// (`ExecutionCounts::statement`).
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

pub(crate) struct ExecutionCounts<'a> {
    kernel: &'a Kernel,
    forest: &'a LoopForest,
    pub(crate) trip_info: TripCountResults,
    bind_map: &'a HashMap<SymbolicConstant, i64>,
    /// Trip expression per loop for aggregation (an opaque symbol when
    /// unresolved, one per loop name), already bound.
    pub(crate) trip_exprs: Vec<SymExpr>,
    /// Loop is conditionally entered within its parent scope.
    cond_entry: Vec<bool>,
    /// Which threads run each block and each guarded instruction.
    pub(crate) sets: ThreadSets,
    /// Per statement: it runs, but nothing with an effect reads its results.
    pub(crate) unused: Vec<bool>,
    /// The block shape when known (`--launch` or `.reqntid`), else zeros.
    pub(crate) shape: [u32; 3],
    /// With the grid bound, the threads and warps over the launch of each
    /// block's set (`None`) and of each guarded statement's, where
    /// `Pred::launch` counts them.
    pub(crate) launches: HashMap<(BlockId, Option<usize>), (u64, u64)>,
    /// The same for each `cp.async`'s source read, by statement.
    source_launches: HashMap<usize, (u64, u64)>,
}

impl<'a> ExecutionCounts<'a> {
    pub(crate) fn new(
        data: &KernelAnalyses<'a>,
        bind_map: &'a HashMap<SymbolicConstant, i64>,
        trip_info: TripCountResults,
        sets: ThreadSets,
        unused: Vec<bool>,
        blocks: &mut [BlockMeasurements],
    ) -> Self {
        let KernelAnalyses {
            kernel,
            cfg,
            forest,
            ..
        } = *data;
        let trip_exprs: Vec<SymExpr> = trip_info
            .trips
            .iter()
            .enumerate()
            .map(|(i, t)| match t {
                Ok(e) => e.bind(bind_map),
                Err(_) => SymExpr::Sym(SymbolicConstant::TripCount(LoopId(i as u32))),
            })
            .collect();

        let exit_blocks: Vec<BlockId> = (0..cfg.blocks.len() as u32)
            .map(BlockId)
            .filter(|&b| cfg.block(b).succs.is_empty())
            .collect();
        let cond_entry: Vec<bool> = (0..forest.loops.len())
            .map(|i| {
                let l = &forest.loops[i];
                let targets: &[BlockId] = match l.parent {
                    Some(p) => &forest.get(p).latches,
                    None => &exit_blocks,
                };
                let entry = trip_info.entries[i];
                targets.is_empty() || !targets.iter().all(|&t| forest.doms.dominates(entry, t))
            })
            .collect();

        let shape = SpecialVector::Ntid.components().map(|reg| {
            bind_map
                .get(&SymbolicConstant::SpecialReg(reg))
                .map(|&v| v as u32)
                .unwrap_or(0)
        });
        let mut b = ExecutionCounts {
            kernel,
            forest,
            trip_info,
            bind_map,
            trip_exprs,
            cond_entry,
            sets,
            unused,
            shape,
            launches: HashMap::new(),
            source_launches: HashMap::new(),
        };
        if let Some(grid) = b.grid() {
            for (block, blk) in cfg.blocks.iter_enumerated() {
                let guarded = (blk.start..blk.end).filter(|s| b.sets.guards.contains_key(s));
                for stmt in [None].into_iter().chain(guarded.map(Some)) {
                    let set = match stmt {
                        Some(s) => b.sets.of(block, s),
                        None => b.sets.blocks[block.0 as usize].clone(),
                    };
                    if let Some(n) = set.launch(shape, grid) {
                        b.launches.insert((block, stmt), n);
                    }
                }
            }
            for &s in b.sets.sources.keys() {
                let set = b.sets.of_access(data.block_of(s), s, 1);
                if let Some(n) = set.launch(shape, grid) {
                    b.source_launches.insert(s, n);
                }
            }
        }
        let doms = dominators(&cfg.without(&b.sets.dead_edges));
        let always = |x: BlockId, targets: &[BlockId]| {
            let mut live = (targets.iter())
                .filter(|&&t| doms.is_reachable(t))
                .peekable();
            live.peek().is_some() && live.all(|&t| doms.dominates(x, t))
        };
        for bm in blocks {
            let x = bm.block;
            let targets = match b.forest.block_loop[x.0 as usize] {
                Some(l) => &b.forest.get(l).latches,
                None => &exit_blocks,
            };
            if always(x, targets) {
                bm.qualifier = CountQualifier::Exact;
            }
        }
        for (i, l) in b.forest.loops.iter().enumerate() {
            let targets = match l.parent {
                Some(p) => &b.forest.get(p).latches,
                None => &exit_blocks,
            };
            b.cond_entry[i] &= !always(b.trip_info.entries[i], targets);
        }
        b
    }

    /// Loop chain of a block, innermost first.
    /// Whether statement `stmt` of block `b` counts: some thread runs it,
    /// and something with an effect reads its results.
    pub(crate) fn counted(&self, b: BlockId, stmt: usize) -> bool {
        !self.sets.never(b, stmt) && !self.unused[stmt]
    }

    pub(crate) fn chain(&self, b: BlockId) -> Vec<LoopId> {
        let mut out = Vec::new();
        let mut cur = self.forest.block_loop[b.0 as usize];
        while let Some(l) = cur {
            out.push(l);
            cur = self.forest.get(l).parent;
        }
        out
    }

    /// Where comparisons with the counter of loop `l`, one of `b`'s loops
    /// up to `below` with a constant trip count, select the set's
    /// threads, which do not vary by CTA, and the loops inside `l` are
    /// read: its threads and warps per CTA summed over `l`'s trips, and
    /// the product of the other loops' counts.
    fn per_trip(
        &self,
        b: BlockId,
        set: &Pred,
        below: Option<LoopId>,
    ) -> Option<(u64, u64, SymExpr)> {
        let l = set.trip_loop()?;
        let chain = self.chain(b);
        let cut = match below {
            Some(m) => chain.iter().position(|&x| x == m)?,
            None => chain.len(),
        };
        let inner = chain[..cut].iter().position(|&x| x == l)?;
        let n = u64::try_from(self.trip_exprs[l.0 as usize].as_const()?).ok()?;
        if set.varies_by_cta()
            || chain[..inner]
                .iter()
                .any(|i| self.trip_info.trips[i.0 as usize].is_err())
        {
            return None;
        }
        let [(t, t_most), (w, w_most)] = set.over_trips(self.shape, l, n)?;
        if t != t_most || w != w_most {
            return None;
        }
        let rest = (chain[..cut].iter())
            .filter(|&&x| x != l)
            .fold(SymExpr::Const(1), |p, x| {
                SymExpr::mul(p, self.trip_exprs[x.0 as usize].clone())
            });
        Some((t, w, rest))
    }

    /// A loop around `b` counts trips that read the CTA index.
    fn per_cta_trips(&self, b: BlockId) -> bool {
        (self.chain(b).iter()).any(|l| self.trips_per_cta(*l))
    }

    /// Loop `l`'s count reads the CTA index, or a uniform value, one per
    /// CTA.
    fn trips_per_cta(&self, l: LoopId) -> bool {
        (self.trip_exprs[l.0 as usize].symbols().iter()).any(|s| {
            matches!(
                s,
                SymbolicConstant::SpecialReg(SpecialReg::Vector(SpecialVector::Ctaid, _))
                    | SymbolicConstant::Unknown(_)
            )
        })
    }

    /// `b`'s loops whose trip counts read the CTA index, and the product
    /// of the others.
    fn cta_split(&self, b: BlockId) -> (Vec<LoopId>, SymExpr) {
        let (cta, rest): (Vec<LoopId>, Vec<LoopId>) =
            (self.chain(b).into_iter()).partition(|&l| self.trips_per_cta(l));
        let rest = (rest.iter()).fold(SymExpr::Const(1), |p, l| {
            SymExpr::mul(p, self.trip_exprs[l.0 as usize].clone())
        });
        (cta, rest)
    }

    /// The product of the loops' counts as the axes they read and its
    /// value at each CTA: each count evaluated and never negative, and a
    /// `!=` latch's step dividing its distance into a positive count.
    fn cta_weight(&self, loops: Vec<LoopId>) -> ([bool; 3], impl Fn([i64; 3]) -> Option<u64> + '_) {
        let regs = SpecialVector::Ctaid.components();
        let read = regs.map(|reg| {
            let leaf = SymbolicConstant::SpecialReg(reg);
            (loops.iter()).any(|l| self.trip_exprs[l.0 as usize].symbols().contains(&leaf))
        });
        let weight = move |c: [i64; 3]| {
            let at: HashMap<SymbolicConstant, i64> = regs
                .map(SymbolicConstant::SpecialReg)
                .into_iter()
                .zip(c)
                .collect();
            loops.iter().try_fold(1u64, |w, l| {
                let n = self.trip_exprs[l.0 as usize].bind(&at).as_const()?;
                if let Some((x, step)) = &self.trip_info.distances[l.0 as usize] {
                    let x = x.bind(self.bind_map).bind(&at).as_const()?;
                    if x.checked_rem(*step)? != 0 || x.checked_div(*step)? < 1 {
                        return None;
                    }
                }
                w.checked_mul(u64::try_from(n).ok()?)
            })
        };
        (read, weight)
    }

    /// The weight summed over the grid; `None` without the grid or past
    /// `CTA_BUDGET` CTAs on the axes it reads.
    fn cta_sum(&self, loops: &[LoopId]) -> Option<u64> {
        let grid = self.grid()?;
        let (read, weight) = self.cta_weight(loops.to_vec());
        let n = |i: usize| if read[i] { grid[i] } else { 1 };
        if (0..3).map(|i| u64::from(n(i))).product::<u64>() > CTA_BUDGET {
            return None;
        }
        let mut sum = 0u64;
        for c in
            (0..n(2)).flat_map(|z| (0..n(1)).flat_map(move |y| (0..n(0)).map(move |x| [x, y, z])))
        {
            sum = sum.checked_add(weight(c.map(i64::from))?)?;
        }
        let other = (0..3).filter(|&i| !read[i]).map(|i| u64::from(grid[i]));
        { other }.try_fold(sum, u64::checked_mul)
    }

    /// The set's threads and warps over the launch, CTA `c` counted the
    /// weight's value there times.
    fn weighted_launch(&self, set: &Pred, loops: &[LoopId]) -> Option<(u64, u64)> {
        let (read, weight) = self.cta_weight(loops.to_vec());
        set.weighted_launch(self.shape, self.grid()?, read, &weight)
    }

    /// The product of `b`'s loops' trip counts up to `below`, and whether
    /// a loop among them may not be entered: one entered conditionally,
    /// unless its header's threads are counted, per CTA or at every CTA
    /// of the launch. Over the launch, the loops whose counts read the
    /// CTA index are `split` out and summed over the grid as `ctas`.
    pub(crate) fn multiplier<'c>(
        &self,
        b: BlockId,
        below: Option<LoopId>,
        per: Per<'c>,
    ) -> Option<Multiplier<'c>> {
        let chain = self.chain(b);
        let cut = match below {
            Some(l) => chain.iter().position(|&x| x == l)?,
            None => chain.len(),
        };
        let mut mult = SymExpr::Const(1);
        let mut at_most = false;
        for &l in &chain[..cut] {
            mult = SymExpr::mul(mult, self.trip_exprs[l.0 as usize].clone());
            let header = self.forest.get(l).header;
            let counted = (self.sets.blocks[header.0 as usize].count(self.shape))
                .is_some_and(|c| c.certain == c.possible);
            let entered = (per.cta().is_some() && counted)
                || (matches!(per, Per::Launch(..)) && self.launches.contains_key(&(header, None)));
            at_most |= self.cond_entry[l.0 as usize] && !entered;
        }
        let launch = matches!(per, Per::Launch(..));
        let unread = launch && (chain.iter()).any(|l| self.trip_info.trips[l.0 as usize].is_err());
        let split = (launch && self.per_cta_trips(b)).then(|| self.cta_split(b));
        let ctas = split.as_ref().map_or(per.ctas(), |(p, _)| self.cta_sum(p));
        let (split, trips) = split.map_or((None, mult), |(p, rest)| (Some(p), rest));
        Some(Multiplier {
            per,
            below,
            at_most,
            unread,
            split,
            trips: Some(trips),
            ctas,
        })
    }

    /// The threads and warps that run `set` in block `bm` as `m` sums
    /// them: trip by trip where a loop counter selects them, weighted by
    /// the CTA index where `m`'s loop counts read it, at the launch's own
    /// count (`launch`), else per CTA times the CTAs; with the trips they
    /// run, `m`'s unless they are summed trip by trip. `None` when a
    /// count is unknown or does not fit in u64.
    pub(crate) fn running(
        &self,
        bm: &BlockMeasurements,
        m: &Multiplier,
        set: &Pred,
        launch: Option<(u64, u64)>,
    ) -> Option<Participation> {
        let bound = m.at_most || m.per.cta().is_some_and(|c| !c.exact);
        let counted = |threads, warps, trips| Participation {
            threads,
            warps: Some((warps, bound)),
            at_most: bound,
            explained: true,
            trips,
        };
        if m.per.cta().is_some()
            && m.split.is_none()
            && m.trips.is_some()
            && let Some((t, w, rest)) = self.per_trip(bm.block, set, m.below)
        {
            let n = match m.per {
                Per::Launch(_, n) => n?,
                _ => 1,
            };
            return Some(counted(t.checked_mul(n)?, w.checked_mul(n)?, Some(rest)));
        }
        let (t, w, at_most, explained) = self.executing(bm, set, m.at_most, m.per.cta());
        let scaled = |n: u64| {
            let warps = match w {
                Some((w, bound)) => Some((u64::try_from(w).ok()?.checked_mul(n)?, bound)),
                None => None,
            };
            let threads = u64::try_from(t).ok()?.checked_mul(n)?;
            let trips = m.trips.clone();
            Some(Participation {
                threads,
                warps,
                at_most,
                explained,
                trips,
            })
        };
        let Per::Launch(..) = m.per else {
            return scaled(1);
        };
        let launch = match &m.split {
            Some(p) if set.varies_by_cta() => Some(self.weighted_launch(set, p)?),
            Some(_) => None,
            None => launch,
        };
        match launch {
            Some((t, w)) => Some(counted(t, w, m.trips.clone())),
            None => scaled(m.ctas?),
        }
    }

    /// Memory operand `operand` of statement `stmt` of block `bm`, or the
    /// statement with operand 0, under its guard: none or one every
    /// thread reads as true runs as `block`, a bound if the statement is
    /// predicated (a branch); one every thread reads as false skips it;
    /// another runs its own set, a cp.async source operand the threads
    /// that read it, a bound unless the count explains the set. `cache`
    /// keeps each guarded set's count per block.
    pub(crate) fn statement(
        &self,
        bm: &BlockMeasurements,
        m: &Multiplier,
        stmt: usize,
        operand: usize,
        block: Option<&Participation>,
        cache: &mut HashMap<(usize, bool), Option<Participation>>,
    ) -> Resolution {
        let counted = |p: Option<Participation>| p.map_or(Resolution::Unknown, Resolution::Counted);
        let source = operand == 1 && self.sets.sources.contains_key(&stmt);
        match self.sets.guard(stmt, operand) {
            Some(Pred::False) => Resolution::Skipped,
            Some(Pred::True) => counted(block.cloned()),
            Some(_) => counted(
                (cache.entry((stmt, source)))
                    .or_insert_with(|| {
                        let (set, launch) = match source {
                            true => (
                                self.sets.of_access(bm.block, stmt, 1),
                                self.source_launches.get(&stmt).copied(),
                            ),
                            false => (
                                self.sets.of(bm.block, stmt),
                                self.launches.get(&(bm.block, Some(stmt))).copied(),
                            ),
                        };
                        let p = self.running(bm, m, &set, launch)?;
                        Some(Participation {
                            at_most: p.at_most || !p.explained,
                            ..p
                        })
                    })
                    .clone(),
            ),
            None => {
                let predicated =
                    matches!(&self.kernel.stmts[stmt], Stmt::Instr(i) if i.predicate.is_some());
                counted(block.map(|p| Participation {
                    at_most: p.at_most || predicated,
                    ..p.clone()
                }))
            }
        }
    }

    // A block selected by thread-index branches runs on exactly
    // its threads: per CTA that count, and not a bound.
    fn executing(
        &self,
        bm: &BlockMeasurements,
        set: &Pred,
        chain_at_most: bool,
        cta: Option<&CtaSize>,
    ) -> (i64, Option<(i64, bool)>, bool, bool) {
        let counted = set.count(self.shape);
        let selected = cta.and(counted.map(|c| c.possible));
        let explained = selected.is_some() && counted.is_some_and(|c| c.certain == c.possible);
        let conditional = bm.qualifier == CountQualifier::AtMost && !explained;
        let at_most = conditional || chain_at_most || cta.is_some_and(|c| !c.exact);
        let threads = match (cta, selected) {
            (Some(_), Some(n)) => i64::from(n),
            (Some(c), None) => c.threads as i64,
            (None, _) => 1,
        };
        let warps = cta.map(|c| match set.warps(self.shape) {
            Some(w) => (i64::from(w.possible), at_most || w.certain != w.possible),
            None => (c.warps as i64, at_most),
        });
        (threads, warps, at_most, explained)
    }

    /// The grid, when bound.
    pub(crate) fn grid(&self) -> Option<[u32; 3]> {
        let [x, y, z] = (SpecialVector::Nctaid.components()).map(|reg| {
            self.bind_map
                .get(&SymbolicConstant::SpecialReg(reg))
                .and_then(|&n| u32::try_from(n).ok())
        });
        Some([x?, y?, z?])
    }
}
