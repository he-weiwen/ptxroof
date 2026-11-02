//! The input PTX with what the analysis says of each line appended to
//! it as a ` // ` comment.

use crate::analysis::affine::affine_expr::{AffineExpr, Cmp, Var};
use crate::analysis::affine::lane_eval::depends_on_lane;
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::unknown_reason::UnknownReason;
use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::execution_counts::{
    CountAccumulator, CtaSize, Participation, Per, Resolution,
};
use crate::analysis::instruction_counts::collect::BlockMeasurements;
use crate::analysis::instruction_counts::measurement::{MeasureKind, Pipe};
use crate::analysis::memory_footprint::{ByteRange, CountRange};
use crate::analysis::register_effects::writes;
use crate::analysis::thread_participation::{Atom, AtomKey, Constraint, Pred};
use crate::analysis::uniformity_info::Uniform;
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Opcode, Register, Stmt};
use crate::report::build::{AnalyzeError, AnalyzeOptions, KernelReportBuilder, each_kernel};
use crate::report::schema::Access;
use crate::ssa::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Per statement, the parts of its annotation.
type Notes = BTreeMap<usize, Vec<String>>;

/// The lines of the branches deciding a condition's uniform unknowns, and
/// the larger of the threads each is the same for.
type Tags = (BTreeSet<u32>, Uniform);

/// `source` with ` // ` and the annotation after each line that has one,
/// the annotations of its statements joined by ` ; `.
pub fn annotate(source: &str, opts: &AnalyzeOptions) -> Result<String, AnalyzeError> {
    let mut notes: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    each_kernel(source, opts, |builder| builder.annotations(&mut notes))?;
    let mut out = String::with_capacity(source.len());
    for (line, text) in (1..).zip(source.split_inclusive('\n')) {
        let (text, end) = text.strip_suffix('\n').map_or((text, ""), |t| (t, "\n"));
        out.push_str(text);
        if let Some(n) = notes.get(&line) {
            out.push_str(" // ");
            out.push_str(&n.join(" ; "));
        }
        out.push_str(end);
    }
    Ok(out)
}

impl KernelReportBuilder<'_> {
    /// Each block's annotation, a loop header's trip count first, on its
    /// label's line, else on its first instruction's after `[<block
    /// name>] `,
    /// then each statement's on its own line, `unused` alone where nothing
    /// with an effect reads its results. Counts are over the launch given
    /// the grid and the block shape, else per CTA given the shape, else
    /// per thread.
    fn annotations(&self, notes: &mut BTreeMap<u32, Vec<String>>) {
        let threads: u64 = self.counts.shape.iter().map(|&n| u64::from(n)).product();
        let cta = (threads > 0).then(|| CtaSize {
            threads,
            warps: threads.div_ceil(32),
            exact: true,
        });
        let ctas = (self.counts.grid())
            .and_then(|g| g.iter().try_fold(1u64, |n, &d| n.checked_mul(d.into())));
        let per = match (&cta, ctas) {
            (Some(c), Some(n)) => Per::Launch(c, Some(n)),
            (Some(c), None) => Per::Cta(c),
            (None, _) => Per::Thread,
        };
        if let Some(entry) = self.entry(per) {
            let line = self.lines.line(self.kernel.offset);
            notes.entry(line).or_default().push(entry);
        }
        for bm in &self.blocks {
            let b = bm.block;
            let (first, end) = (self.cfg.block(b).start, self.cfg.block(b).end);
            let start = (first..end)
                .find(|&s| matches!(self.kernel.stmts[s], Stmt::Label { .. } | Stmt::Instr(_)))
                .unwrap_or(first);
            let header = self.forest.block_loop[b.0 as usize]
                .filter(|&l| self.forest.get(l).header == b)
                .map(|l| {
                    let bytes = self.loop_global(l).map(|b| format!(" | {b}"));
                    format!("loop: trips {}{}", self.trips(l), bytes.unwrap_or_default())
                });
            let (exec, mut parts) = self.execs(bm, per);
            let block_threads = self.threads(b, &self.counts.sets.blocks[b.0 as usize]);
            for s in bm.instructions.iter().map(|i| i.provenance) {
                if self.counts.unused[s] {
                    parts.insert(s, vec!["unused".to_owned()]);
                } else {
                    let threads = (self.guarded(s))
                        .then(|| self.threads(b, &self.counts.sets.of(b, s)))
                        .flatten()
                        .filter(|t| Some(t) != block_threads.as_ref());
                    let part = parts.entry(s).or_default();
                    part.splice(0..0, threads);
                    part.extend(self.uncounted(bm, s));
                    part.extend(self.operands(s));
                }
            }
            for s in (first..end).filter(|&s| matches!(self.kernel.stmts[s], Stmt::Unparsed { .. }))
            {
                parts.insert(s, vec!["not parsed (not counted)".to_owned()]);
            }
            parts.retain(|_, p| !p.is_empty());
            let when = self.runs_when(b);
            let runs = self.counts.sets.blocks[b.0 as usize] != Pred::False;
            let nesting = runs.then(|| self.nesting(b, when.is_some())).flatten();
            let mut own: Vec<String> = header
                .into_iter()
                .chain(block_threads.clone())
                .chain(when)
                .chain(nesting)
                .chain(exec)
                .collect();
            own.extend(parts.remove(&start).unwrap_or_default());
            let prefix = match self.kernel.stmts[start] {
                Stmt::Label { .. } => String::new(),
                _ => format!("[{}] ", self.cfg.block_name(self.module, b)),
            };
            let block =
                (!own.is_empty()).then(|| (start, vec![format!("{prefix}{}", own.join(" | "))]));
            for (s, parts) in block.into_iter().chain(parts) {
                let Some(line) = self.line(s) else {
                    continue;
                };
                notes.entry(line).or_default().push(parts.join(" | "));
            }
        }
    }

    /// The `.entry` line's annotation: what the counts take of the
    /// launch, those given, `grid X×Y×Z`, `block X×Y×Z` and each
    /// parameter's value; then the shared memory each CTA declares, `+
    /// dynamic` where an `.extern .shared` array's size is set at launch;
    /// then the FLOPs and bytes its statements sum to at the scale `per`.
    fn entry(&self, per: Per) -> Option<String> {
        let dims = |d: [u32; 3]| d.map(|n| n.to_string()).join("×");
        let shape = self.counts.shape;
        let params: BTreeMap<usize, i64> = (self.bind_map.iter())
            .filter_map(|(k, &v)| match k {
                SymbolicConstant::KernelParam(i) => Some((*i, v)),
                _ => None,
            })
            .collect();
        let params: Vec<String> = (params.into_iter())
            .map(|(i, v)| format!("{} = {v}", SymbolicConstant::KernelParam(i)))
            .collect();
        let parts: Vec<String> = [
            self.counts.grid().map(|g| format!("grid {}", dims(g))),
            (!shape.contains(&0)).then(|| format!("block {}", dims(shape))),
            (!params.is_empty()).then(|| params.join(", ")),
            self.shared(),
            self.totals(per),
        ]
        .into_iter()
        .flatten()
        .collect();
        (!parts.is_empty()).then(|| parts.join(" | "))
    }

    /// `shared memory N B/CTA`, its static declarations' bytes, and `+
    /// dynamic` where the launch sets more, unless it declares none.
    fn shared(&self) -> Option<String> {
        let sm = self.shared_memory();
        let bytes = self.count(&SymExpr::Const(i64::try_from(sm.static_bytes).ok()?));
        let dynamic = if sm.dynamic { " + dynamic" } else { "" };
        (sm.static_bytes > 0 || sm.dynamic).then(|| format!("shared memory {bytes} B/CTA{dynamic}"))
    }

    /// `per launch:`, `per CTA:` or `per thread:` and what the counted
    /// statements sum to at that scale, `; `-separated: each pipe's FLOPs
    /// by precision, `+ ?` where some are not counted, the pipes none of
    /// whose are counted together, and each space's bytes loaded and
    /// stored; the zeros left out.
    fn totals(&self, per: Per) -> Option<String> {
        let w = self.counts.workload(&self.blocks, None, per);
        let mut items = Vec::new();
        let mut uncounted = Vec::new();
        for (pipe, flops) in &w.flops {
            let mut parts: Vec<String> = (flops.by_precision.iter())
                .filter(|(_, c)| c.expr() != SymExpr::Const(0))
                .map(|(p, c)| format!("{} {}", p.key(), self.sum(c, "")))
                .collect();
            if parts.is_empty() && flops.total.at_least {
                uncounted.push(pipe.key());
                continue;
            }
            if flops.total.at_least {
                parts.push("?".to_owned());
            }
            let name = match pipe {
                Pipe::CudaCore => String::new(),
                p => format!("{} ", p.key()),
            };
            if !parts.is_empty() {
                items.push(format!("{name}flops {}", parts.join(" + ")));
            }
        }
        match uncounted.len() {
            0 => {}
            n if n == Pipe::ALL.len() => items.insert(0, "flops ?".to_owned()),
            _ => items.push(format!("{} flops ?", uncounted.join(", "))),
        }
        let spaces = [
            "global",
            "shared",
            "shared::cluster",
            "local",
            "param",
            "const",
            "generic",
        ];
        for space in spaces {
            let Some((load, store)) = w.bytes.get(space) else {
                continue;
            };
            let parts: Vec<String> = [("load", load), ("store", store)]
                .into_iter()
                .filter(|(_, c)| c.at_least || c.expr() != SymExpr::Const(0))
                .map(|(what, c)| format!("{what} {}", self.sum(c, " B")))
                .collect();
            if !parts.is_empty() {
                items.push(format!("{space} {}", parts.join(", ")));
            }
        }
        let unit = match per {
            Per::Thread => "thread",
            Per::Cta(_) => "CTA",
            Per::Launch(..) => "launch",
        };
        (!items.is_empty()).then(|| format!("per {unit}: {}", items.join("; ")))
    }

    /// A sum and its `unit`: `<= ` before an upper bound, `>= ` before a
    /// lower one, `?` where it is neither or nothing is counted, and `?`
    /// and `(trips of <header>)` where it reads a loop's unknown trips.
    fn sum(&self, c: &CountAccumulator, unit: &str) -> String {
        let expr = c.expr();
        let unknown = (expr.symbols().into_iter()).find_map(|s| match s {
            SymbolicConstant::TripCount(l) => Some(l),
            _ => None,
        });
        if let Some(l) = unknown {
            let header = self.cfg.block_name(self.module, self.forest.get(l).header);
            return format!("?{unit} (trips of {header})");
        }
        let n = match (c.at_most, c.at_least, expr.as_const()) {
            (true, true, _) | (false, true, Some(0)) => "?".to_owned(),
            (_, true, _) => format!(">= {}", self.count(&expr)),
            (true, false, _) => format!("<= {}", self.count(&expr)),
            (false, false, _) => self.count(&expr),
        };
        format!("{n}{unit}")
    }

    /// The global bytes one CTA requests over one run of loop `l`'s own
    /// blocks, not its nested loops', the distinct bytes they touch and
    /// the bytes of the sectors they touch, where known; `<= ` for a
    /// bound.
    fn loop_global(&self, l: LoopId) -> Option<String> {
        let b = self.loop_bytes(l)?;
        let n = |n: u64| i64::try_from(n).map_or(n.to_string(), |n| self.count(&SymExpr::Const(n)));
        let bound = if b.at_most { "<= " } else { "" };
        let range = |r: ByteRange| match r.min == r.max {
            true => format!("{bound}{} B", n(r.min)),
            false => format!("{bound}{}–{} B", n(r.min), n(r.max)),
        };
        let requested = if b.requested_at_most { "<= " } else { "" };
        let unique = b.unique.map_or("?".to_owned(), range);
        let sectors = b.sector_bytes.map(|s| format!(", sectors {}", range(s)));
        Some(format!(
            "global per CTA: requested {requested}{} B, unique {unique}{}",
            n(b.requested),
            sectors.unwrap_or_default()
        ))
    }

    /// The line of statement `s`, but a `.branchtargets` list.
    fn line(&self, s: usize) -> Option<u32> {
        match &self.kernel.stmts[s] {
            Stmt::Label { offset, .. } | Stmt::Unparsed { offset, .. } => {
                Some(self.lines.line(*offset))
            }
            Stmt::Instr(i) => Some(self.lines.line(i.offset)),
            Stmt::BranchTargets(_) => None,
        }
    }

    /// `exec` and how many times block `bm`'s unguarded statements run at
    /// the scale `per`, `<=` for a bound, unless no thread runs it or per
    /// thread they run once; and
    /// each statement whose count differs, but a predicated `bra`. Under a
    /// loop whose trips are not known, `exec ?` and the innermost such loop.
    fn execs(&self, bm: &BlockMeasurements, per: Per) -> (Option<String>, Notes) {
        let b = bm.block;
        let set = &self.counts.sets.blocks[b.0 as usize];
        let unknown = (self.counts.chain(b).into_iter())
            .find(|l| self.counts.trip_info.trips[l.0 as usize].is_err());
        let (Some(m), None, false) = (
            self.counts.multiplier(b, None, per),
            unknown,
            *set == Pred::False,
        ) else {
            let header = unknown
                .map(|l| self.forest.get(l).header)
                .filter(|&h| h != b);
            let exec = header.map_or("exec ?".to_owned(), |h| {
                format!("exec ? (trips of {})", self.cfg.block_name(self.module, h))
            });
            return ((*set != Pred::False).then_some(exec), Notes::new());
        };
        let unit = match per {
            Per::Thread => "thread",
            Per::Cta(_) => "CTA",
            Per::Launch(..) => "launch",
        };
        let text = |p: Option<Participation>| {
            let n = p.as_ref().and_then(|p| {
                Some(SymExpr::mul(
                    SymExpr::Const(i64::try_from(p.threads).ok()?),
                    p.trips.clone()?,
                ))
            });
            let bound = if p.is_some_and(|p| p.at_most) {
                "<= "
            } else {
                ""
            };
            n.map_or("exec ?".to_owned(), |n| {
                format!("exec {bound}{}/{unit}", self.count(&n))
            })
        };
        let launch = self.counts.launches.get(&(b, None)).copied();
        let block = self.counts.running(bm, &m, set, launch);
        let own = text(block.clone());
        let mut cache = HashMap::new();
        let mut parts = Notes::new();
        for i in &bm.instructions {
            let exec =
                match self
                    .counts
                    .statement(bm, &m, i.provenance, 0, block.as_ref(), &mut cache)
                {
                    Resolution::Skipped => format!("exec 0/{unit}"),
                    Resolution::Unknown => text(None),
                    Resolution::Counted(p) => text(Some(p)),
                };
            let Stmt::Instr(instr) = &self.kernel.stmts[i.provenance] else {
                continue;
            };
            if exec != own && !(instr.mnemonic == Opcode::Bra && instr.predicate.is_some()) {
                parts.insert(i.provenance, vec![exec]);
            }
        }
        let once = matches!(per, Per::Thread)
            && block.is_some_and(|p| {
                p.threads == 1 && !p.at_most && p.trips.and_then(|t| t.as_const()) == Some(1)
            });
        ((!once).then_some(own), parts)
    }

    /// Statement `s` has a guard of its own, but a predicated `bra`, whose
    /// guard picks a way out of the block.
    fn guarded(&self, s: usize) -> bool {
        let Stmt::Instr(instr) = &self.kernel.stmts[s] else {
            return false;
        };
        self.counts.sets.guards.contains_key(&s) && instr.mnemonic != Opcode::Bra
    }

    /// What counted statement `s` of block `bm` leaves out of the totals:
    /// its work where it is not classified, its bytes where an access's
    /// size is not known, a call's callee.
    fn uncounted(&self, bm: &BlockMeasurements, s: usize) -> Vec<String> {
        let kinds: Vec<MeasureKind> = (bm.measurements.iter())
            .filter(|m| m.provenance == s && self.counts.counted(bm.block, s))
            .map(|m| m.kind)
            .collect();
        let has = |f: fn(&MeasureKind) -> bool| kinds.iter().any(f);
        [
            (has(|k| matches!(k, MeasureKind::UnknownOps { .. })))
                .then_some("not classified (its FLOPs and bytes not counted)"),
            (has(|k| matches!(k, MeasureKind::UnquantifiedBytes { .. })))
                .then_some("bytes ? (not counted)"),
            (self.cfg.call_sites.contains(&s)).then_some("call (its callee not counted)"),
        ]
        .into_iter()
        .flatten()
        .map(str::to_owned)
        .collect()
    }

    /// Statement `s`'s memory operands but parameters, each `<space>
    /// <direction> [<address>]`, or `addr ?` and why the address is not
    /// known, then a global one's sectors per warp request and the fewest
    /// they need, a shared one's wavefronts, `<= ` for a bound; a product
    /// as `a·b`.
    fn operands(&self, s: usize) -> Vec<String> {
        let range = |r: &CountRange| match r.min == r.max {
            true => r.min.to_string(),
            false => format!("{}–{}", r.min, r.max),
        };
        let scope = self.forest.block_loop[self.data.block_of(s).0 as usize];
        let rows = self.operands.get(&s).map_or(&[][..], Vec::as_slice);
        let row = |(row, traced): &(Access, Option<Result<AffineExpr, UnknownReason>>)| {
            let address = match (traced, &row.address) {
                (Some(Ok(a)), _) => format!("[{}]", self.form(a)),
                (Some(Err(why)), _) => {
                    let why = why.render(self.data, scope, false, &|v| self.var(v, true));
                    format!("addr ? ({why})")
                }
                (None, Some(a)) => format!("[{a}]"),
                (None, None) => format!("addr ? ({})", row.unknown.as_deref().unwrap_or_default()),
            };
            let (what, per_request) = match (row.space.as_str(), &row.sectors_per_request) {
                ("shared", _) => ("wavefronts", &row.wavefronts_per_request),
                (_, n) => ("sectors", n),
            };
            let ideal = match what {
                "wavefronts" => &row.ideal_wavefronts_per_request,
                _ => &row.ideal_sectors_per_request,
            };
            let bound = if row.sectors_at_most { "<= " } else { "" };
            let request = match (per_request, ideal, &row.footprint_unknown) {
                (Some(n), Some(ideal), _) => {
                    format!(
                        " {what}/req {bound}{} (ideal {bound}{})",
                        range(n),
                        range(ideal)
                    )
                }
                (_, _, Some(why)) => format!(" {what}/req ? ({why})"),
                _ => String::new(),
            };
            format!("{} {} {address}{request}", row.space, row.direction).replace(" * ", "·")
        };
        rows.iter().map(row).collect()
    }

    /// An address's form, a loop's iteration k = iv + 1 as `iv` with each
    /// `c·k`'s c in the constant, or as `(iv + 1)` where a floor or a
    /// remainder holds it.
    fn form(&self, a: &AffineExpr) -> String {
        let (shifted, nested) = shifted(a);
        self.shorten(shifted.render(|v| self.var(v, nested)))
    }

    /// A variable: a loop's iteration as `iv[<header label>]`, or for k
    /// itself as `(iv + 1)` where `plus_one`; an unknown value by its
    /// register and line.
    fn var(&self, v: &Var, plus_one: bool) -> String {
        match v {
            Var::Iteration(l) if plus_one => format!("({} + 1)", self.iv(*l)),
            Var::Iteration(l) => self.iv(*l),
            Var::Unknown(u) => self.leaf_name(&SymbolicConstant::Unknown(*u)),
            v => v.to_string(),
        }
    }

    /// A count, a number with its thousands separated by `,`, else an
    /// expression with a product as `a·b`, parenthesized past one term.
    fn count(&self, n: &SymExpr) -> String {
        match n.as_const() {
            Some(n) => {
                let digits = n.unsigned_abs().to_string();
                let mut out = String::from(if n < 0 { "-" } else { "" });
                for (i, c) in digits.chars().enumerate() {
                    if i > 0 && (digits.len() - i) % 3 == 0 {
                        out.push(',');
                    }
                    out.push(c);
                }
                out
            }
            None => match n.render(&|leaf| self.leaf_name(leaf)).replace(" * ", "·") {
                t if t.contains(' ') => format!("({t})"),
                t => t,
            },
        }
    }

    /// The threads in `set`, of block `b`, by warps and lanes; without a block
    /// shape, the condition that selects them, if one does. Where they
    /// read the CTA index and the grid is known: the lanes and the CTAs'
    /// condition as `when` prints it, or where the lanes differ by CTA
    /// the most CTAs' lanes.
    fn threads(&self, b: BlockId, set: &Pred) -> Option<String> {
        let shape = self.counts.shape;
        if shape.contains(&0) {
            return self.condition(set);
        }
        let exact = set.lane_exact();
        let classes = (self.counts.grid()).and_then(|g| Some((g, set.cta_classes(shape, g)?)));
        let Some((grid, classes)) = classes else {
            return Some(warp_lanes(&set.lane_masks(shape, None), shape, exact));
        };
        let total: u64 = grid.iter().map(|&n| u64::from(n)).product();
        let running: Vec<_> = (classes.iter())
            .filter(|(m, _)| m.iter().any(|w| w.0 != 0))
            .collect();
        let n: u64 = running.iter().map(|(_, n)| n).sum();
        let most = running
            .iter()
            .copied()
            .reduce(|a, b| if b.1 > a.1 { b } else { a });
        let Some((masks, most)) = most else {
            return Some("no thread".to_owned());
        };
        let lanes = warp_lanes(masks, shape, exact);
        Some(match running.len() {
            1 => {
                let mut tags = (BTreeSet::new(), Uniform::Cta);
                let parts: Vec<String> = (cta_conjuncts(set).into_iter())
                    .filter_map(|p| self.when(b, &p, true, &mut tags))
                    .collect();
                format!("{lanes} | CTAs {} ({n}/{total})", parts.join(" and "))
            }
            _ => format!(
                "{lanes} in {most}/{total} CTAs, other {} CTAs differ",
                n - most
            ),
        })
    }

    /// `runs when` and the unknowns, each the same for the CTA or a warp,
    /// that block `b`'s threads meet past its immediate dominator's, then
    /// `CTA-uniform` where all are the same for the CTA, else
    /// `warp-uniform`, and the lines of the branches deciding them; unless
    /// no thread runs it.
    fn runs_when(&self, b: BlockId) -> Option<String> {
        let sets = &self.counts.sets;
        let runs = sets.blocks[b.0 as usize] != Pred::False;
        let mut tags = (BTreeSet::new(), Uniform::Cta);
        let when = self.when(
            b,
            runs.then_some(&sets.relative[b.0 as usize])?,
            false,
            &mut tags,
        )?;
        let scope = match tags.1 {
            Uniform::Cta => "CTA-uniform",
            Uniform::Warp => "warp-uniform",
        };
        let lines: Vec<String> = tags.0.iter().map(u32::to_string).collect();
        let branch = match &lines[..] {
            [] => String::new(),
            [line] => format!(", branch at line {line}"),
            _ => format!(", branches at lines {}", lines.join(", ")),
        };
        Some(format!("runs when {when} ({scope}{branch})"))
    }

    /// The line of the nearest block dominating `b` that runs under a
    /// uniform condition of its own, after `nested in` where `b` has one,
    /// `adds`, else after `under`.
    fn nesting(&self, b: BlockId, adds: bool) -> Option<String> {
        let has = |d: BlockId| self.runs_when(d).is_some();
        let mut up = self.forest.doms.idom[b.0 as usize];
        while let Some(d) = up.filter(|&d| !has(d)) {
            up = self.forest.doms.idom[d.0 as usize];
        }
        let line = self.line(self.cfg.block(up?).start)?;
        Some(match adds {
            true => format!("nested in line {line}"),
            false => format!("under line {line}"),
        })
    }

    /// The uniform unknowns of `p`, block `b`'s set on its immediate
    /// dominator, by `atom`, and its comparisons the same for the CTA by
    /// `formula`, joined by ` and `, a conjunct without either left to the
    /// threads shown; an `or` or a `not` holding one with all its parts
    /// (`whole`), the `not` of an `and` or an `or` as the `or` or the `and`
    /// of its parts' negations, a thread-index comparison by `formula` and
    /// any other unknown as `?`; an `or` among the parts of an `and`, and
    /// an `and` among those of an `or`, parenthesized. Each unknown and
    /// comparison the same for the CTA adds the line of the branch
    /// deciding it and its scope to `tags`.
    fn when(&self, b: BlockId, p: &Pred, whole: bool, tags: &mut Tags) -> Option<String> {
        let paren = |t: String, inner: bool| if inner { format!("({t})") } else { t };
        match p {
            Pred::Uniform(u, a) => Some(self.atom(b, *u, a, tags)),
            Pred::And(..) => {
                let parts: Vec<(String, bool)> = (operands(p, false).into_iter())
                    .filter_map(|q| {
                        Some((self.when(b, q, whole, tags)?, matches!(q, Pred::Or(..))))
                    })
                    .collect();
                let several = parts.len() > 1;
                let parts: Vec<String> = (parts.into_iter())
                    .map(|(t, or)| paren(t, or && several))
                    .collect();
                (!parts.is_empty()).then(|| parts.join(" and "))
            }
            Pred::Or(..) | Pred::Not(_)
                if !whole
                    && p.atoms().is_empty()
                    && !p.comparisons().into_iter().any(cta_uniform) =>
            {
                None
            }
            Pred::Or(..) => {
                let parts = (operands(p, true).into_iter())
                    .map(|q| {
                        Some(paren(
                            self.when(b, q, true, tags)?,
                            matches!(q, Pred::And(..)),
                        ))
                    })
                    .collect::<Option<Vec<String>>>()?;
                Some(parts.join(" or "))
            }
            Pred::Not(x) => match &**x {
                Pred::And(y, z) | Pred::Or(y, z) => {
                    let (y, z) = (Box::new(!(**y).clone()), Box::new(!(**z).clone()));
                    let dual = match **x {
                        Pred::And(..) => Pred::Or(y, z),
                        _ => Pred::And(y, z),
                    };
                    self.when(b, &dual, whole, tags)
                }
                x => Some(format!("not ({})", self.when(b, x, true, tags)?)),
            },
            Pred::Cmp(c) if whole || cta_uniform(c) => {
                if cta_uniform(c) {
                    let decider = self.counts.sets.comparison_deciders.get(&(b, c.clone()));
                    tags.0.extend(decider.and_then(|&s| self.line(s)));
                }
                Some(self.formula(&c.form, c.cmp))
            }
            Pred::Elect(_) if whole => Some("elected".to_owned()),
            Pred::Unknown if whole => Some("?".to_owned()),
            _ => None,
        }
    }

    /// Atom `a` of block `b`'s set, the same for the threads `u` names:
    /// its comparison, negated where `a` is, or what it reads; the line of
    /// the branch deciding it but a jump table, and `u`, added to `tags`.
    fn atom(&self, b: BlockId, u: Uniform, a: &Atom, tags: &mut Tags) -> String {
        let sets = &self.counts.sets;
        let cmp = |c: Cmp| if a.negated { c.inverse() } else { c };
        let not = |t: String| if a.negated { format!("not ({t})") } else { t };
        let name = |b: BlockId| self.cfg.block_name(self.module, b);
        let what = match (&a.key, sets.forms.get(&a.key)) {
            (_, Some((form, c))) => self.formula(form, cmp(*c)),
            (AtomKey::OutOfLoop(k), _) => self.formula(&k.form, cmp(k.cmp)),
            (AtomKey::Def(op), _) => {
                let reg = writes(self.module, &self.kernel.stmts[*op])
                    .first()
                    .copied();
                self.predicate(reg.map(|reg| Value::Result { op: *op, reg }), a.negated)
            }
            (AtomKey::Value(v), _) => self.predicate(Some(*v), a.negated),
            (AtomKey::Jump { stmt, .. }, _) => {
                let line = self.line(*stmt).unwrap_or_default();
                not(format!("jump table at line {line} selects this target"))
            }
            (AtomKey::Leave { header, target }, _) => not(format!(
                "leaves loop {} toward {}",
                name(*header),
                name(*target)
            )),
        };
        let line = (sets.deciders.get(&(b, a.key.clone())))
            .filter(|_| !matches!(a.key, AtomKey::Jump { .. }))
            .and_then(|&s| self.line(s));
        tags.0.extend(line);
        tags.1 = tags.1.max(u);
        what
    }

    /// `predicate <register> (line <L>) is true`, or `false` where
    /// `negated`; a φ or an undefined value by its name.
    fn predicate(&self, v: Option<Value>, negated: bool) -> String {
        let truth = if negated { "false" } else { "true" };
        let name = match v {
            Some(Value::Result {
                op,
                reg: Register::Named(r),
            }) => {
                let line = self.line(op).unwrap_or_default();
                format!("{} (line {line})", self.module.interner.resolve(r))
            }
            Some(v) => Self::value_name(self.module, self.kernel, self.cfg, self.lines, v),
            None => "?".to_owned(),
        };
        format!("predicate {name} is {truth}")
    }

    /// `form cmp 0` as `lhs cmp rhs`, a loop's iteration as `shifted`
    /// gives it and an unknown value by its register and line: the terms
    /// and symbols of negative coefficient and the constant moved right,
    /// the comparison swapped where nothing would stay left.
    fn formula(&self, form: &AffineExpr, cmp: Cmp) -> String {
        let (form, nested) = shifted(form);
        let negative = |c: &SymExpr| SymExpr::split_const(c.clone()).0 < 0;
        let zero = || AffineExpr::from_base(SymExpr::Const(0));
        let split = |f: &AffineExpr| {
            let summands = match &f.base {
                SymExpr::Sum(ts) => ts.clone(),
                base => vec![base.clone()],
            };
            let right = |t: &SymExpr| negative(t) || matches!(t, SymExpr::Const(_));
            (f.terms.iter())
                .map(|(v, c)| (negative(c), AffineExpr::term(v.clone(), c.clone())))
                .chain(
                    summands
                        .into_iter()
                        .map(|t| (right(&t), AffineExpr::from_base(t))),
                )
                .fold((zero(), zero()), |(l, r), (right, x)| match right {
                    true => (l, r + x),
                    false => (l + x, r),
                })
        };
        let ((lhs, rhs), cmp) = match split(&form) {
            (lhs, _) if lhs == zero() => (split(&form.scale(SymExpr::Const(-1))), cmp.swapped()),
            sides => (sides, cmp),
        };
        let op = match cmp {
            Cmp::Lt => "<",
            Cmp::Le => "≤",
            Cmp::Gt => ">",
            Cmp::Ge => "≥",
            Cmp::Eq => "=",
            Cmp::Ne => "≠",
        };
        let name = |v: &Var| match v {
            Var::Unknown(u) => Self::value_name(self.module, self.kernel, self.cfg, self.lines, *u),
            v => self.var(v, nested),
        };
        let side = |s: AffineExpr| s.render(name).replace(" * ", "·");
        let rhs = side(rhs.scale(SymExpr::Const(-1)));
        self.shorten(format!("{} {op} {rhs}", side(lhs)))
    }

    /// The condition on the thread and CTA indices that selects `set`, a
    /// loop's iteration k as `iv + 1`.
    fn condition(&self, set: &Pred) -> Option<String> {
        (set.render_named(&|v| self.var(v, true))).map(|t| t.replace(" * ", "·"))
    }

    /// Loop `l`'s iteration from 0, `iv[<header label>]`, LLVM's canonical
    /// induction variable (llvm/include/llvm/Analysis/LoopInfo.h).
    fn iv(&self, l: LoopId) -> String {
        let header = self.forest.get(l).header;
        format!("iv[{}]", self.cfg.block_name(self.module, header))
    }

    /// Loop `l`'s trip count, or `?` and why it is not known; a product
    /// prints as `a·b`.
    fn trips(&self, l: LoopId) -> String {
        match &self.counts.trip_info.trips[l.0 as usize] {
            Ok(e) => {
                (e.bind(self.bind_map).render(&|leaf| self.leaf_name(leaf))).replace(" * ", "·")
            }
            Err(reason) => format!("? ({reason})"),
        }
    }
}

/// `warp <W> × lane <L>` per group of warps with the same lanes, joined by
/// ` ∪ `, `*` for every warp or every lane of one, after `<= ` where the
/// lanes are a bound, unless they are `exact` whenever the warp runs;
/// `no thread` when none may run. A warp's lanes are its threads' linear
/// indices modulo 32 (PTX ISA §3.1).
fn warp_lanes(masks: &[(u32, u32)], shape: [u32; 3], exact: bool) -> String {
    let threads: u64 = shape.iter().map(|&n| u64::from(n)).product();
    let full = |w: usize| u32::MAX >> (32 - (threads - 32 * w as u64).min(32));
    let mut groups: Vec<((bool, String), Vec<i64>)> = Vec::new();
    for (w, &(may, must)) in masks.iter().enumerate().filter(|(_, m)| m.0 != 0) {
        let lanes = match may == full(w) {
            true => "*".to_owned(),
            false => runs(&(0..32).filter(|l| may >> l & 1 == 1).collect::<Vec<_>>()),
        };
        let key = (may != must && !exact, lanes);
        match groups.iter_mut().find(|g| g.0 == key) {
            Some(g) => g.1.push(w as i64),
            None => groups.push((key, vec![w as i64])),
        }
    }
    let group = |((bound, lanes), warps): &((bool, String), Vec<i64>)| {
        let warps = match warps.len() == masks.len() {
            true => "*".to_owned(),
            false => runs(warps),
        };
        format!(
            "{}warp {warps} × lane {lanes}",
            if *bound { "<= " } else { "" }
        )
    };
    match groups.is_empty() {
        true => "no thread".to_owned(),
        false => groups.iter().map(group).collect::<Vec<_>>().join(" ∪ "),
    }
}

/// `a` with each `c·k` of a loop's iteration k = iv + 1 as `c·iv`, its c
/// in the constant, unless a floor or a remainder holds a counter
/// (`true`): then k stays, to print as `(iv + 1)`.
fn shifted(a: &AffineExpr) -> (AffineExpr, bool) {
    let nested = (a.terms.keys()).any(|v| v.counter().is_some() && !matches!(v, Var::Iteration(_)));
    let k = (a.terms.iter()).filter(|(v, _)| matches!(v, Var::Iteration(_)) && !nested);
    let base = k.fold(a.base.clone(), |base, (_, c)| SymExpr::add(base, c.clone()));
    let shifted = AffineExpr {
        terms: a.terms.clone(),
        base,
    };
    (shifted, nested)
}

/// Comparison `c` reads no thread or CTA index, so it is the same for
/// the CTA, a loop counter's or a parameter's.
fn cta_uniform(c: &Constraint) -> bool {
    c.cta.is_none() && c.form.terms.iter().all(|(v, _)| !depends_on_lane(v))
}

/// The parts of `p` under the `and`s at its top, or the `or`s where `or`.
fn operands(p: &Pred, or: bool) -> Vec<&Pred> {
    match (p, or) {
        (Pred::And(x, y), false) | (Pred::Or(x, y), true) => {
            [operands(x, or), operands(y, or)].concat()
        }
        _ => vec![p],
    }
}

/// The parts of a conjunction that read the CTA index, the `not` of an
/// `or` taken as the `and` of its parts' negations.
fn cta_conjuncts(set: &Pred) -> Vec<Pred> {
    if let Pred::Not(x) = set
        && let Pred::Or(a, b) = &**x
    {
        return [!(**a).clone(), !(**b).clone()]
            .iter()
            .flat_map(cta_conjuncts)
            .collect();
    }
    match set {
        Pred::And(a, b) => [cta_conjuncts(a), cta_conjuncts(b)].concat(),
        p if p.varies_by_cta() => vec![p.clone()],
        _ => Vec::new(),
    }
}

/// Increasing values as `a-b` for a run of consecutive ones, `a-b:s` for
/// three or more `s` apart, else one by one, joined by `,`.
fn runs(xs: &[i64]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < xs.len() {
        let step = xs.get(i + 1).map_or(1, |x| x - xs[i]);
        let end = (i + 1..xs.len())
            .find(|&j| xs[j] - xs[j - 1] != step)
            .unwrap_or(xs.len());
        let len = if end - i == 2 && step != 1 {
            1
        } else {
            end - i
        };
        let (first, last) = (xs[i], xs[i + len - 1]);
        out.push(match (len, step) {
            (1, _) => first.to_string(),
            (_, 1) => format!("{first}-{last}"),
            _ => format!("{first}-{last}:{step}"),
        });
        i += len;
    }
    out.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::bindings::parse_bind;

    #[test]
    fn runs_of_values_print_as_ranges_strides_and_singles() {
        assert_eq!(runs(&[0, 1, 2, 5, 6, 7]), "0-2,5-7");
        assert_eq!(runs(&[0, 8, 16, 24]), "0-24:8");
        assert_eq!(runs(&[0, 5]), "0,5");
        assert_eq!(runs(&[0, 2, 3, 4]), "0,2-4");
        assert_eq!(runs(&[3]), "3");
    }

    #[test]
    fn warps_with_the_same_lanes_print_as_one_group() {
        let shape = [80, 1, 1];
        let all = [(u32::MAX, u32::MAX), (u32::MAX, u32::MAX), (0xffff, 0xffff)];
        assert_eq!(warp_lanes(&all, shape, true), "warp * × lane *");
        let lane_0 = [(1, 1), (1, 1), (1, 0)];
        assert_eq!(
            warp_lanes(&lane_0, shape, false),
            "warp 0-1 × lane 0 ∪ <= warp 2 × lane 0"
        );
        assert_eq!(warp_lanes(&lane_0, shape, true), "warp * × lane 0");
        assert_eq!(warp_lanes(&[(0, 0); 3], shape, true), "no thread");
    }

    /// Without a block shape, a block's annotation is the condition on the
    /// thread index that selects it, a loop's iteration k = iv + 1, and
    /// its count per thread but where every thread runs it once; the
    /// kernel's, its bytes per thread, a bound under the loop's guard.
    #[test]
    fn without_a_block_shape_a_block_names_its_condition() {
        let src = "\
.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 p)
{
	.reg .pred %p<4>;
	.reg .b32 %r<3>;
	.reg .b64 %rd<2>;
	ld.param.u64 %rd1, [p];
	mov.u32 %r1, %tid.x;
	mov.u32 %r2, 0;
$L__BB0_1:
	add.s32 %r2, %r2, 1;
	setp.ge.u32 %p2, %r1, %r2;
	@%p2 bra $L__BB0_3;
	st.global.u32 [%rd1], %r2;
$L__BB0_3:
	setp.lt.u32 %p3, %r2, 4;
	@%p3 bra $L__BB0_1;
	setp.gt.u32 %p1, %r1, 31;
	@%p1 bra $L__BB0_5;
	st.global.u32 [%rd1], %r1;
$L__BB0_5:
	ret;
}
";
        assert_eq!(
            notes(src, &AnalyzeOptions::default()),
            [
                "per thread: global store <= 20 B; param load 8 B",
                "loop: trips 4 | exec 4/thread",
                "[<block 2>] %tid.x - (iv[$L__BB0_1] + 1) < 0 | exec <= 4/thread | \
                 global store [param_0] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
                "exec 4/thread",
                "[<block 5>] %tid.x <= 31 | exec <= 1/thread | \
                 global store [param_0] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
            ]
        );
    }

    /// A block under uniform conditions names them past its immediate
    /// dominator's: each as its comparison with the branch's polarity,
    /// the sides swapped where every term would move right, two in a
    /// conjunction joined by `and`, the `not` of an `or` as the `and` of
    /// its parts' negations, its thread-index part left to the threads
    /// shown; then `CTA-uniform` and the deciding branch's line.
    #[test]
    fn a_block_names_the_uniform_conditions_it_runs_under() {
        let src = "\
.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 p, .param .u32 n, .param .u32 m)
.reqntid 64, 1, 1
{
	.reg .pred %p<8>;
	.reg .b32 %r<4>;
	.reg .b64 %rd<2>;
	ld.param.u64 %rd1, [p];
	ld.param.u32 %r1, [n];
	ld.param.u32 %r2, [m];
	mov.u32 %r3, %tid.x;
	setp.lt.s32 %p1, %r1, 1;
	setp.ne.s32 %p2, %r3, 0;
	or.pred %p3, %p1, %p2;
	@%p3 bra $L__BB0_2;
	st.global.u32 [%rd1], %r3;
$L__BB0_2:
	setp.gt.s32 %p4, %r2, 3;
	and.pred %p5, %p4, %p1;
	@!%p5 bra $L__BB0_4;
	st.global.u32 [%rd1], %r1;
$L__BB0_4:
	@%p1 bra $L__BB0_6;
	st.global.u32 [%rd1], %r2;
$L__BB0_6:
	setp.lt.s32 %p6, 5, %r2;
	@%p6 bra $L__BB0_8;
	st.global.u32 [%rd1], %r3;
$L__BB0_8:
	ret;
}
";
        let out = annotate(src, &AnalyzeOptions::default()).expect("annotates");
        let when: Vec<&str> = (out.lines())
            .filter_map(|l| l.split_once("runs when ")?.1.split(" | ").next())
            .collect();
        assert_eq!(
            when,
            [
                "param_1 ≥ 1 (CTA-uniform, branch at line 17)",
                "param_2 > 3 and param_1 < 1 (CTA-uniform, branch at line 22)",
                "param_1 ≥ 1 (CTA-uniform, branch at line 25)",
                "param_2 ≤ 5 (CTA-uniform, branch at line 29)",
            ]
        );
    }

    /// Two statements on one line: their annotations joined by ` ; `.
    #[test]
    fn statements_on_one_line_join_their_annotations() {
        let src = "\
.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 p)
.reqntid 32, 1, 1
{
	.reg .b32 %r<2>;
	.reg .b64 %rd<2>;
	ld.param.u64 %rd1, [p]; mov.u32 %r1, %tid.x;
	st.global.u32 [%rd1], %r1; st.global.u32 [%rd1+4], %r1;
	ret;
}
";
        let out = annotate(src, &AnalyzeOptions::default()).expect("annotates");
        let line = out
            .lines()
            .find(|l| l.contains("[%rd1+4]"))
            .expect("the stores");
        assert_eq!(
            line.split_once(" // ").map(|(_, n)| n),
            Some(
                "global store [param_0] sectors/req 1 (ideal 1) ; \
                 global store [param_0 + 4] sectors/req 1 (ideal 1)"
            )
        );
    }

    /// Each annotation of `src` under `opts`, in line order.
    fn notes(src: &str, opts: &AnalyzeOptions) -> Vec<String> {
        let out = annotate(src, opts).expect("annotates");
        (out.lines())
            .filter_map(|l| l.split_once(" // "))
            .map(|(_, n)| n.to_owned())
            .collect()
    }

    /// The annotation of the first line of `src` holding `ptx`, under
    /// `opts`; `None` where that line has none.
    fn note(src: &str, opts: &AnalyzeOptions, ptx: &str) -> Option<String> {
        let out = annotate(src, opts).expect("annotates");
        let line = (out.lines())
            .find(|l| l.split(" // ").next().is_some_and(|p| p.contains(ptx)))
            .expect("a line holds it");
        line.split_once(" // ").map(|(_, n)| n.to_owned())
    }

    /// What the counted statements of `src`'s kernel sum to per thread
    /// under `opts`, as the `.entry` line prints it without a block shape.
    fn per_thread(src: &str, opts: &AnalyzeOptions) -> Option<String> {
        let mut out = None;
        each_kernel(src, opts, |b| out = b.totals(Per::Thread)).expect("analyzes");
        out
    }

    /// Each memory operand of `src`'s kernel as its line's annotation
    /// prints it, an `unused` instruction's too.
    fn memory_operands(src: &str, opts: &AnalyzeOptions) -> Vec<String> {
        let mut out = Vec::new();
        each_kernel(src, opts, |b| {
            (0..b.kernel.stmts.len())
                .filter(|&s| matches!(b.kernel.stmts[s], Stmt::Instr(_)))
                .for_each(|s| out.extend(b.operands(s)));
        })
        .expect("analyzes");
        out
    }

    /// Blocks selected by thread-index branches: the two halves of a
    /// warp-specialized CTA, warps 4-7 and 0-3, and an elected thread,
    /// lane 0 of warp 0. Per CTA their counts are exact: 128 threads add
    /// once and 128 multiply twice, 384 FLOPs. Per thread they stay
    /// bounds: a thread runs one path. Triton broadcasts the warp index
    /// from lane 0 before comparing; the same warps run that half.
    #[test]
    fn thread_index_branches_select_blocks() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 256, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\nshr.u32 %r2, %r1, 5;\n\
                   setp.lt.u32 %p1, %r2, 4;\n@%p1 bra $L__A;\n\
                   add.f32 %f1, %f1, %f1;\nbra.uni $L__J;\n\
                   $L__A:\nmul.f32 %f1, %f1, %f1;\nmul.f32 %f1, %f1, %f1;\n\
                   $L__J:\nsetp.ne.s32 %p2, %r1, 0;\n@%p2 bra $L__END;\n\
                   st.global.f32 [%rd1], %f1;\n$L__END:\nret;\n}\n";
        assert_eq!(
            notes(src, &opts),
            [
                "block 256×1×1 | per CTA: flops f32 384; global store 4 B; param load 2,048 B",
                "[<block 0>] warp * × lane * | exec 256/CTA",
                "[<block 1>] warp 4-7 × lane * | exec 128/CTA",
                "warp 0-3 × lane * | exec 128/CTA",
                "warp * × lane * | exec 256/CTA",
                "[<block 4>] warp 0 × lane 0 | exec 1/CTA | \
                 global store [param_0] sectors/req 1 (ideal 1)",
                "warp * × lane * | exec 256/CTA",
            ]
        );
        assert_eq!(
            per_thread(src, &opts).as_deref(),
            Some("per thread: flops f32 <= 3; global store <= 4 B; param load 8 B")
        );
        let src = src.replace(
            "setp.lt.u32 %p1, %r2, 4;",
            "shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;\nsetp.lt.u32 %p1, %r3, 4;",
        );
        assert_eq!(
            note(&src, &opts, "$L__A:").as_deref(),
            Some("warp 0-3 × lane * | exec 128/CTA")
        );
    }

    /// A guarded redefinition of the guard register is not read: the
    /// store's bytes per CTA are a bound. Nine `and.pred` steps exceed
    /// the reading depth: 64 × 4 B as a bound; the base predicate read
    /// directly afterwards still selects warp 0 exactly, 32 × 2 B.
    #[test]
    fn guarded_predicate_definitions_and_deep_chains_stay_bounds() {
        let opts = AnalyzeOptions::default();
        let head = ".version 8.7\n.target sm_80\n.address_size 64\n\
                    .visible .entry k(\n.param .u64 k_param_0\n)\n\
                    .reqntid 64, 1, 1\n{\n\
                    ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n\
                    setp.lt.u32 %p0, %r1, 32;\nsetp.eq.u32 %p9, %r1, 0;\n";
        let src = format!(
            "{head}@%p9 setp.lt.u32 %p0, %r1, 64;\n@%p0 st.global.u32 [%rd1], %r1;\nret;\n}}\n"
        );
        assert_eq!(
            note(&src, &opts, ".entry").as_deref(),
            Some("block 64×1×1 | per CTA: global store <= 256 B; param load 512 B")
        );
        let chain: String = (1..=9)
            .map(|i| format!("and.pred %p{i}, %p{}, %p{};\n", i - 1, i - 1))
            .collect();
        let src = format!(
            "{head}{chain}@%p9 st.global.u32 [%rd1], %r1;\n@%p0 st.global.u16 [%rd1+4], %rs1;\nret;\n}}\n"
        );
        assert_eq!(
            note(&src, &opts, ".entry").as_deref(),
            Some("block 64×1×1 | per CTA: global store <= 320 B; param load 512 B")
        );
        assert_eq!(
            note(&src, &opts, "st.global.u16").as_deref(),
            Some(
                "warp 0 × lane * | exec 32/CTA | global store [param_0 + 4] sectors/req 1 (ideal 1)"
            )
        );
    }

    /// A guard every thread reads as true keeps the guarded load's bytes
    /// per CTA exact.
    #[test]
    fn a_guard_every_thread_reads_as_true_keeps_counts_exact() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nsetp.eq.u32 %p1, 1, 1;\n\
                   @%p1 ld.global.u32 %r1, [%rd1];\nst.global.u32 [%rd1], %r1;\nret;\n}\n";
        assert_eq!(
            note(src, &AnalyzeOptions::default(), ".entry").as_deref(),
            Some("block 32×1×1 | per CTA: global load 128 B, store 128 B; param load 256 B")
        );
    }

    /// The store's block is reached only through two branches no thread
    /// takes, so the load it alone reads is `unused`, with or without a
    /// launch shape ptxas never sees.
    #[test]
    fn a_launch_shape_leaves_the_unused_instructions_alone() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.global.u32 %r1, [%rd1];\n\
                   mov.u32 %r9, %tid.x;\nmov.pred %p1, 0;\nsetp.lt.u32 %p2, %r9, 4;\n\
                   @%p2 bra $L__B;\n@%p1 bra $L__S;\nbra.uni $L__E;\n\
                   $L__B:\n@%p1 bra $L__S;\nbra.uni $L__E;\n\
                   $L__S:\nst.global.u32 [%rd1], %r1;\n$L__E:\nret;\n}\n";
        let unused = |launch| {
            let opts = AnalyzeOptions {
                launch,
                ..Default::default()
            };
            let out = annotate(src, &opts).expect("annotates");
            (out.lines())
                .filter_map(|l| l.split_once(" // "))
                .filter(|(_, n)| n.ends_with("unused"))
                .map(|(ptx, _)| ptx.to_owned())
                .collect::<Vec<_>>()
        };
        assert!(unused(None).contains(&"ld.global.u32 %r1, [%rd1];".to_owned()));
        assert_eq!(unused(None), unused(Some([32, 1, 1])));
    }

    /// `%laneid` in a one-dimensional block is `%tid.x` mod 32: lane 0 of
    /// each of 2 warps stores 4 B, 8 B per CTA, exact. In two rows of 32
    /// the lane is not read from `%tid.x` alone: 64 × 4 B, a bound.
    #[test]
    fn laneid_is_the_lane_in_a_one_dimensional_block() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n\
                   .reqntid 64, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %laneid;\n\
                   setp.eq.u32 %p1, %r1, 0;\n@%p1 st.global.u32 [%rd1], %r1;\nret;\n}\n";
        assert_eq!(
            note(src, &opts, ".entry").as_deref(),
            Some("block 64×1×1 | per CTA: global store 8 B; param load 512 B")
        );
        let src = src.replace(".reqntid 64, 1, 1", ".reqntid 32, 2, 1");
        assert_eq!(
            note(&src, &opts, ".entry").as_deref(),
            Some("block 32×2×1 | per CTA: global store <= 256 B; param load 512 B")
        );
    }

    /// Over a launch, a count that reads the CTA index is summed over the
    /// grid: `k < %ctaid.x` from 0 runs max(c, 1) trips in CTA c, 7 over
    /// 4 CTAs. A `!=` latch's count holds where its step divides its
    /// distance into a positive count: `k != 2 - %ctaid.x` by 1 is
    /// unknown with CTA 2, whose counter never meets 0, and by 2 with CTA
    /// 1; two nested loops of `1 - 2 * %ctaid.x` trips are unknown with
    /// CTA 1, though the product of their counts is 1 there. A guard
    /// skipping CTA 0, where `k != %ctaid.x` never meets its bound, leaves
    /// 1 + 2 + 3 trips. Past 2^24 CTAs on the axes read it is unknown.
    /// Each is the `.entry` line's global bytes stored over the launch,
    /// `?` where unknown.
    #[test]
    fn a_count_over_the_cta_index_is_summed_over_the_grid() {
        let store = |body: &str, grid: [u32; 3]| {
            let src = format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0,\n.param .u32 k_param_1\n)\n\
                 .reqntid 32, 1, 1\n{{\n\
                 ld.param.u64 %rd1, [k_param_0];\nld.param.u32 %r5, [k_param_1];\n\
                 mov.u32 %r1, %ctaid.x;\nmov.u32 %r6, %ctaid.y;\n{body}\nret;\n}}\n"
            );
            let opts = AnalyzeOptions {
                bindings: vec![parse_bind("1:n=2").expect("binding")],
                grid: Some(grid),
                ..Default::default()
            };
            let entry = note(&src, &opts, ".entry").expect("the launch's sums");
            let (_, sums) = entry.split_once("per launch: ").expect("a grid");
            (sums.split("; ")).find_map(|s| s.strip_prefix("global store ").map(str::to_owned))
        };
        let one = |bound: &str, step: i64, cmp: &str| {
            format!(
                "{bound}\nmov.u32 %r2, 0;\n$L__L:\nadd.s32 %r2, %r2, {step};\n\
                 st.global.u32 [%rd1], %r2;\nsetp.{cmp}.u32 %p1, %r2, %r3;\n@%p1 bra $L__L;"
            )
        };
        let (known, unknown) = (|n: &str| Some(format!("{n} B")), Some("? B".to_owned()));
        let below = one("mov.u32 %r3, %r1;", 1, "lt");
        assert_eq!(store(&below, [4, 1, 1]), known("896"));
        assert_eq!(store(&below, [4, 3, 1]), known("2,688"));
        let ne = |step| one("sub.s32 %r3, %r5, %r1;", step, "ne");
        assert_eq!(store(&ne(1), [2, 1, 1]), known("384"));
        assert_eq!(store(&ne(1), [3, 1, 1]), unknown);
        assert_eq!(store(&ne(2), [1, 1, 1]), known("128"));
        assert_eq!(store(&ne(2), [2, 1, 1]), unknown);
        let nested = "mov.u32 %r9, 1;\nshl.b32 %r7, %r1, 1;\nsub.s32 %r3, %r9, %r7;\n\
                      mov.u32 %r8, 0;\n$L__O:\nadd.s32 %r8, %r8, 1;\nmov.u32 %r2, 0;\n\
                      $L__L:\nadd.s32 %r2, %r2, 1;\nst.global.u32 [%rd1], %r2;\n\
                      setp.ne.u32 %p1, %r2, %r3;\n@%p1 bra $L__L;\n\
                      setp.ne.u32 %p2, %r8, %r3;\n@%p2 bra $L__O;";
        assert_eq!(store(nested, [1, 1, 1]), known("128"));
        assert_eq!(store(nested, [2, 1, 1]), unknown);
        let guarded = "setp.lt.s32 %p3, %r1, 1;\n@%p3 bra $L__E;\nmov.u32 %r2, 0;\n\
                       $L__L:\nadd.s32 %r2, %r2, 1;\nst.global.u32 [%rd1], %r2;\n\
                       setp.ne.u32 %p1, %r2, %r1;\n@%p1 bra $L__L;\n$L__E:";
        assert_eq!(store(guarded, [4, 1, 1]), known("768"));
        let both = one("add.s32 %r3, %r1, %r6;", 1, "lt");
        assert_eq!(store(&both, [8192, 4096, 1]), unknown);
    }

    /// A loop entered only by the threads a thread-index condition picks
    /// is a bound per thread, but per CTA its threads are counted:
    /// `%tid.x < 64` of 128 store 4 B on each of n trips.
    #[test]
    fn a_loop_entered_by_counted_threads_is_exact_per_cta() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0,\n.param .u32 k_param_1\n)\n\
                   .reqntid 128, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.param.u32 %r5, [k_param_1];\n\
                   mov.u32 %r1, %tid.x;\nsetp.ge.u32 %p2, %r1, 64;\n@%p2 bra $L__E;\n\
                   mov.u32 %r2, 0;\n$L__L:\nadd.s32 %r2, %r2, 1;\nst.global.u32 [%rd1], %r2;\n\
                   setp.lt.u32 %p1, %r2, %r5;\n@%p1 bra $L__L;\n$L__E:\nret;\n}\n";
        assert_eq!(
            per_thread(src, &opts).as_deref(),
            Some("per thread: global store <= (4·max(param_1, 1)) B; param load 12 B")
        );
        assert_eq!(
            note(src, &opts, ".entry").as_deref(),
            Some(
                "block 128×1×1 | per CTA: global store (256·max(param_1, 1)) B; param load 1,536 B"
            )
        );
    }

    /// A block a loop counter's mask selects, but that a short-circuit
    /// on loaded values may also skip, is a bound per CTA however its
    /// trips count.
    #[test]
    fn a_bypassed_block_under_a_counter_mask_is_a_bound() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 64, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.global.u32 %r7, [%rd1];\n\
                   ld.global.u32 %r8, [%rd1+4];\nmov.u32 %r2, %tid.x;\nmov.u32 %r3, 0;\n\
                   $L__H:\nshl.b32 %r4, %r3, 6;\nadd.s32 %r5, %r4, %r2;\n\
                   setp.ge.u32 %p2, %r5, 300;\n@%p2 bra $L__T;\n\
                   setp.ne.u32 %p4, %r7, 0;\n@%p4 bra $L__S;\n\
                   setp.eq.u32 %p5, %r8, 0;\n@%p5 bra $L__T;\n\
                   $L__S:\nst.global.u32 [%rd1+8], %r5;\n\
                   $L__T:\nadd.s32 %r3, %r3, 1;\nsetp.lt.u32 %p3, %r3, 8;\n@%p3 bra $L__H;\nret;\n}\n";
        assert_eq!(
            note(src, &AnalyzeOptions::default(), ".entry").as_deref(),
            Some("block 64×1×1 | per CTA: global load 512 B, store <= 2,048 B; param load 512 B")
        );
    }

    /// `a < b` as `s32` with `a = b + k`, `b = %tid.x + 2^31 - 16`: the
    /// difference is the counter alone, but `b` leaves the type, so the
    /// lanes differ (lane 15 alone at k = 1): the store's threads, count
    /// and sectors per warp request under the guard are bounds.
    #[test]
    fn a_counter_difference_past_its_type_is_no_per_trip_comparison() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n\
                   add.u32 %r2, %r1, 2147483632;\nmul.wide.u32 %rd2, %r1, 4;\n\
                   add.s64 %rd3, %rd1, %rd2;\nmov.u32 %r3, 0;\n$L__L:\nadd.u32 %r3, %r3, 1;\n\
                   add.u32 %r4, %r2, %r3;\nsetp.lt.s32 %p1, %r4, %r2;\n\
                   @%p1 st.global.u32 [%rd3], %r3;\nsetp.lt.u32 %p2, %r3, 4;\n\
                   @%p2 bra $L__L;\nret;\n}\n";
        assert_eq!(
            note(src, &AnalyzeOptions::default(), "st.global").as_deref(),
            Some(
                "<= warp * × lane * | exec <= 128/CTA | \
                 global store [4·%tid.x + param_0] sectors/req <= 5 (ideal <= 4)"
            )
        );
    }

    /// §10.32: `%dynamic_smem_size` is set "for the CTA … at launch time",
    /// so a branch on it keeps a warp whole; §10.23: `%clock` is a cycle
    /// counter, so a branch on it may split one, and the load's sectors
    /// per warp request are a bound. The load is `unused`: its operand as
    /// its line would print it.
    #[test]
    fn a_branch_on_a_per_cta_special_register_keeps_the_lanes_exact() {
        let opts = AnalyzeOptions::default();
        let sectors = |sreg: &str| {
            let src = format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 128, 1, 1\n{{\n\
                 ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, {sreg};\n\
                 setp.lt.u32 %p1, %r1, 1024;\n@%p1 bra $L__end;\n\
                 mov.u32 %r2, %tid.x;\nmul.wide.u32 %rd2, %r2, 4;\n\
                 add.s64 %rd3, %rd1, %rd2;\nld.global.f32 %f1, [%rd3];\n$L__end:\nret;\n}}\n"
            );
            memory_operands(&src, &opts)
        };
        assert_eq!(
            sectors("%dynamic_smem_size"),
            ["global load [4·%tid.x + param_0] sectors/req 4–5 (ideal 4)"]
        );
        assert_eq!(
            sectors("%clock"),
            ["global load [4·%tid.x + param_0] sectors/req <= 5 (ideal <= 4)"]
        );
    }

    /// A guard `%tid.x < 16 and min(p, q) < 7`: the second half is not
    /// read but uniform, so each warp's lanes are still known and the
    /// load's sectors per warp request exact. The load is `unused`: its
    /// operand as its line would print it.
    #[test]
    fn a_cta_uniform_half_of_an_and_keeps_the_lanes_exact() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0,\n.param .u32 k_param_1\n)\n\
                   .reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.param.u32 %r3, [k_param_1];\n\
                   mov.u32 %r1, %tid.x;\nsetp.lt.u32 %p1, %r1, 16;\n\
                   max.s32 %r5, %r3, 9;\nmin.s32 %r6, %r5, %r3;\n\
                   setp.lt.s32 %p2, %r6, 7;\nand.pred %p3, %p1, %p2;\n\
                   mul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                   @%p3 ld.global.f32 %f1, [%rd3];\nret;\n}\n";
        assert_eq!(
            memory_operands(src, &AnalyzeOptions::default()),
            ["global load [4·%tid.x + param_0] sectors/req 2–3 (ideal 2)"]
        );
    }
}
