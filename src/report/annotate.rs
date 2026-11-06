//! The input PTX with what the analysis says of each line appended to
//! it as a ` // ` comment.

use crate::analysis::affine::affine_expr::{AffineExpr, Var};
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::unknown_reason::UnknownReason;
use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::execution_counts::{
    CountAccumulator, CtaSize, Participation, Per, Resolution,
};
use crate::analysis::instruction_counts::collect::BlockMeasurements;
use crate::analysis::instruction_counts::measurement::{MeasureKind, Pipe};
use crate::analysis::memory_footprint::{ByteRange, CountRange};
use crate::analysis::thread_participation::{Atom, AtomKey, Constraint, Pred};
use crate::analysis::uniformity_info::Uniform;
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Opcode, Register, Stmt};
use crate::ptx::modifier::Cmp;
use crate::ptx::register_effects::writes;
use crate::ptx::thread_hierarchy::{WARP_SZ, ctas, threads};
use crate::report::build::{
    Access, AnalyzeError, AnalyzeOptions, KernelReportBuilder, each_kernel,
};

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
        let threads = threads(self.counts.shape) as u64;
        let cta = (threads > 0).then(|| CtaSize {
            threads,
            warps: threads.div_ceil(WARP_SZ as u64),
            exact: true,
        });
        let ctas = self.counts.grid().and_then(ctas);
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
            for &s in &bm.instructions {
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
    /// parameter's value, and the alignment taken for pointers; then the
    /// shared memory each CTA declares, `+ dynamic` where an `.extern
    /// .shared` array's size is set at launch; then the FLOPs and bytes
    /// its statements sum to at the scale `per`.
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
            self.assumed_align(),
            self.shared(),
            self.totals(per),
        ]
        .into_iter()
        .flatten()
        .collect();
        (!parts.is_empty()).then(|| parts.join(" | "))
    }

    /// `pointer params assumed N B aligned` where a global access that
    /// touches sectors sits on a pointer parameter whose `.ptr
    /// .align` declares less than the `N` taken.
    fn assumed_align(&self) -> Option<String> {
        let below = |a: &AffineExpr| {
            let i = self.pointer(a)?;
            Some(self.kernel.params[i].ptr_align.unwrap_or(1) < self.ptr_align)
        };
        let counted = |row: &Access| row.sectors_per_request.is_some_and(|n| n.max > 0);
        let assumed = (self.operands.values().flatten())
            .filter(|(row, _)| row.space != "shared" && counted(row))
            .filter_map(|(_, traced)| below(traced.as_ref()?.as_ref().ok()?))
            .any(|b| b);
        assumed.then(|| format!("pointer params assumed {} B aligned", self.ptr_align))
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
        for &s in &bm.instructions {
            let exec = match self
                .counts
                .statement(bm, &m, s, 0, block.as_ref(), &mut cache)
            {
                Resolution::Skipped => format!("exec 0/{unit}"),
                Resolution::Unknown => text(None),
                Resolution::Counted(p) => text(Some(p)),
            };
            let Stmt::Instr(instr) = &self.kernel.stmts[s] else {
                continue;
            };
            if exec != own && !(instr.mnemonic == Opcode::Bra && instr.predicate.is_some()) {
                parts.insert(s, vec![exec]);
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
            (has(|k| matches!(k, MeasureKind::UnknownOps)))
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
            Var::Iteration(l) if !plus_one => Self::iv(self.data, *l),
            v => Self::var_text(self.data, self.lines, v),
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
        let total = ctas(grid)?;
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
    /// (`whole`), a thread-index comparison by `formula` and
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
            Pred::Not(x) => Some(format!("not ({})", self.when(b, x, true, tags)?)),
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
        let zero = || AffineExpr::constant(0);
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

    /// The condition on the thread and CTA indices that selects `set`,
    /// each comparison as `formula` writes it.
    fn condition(&self, set: &Pred) -> Option<String> {
        set.render_named(&|c| self.formula(&c.form, c.cmp))
    }

    /// Loop `l`'s trip count, or `?` and why it is not known; a product
    /// prints as `a·b`.
    fn trips(&self, l: LoopId) -> String {
        match &self.counts.trip_info.trips[l.0 as usize] {
            Ok(e) => (e.render(&|leaf| self.leaf_name(leaf))).replace(" * ", "·"),
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
    let (threads, lanes) = (threads(shape) as u64, WARP_SZ as u64);
    let full = |w: usize| u32::MAX >> (lanes - (threads - lanes * w as u64).min(lanes));
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
    c.cta.is_none() && !c.form.depends_on_tid()
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

/// The parts of a conjunction that read the CTA index.
fn cta_conjuncts(set: &Pred) -> Vec<Pred> {
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
    use crate::report::build::compare;
    use crate::report::measured::{Comparison, NcuLine, NcuRecord};

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
                "[<block 2>] %tid.x < iv[$L__BB0_1] + 1 | exec <= 4/thread | \
                 global store [param_0] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
                "exec 4/thread",
                "[<block 5>] %tid.x ≤ 31 | exec <= 1/thread | \
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
                "block 256×1×1 | pointer params assumed 32 B aligned | per CTA: flops f32 384; global store 4 B; param load 2,048 B",
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
            Some(
                "block 64×1×1 | pointer params assumed 32 B aligned | per CTA: global store <= 256 B; param load 512 B"
            )
        );
        let chain: String = (1..=9)
            .map(|i| format!("and.pred %p{i}, %p{}, %p{};\n", i - 1, i - 1))
            .collect();
        let src = format!(
            "{head}{chain}@%p9 st.global.u32 [%rd1], %r1;\n@%p0 st.global.u16 [%rd1+4], %rs1;\nret;\n}}\n"
        );
        assert_eq!(
            note(&src, &opts, ".entry").as_deref(),
            Some(
                "block 64×1×1 | pointer params assumed 32 B aligned | per CTA: global store <= 320 B; param load 512 B"
            )
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
            Some(
                "block 32×1×1 | pointer params assumed 32 B aligned | per CTA: global load 128 B, store 128 B; param load 256 B"
            )
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
            Some(
                "block 64×1×1 | pointer params assumed 32 B aligned | per CTA: global store 8 B; param load 512 B"
            )
        );
        let src = src.replace(".reqntid 64, 1, 1", ".reqntid 32, 2, 1");
        assert_eq!(
            note(&src, &opts, ".entry").as_deref(),
            Some(
                "block 32×2×1 | pointer params assumed 32 B aligned | per CTA: global store <= 256 B; param load 512 B"
            )
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
                "block 128×1×1 | pointer params assumed 32 B aligned | per CTA: global store (256·max(param_1, 1)) B; param load 1,536 B"
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
            Some(
                "block 64×1×1 | pointer params assumed 32 B aligned | per CTA: global load 512 B, store <= 2,048 B; param load 512 B"
            )
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
                 global store [4·%tid.x + param_0] sectors/req <= 4 (ideal <= 4)"
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
            ["global load [4·%tid.x + param_0] sectors/req 4 (ideal 4)"]
        );
        assert_eq!(
            sectors("%clock"),
            ["global load [4·%tid.x + param_0] sectors/req <= 4 (ideal <= 4)"]
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
            ["global load [4·%tid.x + param_0] sectors/req 2 (ideal 2)"]
        );
    }

    /// The global bytes per CTA the header of `src`'s loop at `label`
    /// prints under `opts`.
    fn loop_bytes(src: &str, opts: &AnalyzeOptions, label: &str) -> Option<String> {
        let header = (note(src, opts, label))
            .filter(|n| n.starts_with("loop: "))
            .expect("a loop header");
        (header.split(" | ")).find_map(|p| p.strip_prefix("global per CTA: ").map(str::to_owned))
    }

    /// A loop header's global bytes per CTA count every thread and trip
    /// once: 32 lanes × 4 B × 8 trips per load; the first sweeps 1,024 B
    /// in 4 or 5 sectors a trip, the second is one word in one sector.
    /// Guarded on `%tid.x != 0`, the second load runs on 31 lanes;
    /// guarded on a loaded value, on every lane as a bound; in a block a
    /// loaded value skips, a bound though not guarded. Without a block
    /// shape there are none.
    #[test]
    fn loop_bytes_count_every_thread_and_iteration_once() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0,\n.param .u64 k_param_1\n)\n\
                   .reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.param.u64 %rd5, [k_param_1];\n\
                   mov.u32 %r1, %tid.x;\nmul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                   mov.u32 %r2, 0;\nsetp.ne.s32 %p2, %r1, 0;\n\
                   $L__LOOP:\nld.global.f32 %f1, [%rd3];\nld.global.f32 %f2, [%rd5];\n\
                   add.s64 %rd3, %rd3, 128;\nadd.s32 %r2, %r2, 1;\nsetp.lt.u32 %p1, %r2, 8;\n\
                   @%p1 bra $L__LOOP;\nadd.f32 %f3, %f1, %f2;\nst.global.f32 [%rd5], %f3;\nret;\n}\n";
        assert_eq!(
            loop_bytes(src, &opts, "$L__LOOP:").as_deref(),
            Some("requested 2,048 B, unique 1,028 B, sectors 1,280 B")
        );
        let guarded = src.replace("ld.global.f32 %f2", "@%p2 ld.global.f32 %f2");
        assert_eq!(
            loop_bytes(&guarded, &opts, "$L__LOOP:").as_deref(),
            Some("requested 2,016 B, unique 1,028 B, sectors 1,280 B")
        );
        let loaded = guarded.replace(
            "setp.ne.s32 %p2, %r1, 0;",
            "ld.global.u32 %r9, [%rd5];\nsetp.ne.s32 %p2, %r9, 0;",
        );
        assert_eq!(
            loop_bytes(&loaded, &opts, "$L__LOOP:").as_deref(),
            Some("requested <= 2,048 B, unique <= 1,028 B, sectors <= 1,280 B")
        );
        let branched = src.replace(
            "ld.global.f32 %f2, [%rd5];",
            "ld.global.u32 %r9, [%rd5];\nsetp.ne.s32 %p3, %r9, 0;\n@%p3 bra $L__SKIP;\n\
             ld.global.f32 %f2, [%rd5];\n$L__SKIP:",
        );
        assert_eq!(
            loop_bytes(&branched, &opts, "$L__LOOP:").as_deref(),
            Some("requested <= 3,072 B, unique <= 1,028 B, sectors <= 1,536 B")
        );
        let src = src.replace(".reqntid 32, 1, 1\n", "");
        assert_eq!(loop_bytes(&src, &opts, "$L__LOOP:"), None);
    }

    /// A store at a floor of the counter, `⌊(iv + 1)/2⌋`, over 10^9
    /// trips: too many, so its loop header has no global bytes per CTA.
    #[test]
    fn a_floor_of_the_counter_over_too_many_trips_has_no_footprint() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, 0;\n\
                   $L__LOOP:\nadd.u32 %r1, %r1, 1;\nshr.u32 %r3, %r1, 1;\n\
                   mul.wide.u32 %rd2, %r3, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                   st.global.u32 [%rd3], 0;\nsetp.lt.u32 %p1, %r1, 1000000000;\n\
                   @%p1 bra $L__LOOP;\nret;\n}\n";
        assert_eq!(
            loop_bytes(src, &AnalyzeOptions::default(), "$L__LOOP:"),
            None
        );
    }

    /// `%r10` is a 64-bit `selp` of 0 and `q − p`, declared after a
    /// 32-bit `%r<2>` that does not hold it: `p + %r10` may be `q`, so
    /// the loop's distinct global bytes per CTA are 512 to 1,024 B.
    #[test]
    fn a_wide_cta_uniform_offset_names_no_pointer() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0,\n.param .u64 k_param_1\n)\n\
                   .reqntid 32, 1, 1\n{\n.reg .pred %p<3>;\n.reg .b32 %r<2>;\n\
                   .reg .b64 %r10;\n.reg .b32 %s<8>;\n.reg .b64 %rd<8>;\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.param.u64 %rd2, [k_param_1];\n\
                   mov.u32 %s1, %ctaid.x;\nsetp.eq.u32 %p2, %s1, 0;\n\
                   sub.s64 %rd7, %rd2, %rd1;\nselp.b64 %r10, 0, %rd7, %p2;\n\
                   mov.u32 %s2, %tid.x;\nmov.u32 %s3, 0;\n$L__LOOP:\n\
                   mad.lo.s32 %s4, %s3, 32, %s2;\nmul.wide.u32 %rd3, %s4, 4;\n\
                   add.s64 %rd4, %rd1, %r10;\nadd.s64 %rd5, %rd4, %rd3;\n\
                   st.global.u32 [%rd5], %s2;\nadd.s64 %rd6, %rd2, %rd3;\n\
                   st.global.u32 [%rd6], %s2;\nadd.s32 %s3, %s3, 1;\n\
                   setp.lt.u32 %p1, %s3, 4;\n@%p1 bra $L__LOOP;\nret;\n}\n";
        assert_eq!(
            loop_bytes(src, &AnalyzeOptions::default(), "$L__LOOP:").as_deref(),
            Some("requested 1,024 B, unique 512–1,024 B, sectors 1,024–1,152 B")
        );
    }

    /// A zero-source cp.async through a loaded pointer: its address is
    /// unknown, but it reads no global bytes, and the loop's global bytes
    /// per CTA stay known.
    #[test]
    fn an_access_of_no_bytes_leaves_a_loop_footprint_whole() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   .shared .align 16 .b8 dst[512];\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, dst;\nmov.u32 %r2, 0;\n\
                   $L__LOOP:\nld.global.u64 %rd4, [%rd1];\n\
                   cp.async.cg.shared.global [%r1], [%rd4], 16, 0;\n\
                   add.s32 %r2, %r2, 1;\nsetp.lt.u32 %p1, %r2, 2;\n@%p1 bra $L__LOOP;\n\
                   cp.async.wait_all;\nret;\n}\n";
        assert_eq!(
            loop_bytes(src, &AnalyzeOptions::default(), "$L__LOOP:").as_deref(),
            Some("requested 512 B, unique 8 B, sectors 64 B")
        );
    }

    /// A loop of 2^62 trips whose 1024 threads each read 4 bytes: the
    /// bytes it requests pass 2^64, so its header has no global bytes per
    /// CTA; at 4 trips, 32,768 B.
    #[test]
    fn loop_bytes_past_u64_are_unknown() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
            .visible .entry k(.param .u64 k_param_0, .param .u64 k_param_1)\n{\n\
            .reg .pred %p<2>;\n.reg .f32 %f<2>;\n.reg .b32 %r<2>;\n.reg .b64 %rd<6>;\n\
            ld.param.u64 %rd1, [k_param_0];\n\
            ld.param.u64 %rd2, [k_param_1];\n\
            mov.u32 %r1, %tid.x;\n\
            mul.wide.u32 %rd3, %r1, 4;\n\
            add.s64 %rd4, %rd1, %rd3;\n\
            mov.u64 %rd5, 0;\n\
            $L__L:\n\
            ld.global.f32 %f1, [%rd4];\n\
            st.global.f32 [%rd4], %f1;\n\
            add.s64 %rd5, %rd5, 1;\n\
            setp.lt.u64 %p1, %rd5, %rd2;\n\
            @%p1 bra $L__L;\n\
            ret;\n}\n";
        let bytes = |n: i64| {
            let opts = AnalyzeOptions {
                bindings: vec![parse_bind(&format!("1:N={n}")).expect("binding")],
                launch: Some([1024, 1, 1]),
                ..Default::default()
            };
            loop_bytes(src, &opts, "$L__L:")
        };
        assert_eq!(
            bytes(4).as_deref(),
            Some("requested 32,768 B, unique 4,096 B, sectors 32,768 B")
        );
        assert_eq!(bytes(1 << 62), None);
    }

    /// Each memory operand of `src`'s kernel, its space, direction and
    /// address as its line's annotation prints them, an `unused`
    /// instruction's too, whose line says only `unused`.
    fn addresses(src: &str, opts: &AnalyzeOptions) -> Vec<String> {
        (memory_operands(src, opts).into_iter())
            .map(|o| {
                let end = [" sectors/req", " wavefronts/req"]
                    .iter()
                    .filter_map(|r| o.find(r))
                    .min();
                o[..end.unwrap_or(o.len())].to_owned()
            })
            .collect()
    }

    /// `%r1` steps by 2^30 for 8 trips: `mul.wide.u32` reads its 32 bits,
    /// 0, 2^30, 2^31 and 3·2^30 twice over, which no multiple of the
    /// iteration gives, so the address is one unknown value for every
    /// thread on each trip; stepping by 1 it is the loop's iteration,
    /// `iv[$L__LOOP]`.
    #[test]
    fn a_counter_read_across_its_type_is_unknown() {
        let src = |step: &str| {
            format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 1, 1, 1\n{{\n\
                 ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, 0;\nmov.u32 %r5, 0;\n\
                 $L__LOOP:\nmul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                 st.global.u32 [%rd3], 0;\nadd.u32 %r1, %r1, {step};\n\
                 add.u32 %r5, %r5, 1;\nsetp.lt.u32 %p1, %r5, 8;\n@%p1 bra $L__LOOP;\nret;\n}}\n"
            )
        };
        let address = |step: &str| addresses(&src(step), &AnalyzeOptions::default());
        assert_eq!(address("1"), ["global store [4·iv[$L__LOOP] + param_0]"]);
        assert_eq!(
            address("1073741824"),
            ["global store [uniform(%rd2@13) + param_0]"]
        );
    }

    /// A shared load at `%tid.x` split by 8, its array by its demangled
    /// name, and a global load in a loop, moving 4 B a trip of `$L__L`.
    #[test]
    fn accesses_are_affine_addresses() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   .shared .align 2 .b8 _ZZ1kE2As[128];\n\
                   ld.param.u64 %rd1, [k_param_0];\ncvta.to.global.u64 %rd2, %rd1;\n\
                   mov.u32 %r1, %tid.x;\nshr.u32 %r2, %r1, 3;\nand.b32 %r3, %r1, 7;\n\
                   shl.b32 %r4, %r2, 4;\nshl.b32 %r5, %r3, 1;\nadd.s32 %r6, %r4, %r5;\n\
                   mov.u32 %r7, _ZZ1kE2As;\nadd.s32 %r8, %r7, %r6;\nld.shared.u16 %rs1, [%r8+2];\n\
                   mov.u64 %rd3, %rd2;\nmov.u32 %r9, 0;\n\
                   $L__L:\nld.global.f32 %f1, [%rd3];\nadd.s64 %rd3, %rd3, 4;\n\
                   add.s32 %r9, %r9, 1;\nsetp.lt.s32 %p1, %r9, 8;\n@%p1 bra $L__L;\nret;\n}\n";
        assert_eq!(
            addresses(src, &AnalyzeOptions::default()),
            [
                "shared load [16·⌊%tid.x/8⌋ + 2·(%tid.x mod 8) + As + 2]",
                "global load [4·iv[$L__L] + param_0]",
            ]
        );
    }

    /// A max of a parameter scaling the thread index divides once the
    /// parameter is bound: `(max(n, 1)·%tid.x) >> 2` at n = 2 is
    /// ⌊%tid.x/2⌋, and its `rem` by 4 is 2·(%tid.x mod 2).
    #[test]
    fn a_bound_max_coefficient_divides() {
        let opts = AnalyzeOptions {
            bindings: vec![parse_bind("1:n=2").expect("binding")],
            ..Default::default()
        };
        let src = |reduce: &str| {
            format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0,\n.param .u32 k_param_1\n)\n{{\n\
                 ld.param.u64 %rd1, [k_param_0];\nld.param.u32 %r1, [k_param_1];\n\
                 max.s32 %r2, %r1, 1;\nmov.u32 %r3, %tid.x;\nmul.lo.s32 %r4, %r2, %r3;\n\
                 {reduce}\nmul.wide.u32 %rd2, %r5, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                 st.global.u32 [%rd3], %r5;\nret;\n}}\n"
            )
        };
        let address = |reduce: &str| addresses(&src(reduce), &opts);
        assert_eq!(
            address("shr.u32 %r5, %r4, 2;"),
            ["global store [4·⌊%tid.x/2⌋ + param_0]"]
        );
        assert_eq!(
            address("rem.u32 %r5, %r4, 4;"),
            ["global store [8·(%tid.x mod 2) + param_0]"]
        );
    }

    /// `mov.b64 {%r2, %r3}, %rd1` writes %r2 as half of %rd1, not %rd1:
    /// an address through it is a uniform variable, not param_0.
    #[test]
    fn a_vector_destination_is_not_the_whole_source() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.b64 {%r2, %r3}, %rd1;\n\
                   cvt.u64.u32 %rd2, %r2;\nld.global.f32 %f1, [%rd2];\nret;\n}\n";
        assert_eq!(
            addresses(src, &AnalyzeOptions::default()),
            ["global load [uniform(%r2@9)]"]
        );
    }

    /// The base at depth 0, `%r30` at 2 and `%r0` at 32, the deepest
    /// definition affine analysis follows: `%tid.x` below it is a leaf.
    #[test]
    fn a_special_register_at_the_depth_limit_is_read() {
        let chain: String = (1..=30)
            .map(|i| format!("add.u32 %r{i}, %r{}, 1;\n", i - 1))
            .collect();
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u64 k_param_0\n)\n{{\n\
             ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r0, %tid.x;\n{chain}\
             mul.wide.u32 %rd2, %r30, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
             ld.global.f32 %f1, [%rd3];\nret;\n}}\n"
        );
        assert_eq!(
            addresses(&src, &AnalyzeOptions::default()),
            ["global load [4·%tid.x + param_0 + 120]"]
        );
    }

    /// PTX ISA §6.4.3: `{Ra, Rb, Rc, Rd}` take `V.x`, `V.y`, `V.z`, `V.w`;
    /// §10.1: `%tid`'s fourth element "always returns zero".
    #[test]
    fn a_special_vector_is_read_by_element() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\n\
                   mov.v4.u32 {%r1, %r2, %r3, %r4}, %tid;\nmov.u32 %r5, %tid.a;\n\
                   mul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd3, %rd1, %rd2;\nld.global.f32 %f1, [%rd3];\n\
                   mul.wide.u32 %rd4, %r4, 4;\nadd.s64 %rd5, %rd1, %rd4;\nld.global.f32 %f2, [%rd5];\n\
                   mul.wide.u32 %rd6, %r5, 4;\nadd.s64 %rd7, %rd1, %rd6;\nld.global.f32 %f3, [%rd7];\n\
                   ret;\n}\n";
        assert_eq!(
            addresses(src, &AnalyzeOptions::default()),
            [
                "global load [4·%tid.x + param_0]",
                "global load [param_0]",
                "global load [param_0]",
            ]
        );
    }

    /// `ld.param.v2.u32 {%r1, %r2}` loads two words of the parameter: %r2
    /// is the second, a uniform variable, not param_0.
    #[test]
    fn a_vector_parameter_load_element_is_not_the_parameter() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .align 8 .b8 k_param_0[8]\n)\n{\n\
                   ld.param.v2.u32 {%r1, %r2}, [k_param_0];\n\
                   cvt.u64.u32 %rd2, %r2;\nld.global.f32 %f1, [%rd2];\nret;\n}\n";
        assert_eq!(
            addresses(src, &AnalyzeOptions::default()),
            ["global load [uniform(%r2@8)]"]
        );
    }

    /// Both paths write `{%r2, %r3}` from one 64-bit immediate: %r3 is its
    /// upper half, not the immediate, the same for every thread, so the
    /// address is known.
    #[test]
    fn merged_vector_writes_of_one_immediate_are_not_that_immediate() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u32 k_param_0\n)\n{\n\
                   ld.param.u32 %r1, [k_param_0];\nsetp.eq.u32 %p1, %r1, 0;\n\
                   @%p1 bra $L1;\nmov.b64 {%r2, %r3}, 1;\nbra.uni $L2;\n\
                   $L1:\nmov.b64 {%r2, %r3}, 1;\n$L2:\n\
                   cvt.u64.u32 %rd2, %r3;\nld.global.f32 %f1, [%rd2];\nret;\n}\n";
        assert_eq!(
            addresses(src, &AnalyzeOptions::default()),
            ["global load [uniform(%r3@$L2)]"]
        );
    }

    /// An `or` whose constant is the high side: the lane part fits below
    /// it. No power of two fits above 2^62 in an i64: that order is
    /// refused and the other, the 0 below 1, still read.
    #[test]
    fn an_or_above_the_other_sides_largest_value_is_an_add() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n\
                   and.b32 %r2, %r1, 31;\nor.b32 %r3, %r2, 256;\nmul.wide.u32 %rd2, %r3, 4;\n\
                   add.s64 %rd3, %rd1, %rd2;\nld.global.f32 %f1, [%rd3];\nret;\n}\n";
        assert_eq!(
            addresses(src, &opts),
            ["global load [4·(%tid.x mod 32) + param_0 + 1024]"]
        );
        let wide = src.replace(
            "mov.u32 %r1, %tid.x;\n",
            "mov.u64 %rd8, 0x4000000000000000;\nor.b64 %rd9, %rd8, 0;\nadd.s64 %rd1, %rd1, %rd9;\nmov.u32 %r1, %tid.x;\n",
        );
        assert_eq!(
            addresses(&wide, &opts),
            ["global load [4·(%tid.x mod 32) + param_0 + 4611686018427388928]"]
        );
    }

    /// `src`'s kernel against `opts`' record, through `compare`.
    fn comparison(src: &str, opts: &AnalyzeOptions) -> Comparison {
        (compare(src, opts).expect("analyzes")).expect("the record's kernel")
    }

    /// A prediction past u64 is none: 32 threads in each of 2 CTAs run
    /// 2^62 + %ctaid.x trips, 2^63 + 1 over the grid. The loop header
    /// prints that count, and the record still compares.
    #[test]
    fn an_execution_count_past_u64_is_not_predicted() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %ctaid.x;\n\
                   cvt.u64.u32 %rd2, %r1;\nadd.s64 %rd3, %rd2, 4611686018427387904;\n\
                   mov.u64 %rd4, 0;\n$L__L:\nadd.s64 %rd4, %rd4, 1;\nst.global.u32 [%rd1], %r1;\n\
                   setp.lt.u64 %p1, %rd4, %rd3;\n@%p1 bra $L__L;\nret;\n}\n";
        let record = NcuRecord {
            kernel: "k".to_owned(),
            grid: [2, 1, 1],
            block: [32, 1, 1],
            params: vec!["ptr:4".to_owned()],
            lines: Default::default(),
            load_sectors: None,
            store_sectors: None,
            ops_path_tensor_src: None,
        };
        let opts = AnalyzeOptions {
            grid: Some([2, 1, 1]),
            ncu: Some(record),
            ..Default::default()
        };
        let header = note(src, &opts, "$L__L:").expect("a loop header");
        assert_eq!(
            header.split(" | ").next(),
            Some("loop: trips %ctaid.x + 4611686018427387904")
        );
        assert!(compare(src, &opts).expect("analyzes").is_some());
    }

    /// A per-trip launch count past u64, 32 threads of 3 CTAs on
    /// 2^58 - 1 of 2^58 trips, has no prediction: no line differs from
    /// the record.
    #[test]
    fn a_per_trip_launch_count_past_u64_has_no_prediction() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u64 %rd2, 0;\n$L__H:\n\
                   add.s64 %rd2, %rd2, 1;\nsetp.ge.u64 %p1, %rd2, 288230376151711744;\n\
                   @%p1 bra $L__T;\nbar.sync 0;\n$L__T:\n\
                   setp.lt.u64 %p2, %rd2, 288230376151711744;\n@%p2 bra $L__H;\nret;\n}\n";
        let line = |ptx: &str| NcuLine {
            ptx: ptx.to_owned(),
            sass: [("BAR.SYNC.DEFER_BLOCKING".to_owned(), 1)]
                .into_iter()
                .collect(),
            executions: 5,
            l2_theoretical_sectors_global: None,
            l2_theoretical_sectors_global_ideal: None,
            global_access_sizes: Vec::new(),
            global_executions: None,
            shared: Vec::new(),
        };
        let record = NcuRecord {
            kernel: "k".to_owned(),
            grid: [3, 1, 1],
            block: [32, 1, 1],
            params: vec!["ptr:4".to_owned()],
            lines: [(15, line("bar.sync 0;"))].into_iter().collect(),
            load_sectors: None,
            store_sectors: None,
            ops_path_tensor_src: None,
        };
        let opts = AnalyzeOptions {
            grid: Some([3, 1, 1]),
            ncu: Some(record),
            ..Default::default()
        };
        let predicted: Vec<u64> = (comparison(src, &opts).blocks.iter())
            .flat_map(|b| b.differing.iter().map(|l| l.predicted))
            .collect();
        assert_eq!(predicted, Vec::<u64>::new());
    }

    /// A launch of 2^31 − 1 by 65535 by 65535 CTAs, CUDA's largest grid,
    /// of 32 threads: its stores over the launch pass 2^64, so the
    /// record's line has no prediction instead of a wrapped one.
    #[test]
    fn a_count_over_the_launch_past_u64_has_no_prediction() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
            .visible .entry k(.param .u64 k_param_0)\n{\n\
            .reg .f32 %f<2>;\n.reg .b64 %rd<2>;\n\
            ld.param.u64 %rd1, [k_param_0];\n\
            st.global.f32 [%rd1], %f1;\n\
            ret;\n}\n";
        let line = NcuLine {
            ptx: "st.global.f32 [%rd1], %f1;".into(),
            sass: [("STG.E".to_owned(), 1)].into(),
            executions: 32,
            l2_theoretical_sectors_global: Some(1),
            l2_theoretical_sectors_global_ideal: Some(1),
            global_access_sizes: vec![32],
            global_executions: Some(32),
            shared: Vec::new(),
        };
        let record = NcuRecord {
            kernel: "k".into(),
            grid: [2147483647, 65535, 65535],
            block: [32, 1, 1],
            params: vec!["ptr:4".into()],
            lines: [(9, line)].into(),
            load_sectors: None,
            store_sectors: None,
            ops_path_tensor_src: None,
        };
        let opts = AnalyzeOptions {
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        let blocks = comparison(src, &opts).blocks;
        assert_eq!(blocks[0].with_sass, 1);
        assert_eq!(blocks[0].agreeing, 0);
        assert!(blocks[0].differing.is_empty() && blocks[0].sectors.is_empty());
    }

    /// CUDA's grid takes 2^30 by 32768 by 8192 CTAs: 32 threads each
    /// store 2^63 times over the launch, which a u64 prediction holds.
    #[test]
    fn a_launch_count_between_2_63_and_2_64_is_predicted() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
            .visible .entry k(.param .u64 k_param_0)\n{\n\
            .reg .f32 %f<2>;\n.reg .b64 %rd<2>;\n\
            ld.param.u64 %rd1, [k_param_0];\n\
            st.global.f32 [%rd1], %f1;\n\
            ret;\n}\n";
        let line = NcuLine {
            ptx: "st.global.f32 [%rd1], %f1;".into(),
            sass: [("STG.E".to_owned(), 1)].into(),
            executions: 1 << 63,
            l2_theoretical_sectors_global: Some(1),
            l2_theoretical_sectors_global_ideal: Some(1),
            global_access_sizes: vec![32],
            global_executions: Some(32),
            shared: Vec::new(),
        };
        let record = NcuRecord {
            kernel: "k".into(),
            grid: [1 << 30, 32768, 8192],
            block: [32, 1, 1],
            params: vec!["ptr:4".into()],
            lines: [(9, line)].into(),
            load_sectors: None,
            store_sectors: None,
            ops_path_tensor_src: None,
        };
        let opts = AnalyzeOptions {
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        let blocks = comparison(src, &opts).blocks;
        assert_eq!(blocks[0].with_sass, 1);
        assert_eq!(blocks[0].agreeing, 1);
        assert!(blocks[0].differing.is_empty());
    }

    /// Two nested loops of 2^29 trips each on one CTA of 32 threads: the
    /// store runs 2^58 times per thread and 2^63 times over the launch,
    /// past i64 before the CTAs multiply in, which a u64 prediction holds.
    #[test]
    fn trips_times_threads_between_2_63_and_2_64_is_predicted() {
        let src = ".version 8.7\n.target sm_89\n.address_size 64\n\
            .visible .entry k(.param .u64 k_param_0, .param .u32 k_param_1)\n{\n\
            .reg .pred %p<3>;\n.reg .b32 %r<4>;\n.reg .b64 %rd<2>;\n\
            ld.param.u64 %rd1, [k_param_0];\n\
            ld.param.u32 %r1, [k_param_1];\n\
            mov.u32 %r2, 0;\n\
            $L__O:\n\
            mov.u32 %r3, 0;\n\
            $L__I:\n\
            st.global.u32 [%rd1], %r3;\n\
            add.s32 %r3, %r3, 1;\n\
            setp.lt.u32 %p1, %r3, %r1;\n\
            @%p1 bra $L__I;\n\
            add.s32 %r2, %r2, 1;\n\
            setp.lt.u32 %p2, %r2, %r1;\n\
            @%p2 bra $L__O;\n\
            ret;\n}\n";
        let line = NcuLine {
            ptx: "st.global.u32 [%rd1], %r3;".into(),
            sass: [("STG.E".to_owned(), 1)].into(),
            executions: 1 << 63,
            l2_theoretical_sectors_global: None,
            l2_theoretical_sectors_global_ideal: None,
            global_access_sizes: Vec::new(),
            global_executions: None,
            shared: Vec::new(),
        };
        let record = NcuRecord {
            kernel: "k".into(),
            grid: [1, 1, 1],
            block: [32, 1, 1],
            params: vec!["ptr:4".into(), format!("u32:{}", 1 << 29)],
            lines: [(15, line)].into(),
            load_sectors: None,
            store_sectors: None,
            ops_path_tensor_src: None,
        };
        let opts = AnalyzeOptions {
            bindings: record.bindings(),
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        let blocks = comparison(src, &opts).blocks;
        let inner = blocks
            .iter()
            .find(|b| b.name == "$L__I")
            .expect("inner loop");
        assert_eq!((inner.with_sass, inner.agreeing), (1, 1));
    }

    /// A scope holding an unclassified instruction or an unquantified
    /// byte count marks its totals, the `.entry` line's per thread, as
    /// lower bounds: with `wgmma`, the CUDA cores' FLOPs past f32's 2,
    /// the other pipes' and the global and shared bytes; with a
    /// `cp.async` of a register's size alone, the global bytes it loads
    /// and the shared bytes it stores.
    #[test]
    fn unknowns_in_scope_make_totals_lower_bounds() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_90a\n.address_size 64\n\
                   .visible .entry k()\n{\n\
                   ld.global.f32 %f1, [%rd1];\nfma.rn.f32 %f2, %f1, %f1, %f1;\n\
                   st.global.f32 [%rd1], %f2;\nwgmma.fence.sync.aligned;\n\
                   cp.async.cg.shared.global [%r1], [%rd2], %r2;\nret;\n}\n";
        assert_eq!(
            note(src, &opts, ".entry").as_deref(),
            Some(
                "per thread: flops f32 2 + ?; tensor, sfu, atomic flops ?; \
                 global load >= 4 B, store >= 4 B; shared load ? B, store ? B"
            )
        );
        let src = src.replace("wgmma.fence.sync.aligned;\n", "");
        assert_eq!(
            note(&src, &opts, ".entry").as_deref(),
            Some("per thread: flops f32 2; global load >= 4 B, store 4 B; shared store ? B")
        );
    }

    /// Static shared memory per CTA = Σ (element count × element width)
    /// over `.shared` decls; `.extern .shared` is dynamic, not counted:
    /// the `.entry` line's `shared memory`, `+ dynamic` where the launch
    /// sets more, none without either.
    #[test]
    fn shared_memory_static_and_dynamic() {
        let opts = AnalyzeOptions::default();
        let body = |decls: &str| {
            format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k()\n{{\n{decls}ret;\n}}\n"
            )
        };
        let sm = |decls: &str| note(&body(decls), &opts, ".entry");

        // The k5 shape: two byte arrays, 1024 + 1024 = 2048 (matches the
        // `2048 bytes smem` ptxas reports for the k5 fixture).
        assert_eq!(
            sm(".shared .align 2 .b8 As[1024];\n.shared .align 2 .b8 Bs[1024];\n").as_deref(),
            Some("shared memory 2,048 B/CTA")
        );
        // Typed array: per-element width applies, 256 × 4 = 1024.
        assert_eq!(
            sm(".shared .align 4 .f32 buf[256];\n").as_deref(),
            Some("shared memory 1,024 B/CTA")
        );
        // Width coverage: b64 → 8, b128 → 16, 4 × 8 + 2 × 16.
        assert_eq!(
            sm(".shared .b64 w[4];\n.shared .b128 q[2];\n").as_deref(),
            Some("shared memory 64 B/CTA")
        );
        // A scalar reserves its type's size.
        assert_eq!(
            sm(".shared .align 8 .b64 bar;\n").as_deref(),
            Some("shared memory 8 B/CTA")
        );
        // The k1 shape: no shared memory at all.
        assert_eq!(sm(""), None);
        // Dynamic extern shared: out of the static total, flagged.
        assert_eq!(
            sm(".extern .shared .align 8 .b8 dsm[];\n").as_deref(),
            Some("shared memory 0 B/CTA + dynamic")
        );
        // Mixed: static counted, dynamic flagged.
        assert_eq!(
            sm(".shared .align 2 .b8 As[1024];\n.extern .shared .align 8 .b8 dsm[];\n").as_deref(),
            Some("shared memory 1,024 B/CTA + dynamic")
        );
        // LLVM's module-scope form counts for every kernel in the module.
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .extern .shared .align 16 .b8 global_smem[];\n\
                   .visible .entry k()\n{\nret;\n}\n";
        assert_eq!(
            note(src, &opts, ".entry").as_deref(),
            Some("shared memory 0 B/CTA + dynamic")
        );
    }

    /// Each statement the parser cannot read says on its line that it is
    /// not counted.
    #[test]
    fn unparsed_statements_say_they_are_not_counted() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k()\n{\n@@ not ptx;\n.bogus 1;\nret;\n}\n";
        assert_eq!(
            notes(src, &AnalyzeOptions::default()),
            ["not parsed (not counted)", "not parsed (not counted)"]
        );
    }
}

#[cfg(test)]
mod invariants {
    //! Verifier identities: any correct implementation satisfies them, with
    //! no per-test expectations.
    //!
    //! 1. Every Measurement's provenance index resolves to a real
    //!    instruction.
    //! 2. The blocks' instructions sum to the kernel's instruction count.
    //! 3. Two-path consistency: with every parameter bound, the FLOPs the
    //!    annotated PTX's `.entry` line sums per thread equal an
    //!    independently computed sum (per-block flat tallies ×
    //!    numerically evaluated trip chains), the check that catches two
    //!    code paths disagreeing.
    //! 4. The annotated PTX of every fixture, every bound case of 3 and every
    //!    Nsight Compute record is the input, line for line, each line
    //!    followed by nothing or by ` // ` and its annotation.

    use crate::analysis::affine::affine_analysis::AffineAnalysis;
    use crate::analysis::affine::sym_expr::SymbolicConstant;
    use crate::analysis::affine::trip_counts::trip_counts;
    use crate::analysis::control_flow::loop_forest;
    use crate::analysis::instruction_counts::collect::collect;
    use crate::analysis::instruction_counts::measurement::MeasureKind;
    use crate::analysis::instruction_counts::measurement::Precision;
    use crate::analysis::kernel_analyses::KernelAnalyses;
    use crate::analysis::uniformity_info::UniformityInfo;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;
    use crate::report::annotate::annotate;
    use crate::report::measured::NcuRecord;
    use crate::report::{AnalyzeOptions, BindingSpec};
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    const FIXTURES: &[&str] = &[
        "k1/k1.sm_80.ptx",
        "k1/k1.sm_89.ptx",
        "k2/k2.sm_80.ptx",
        "k5/k5.sm_80.ptx",
        "k5/k5.sm_89.ptx",
        "k11/k11.sm_80.ptx",
        "k12/k12.sm_80.ptx",
        "k14/k14.sm_80.ptx",
        "mma_demo/mma_demo.sm_80.ptx",
        "micro/single_loop.ptx",
        "micro/branchy.ptx",
        "micro/irreducible.ptx",
        "micro/no_loc.ptx",
        "micro/data_dep.ptx",
    ];

    fn read(fixture: &str) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        fs::read_to_string(&path).expect("fixture readable")
    }

    #[test]
    fn every_measurement_provenance_resolves_to_an_instruction() {
        for fixture in FIXTURES {
            let src = read(fixture);
            let m = parse(&src).expect("fixture parses");
            for k in &m.kernels {
                let cfg = build_cfg(&m, k);
                let f = loop_forest(&cfg);
                let data = KernelAnalyses::new(&m, k, &cfg, &f);
                let uniformity = UniformityInfo::new(&data, data.kernel.reqntid);
                for bm in collect(&AffineAnalysis::new(&data, &uniformity, None)) {
                    for meas in &bm.measurements {
                        assert!(
                            matches!(k.stmts.get(meas.provenance), Some(Stmt::Instr(_))),
                            "{fixture}: provenance {} is not an instruction",
                            meas.provenance
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn block_instructions_sum_to_kernel_instruction_count() {
        for fixture in FIXTURES {
            let src = read(fixture);
            let m = parse(&src).expect("fixture parses");
            for k in &m.kernels {
                let cfg = build_cfg(&m, k);
                let f = loop_forest(&cfg);
                let data = KernelAnalyses::new(&m, k, &cfg, &f);
                let uniformity = UniformityInfo::new(&data, data.kernel.reqntid);
                let from_blocks: usize = collect(&AffineAnalysis::new(&data, &uniformity, None))
                    .iter()
                    .map(|b| b.instructions.len())
                    .sum();
                let from_stmts = k
                    .stmts
                    .iter()
                    .filter(|s| matches!(s, Stmt::Instr(_)))
                    .count();
                assert_eq!(from_blocks, from_stmts, "{fixture}");
            }
        }
    }

    /// Fixtures with fully resolvable trips and one symbolic parameter.
    const BOUND: &[(&str, usize)] = &[
        ("k1/k1.sm_80.ptx", 2),
        ("k1/k1.sm_89.ptx", 2),
        ("k2/k2.sm_80.ptx", 2),
        ("k5/k5.sm_80.ptx", 2),
        ("k5/k5.sm_89.ptx", 2),
        ("k11/k11.sm_80.ptx", 2),
        ("k12/k12.sm_80.ptx", 2),
        ("k14/k14.sm_80.ptx", 2),
        ("micro/single_loop.ptx", 1),
        ("micro/branchy.ptx", 1),
    ];
    const BOUND_VALUE: i64 = 4099; // deliberately not a multiple of the unrolls

    fn bound(param: usize) -> AnalyzeOptions {
        AnalyzeOptions {
            bindings: vec![BindingSpec {
                index: Some(param),
                name: "N".into(),
                value: BOUND_VALUE,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn bound_flop_totals_agree_with_an_independent_evaluation() {
        for &(fixture, param) in BOUND {
            let src = read(fixture);
            let value = BOUND_VALUE;

            // Path 1: the annotated PTX's `.entry` line.
            let annotated = annotate(&src, &bound(param)).expect("annotates");
            let (reported, cuda_core, f32_only) = entry_flops(&annotated);

            // Path 2: flat per-block tallies × numerically evaluated chains.
            let m = parse(&src).expect("parses");
            let k = &m.kernels[0];
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            let info = trip_counts(&m, k, &cfg, &f, &Default::default());
            let bind_map: HashMap<SymbolicConstant, i64> =
                [(SymbolicConstant::KernelParam(param), value)]
                    .into_iter()
                    .collect();
            let trips_num: Vec<i64> = info
                .trips
                .iter()
                .map(|t| match t {
                    Ok(e) => e.bind(&bind_map).as_const().expect("trips fully bound"),
                    Err(_) => panic!("{fixture}: unexpected unknown trips"),
                })
                .collect();
            let data = KernelAnalyses::new(&m, k, &cfg, &f);
            let uniformity = UniformityInfo::new(&data, data.kernel.reqntid);
            let blocks = collect(&AffineAnalysis::new(&data, &uniformity, None));
            let mut independent = 0i64;
            for bm in &blocks {
                let mut mult = 1i64;
                let mut cur = f.block_loop[bm.block.0 as usize];
                while let Some(l) = cur {
                    mult *= trips_num[l.0 as usize];
                    cur = f.get(l).parent;
                }
                let flat: u64 = bm
                    .measurements
                    .iter()
                    .filter(|m| matches!(m.kind, MeasureKind::Flops { .. }))
                    .map(|m| m.count)
                    .sum();
                independent += flat as i64 * mult;
            }
            assert_eq!(reported, independent, "{fixture}: two paths disagree");
            // Sanity: cuda-core work in this corpus is all f32.
            assert_eq!(f32_only, cuda_core, "{fixture}");
        }
    }

    /// The FLOPs the `.entry` line of `annotated` sums per thread, a bound's
    /// value: of every pipe and precision, of the cuda-core pipe, and of its
    /// f32.
    fn entry_flops(annotated: &str) -> (i64, i64, i64) {
        let entry = (annotated.lines())
            .find(|l| l.contains(".entry "))
            .expect("an .entry line");
        let sums = (entry.split(" | "))
            .find_map(|p| p.strip_prefix("per thread: "))
            .expect("per-thread sums");
        let (mut all, mut cuda_core, mut f32) = (0, 0, 0);
        for item in sums.split("; ") {
            let Some((pipe, counts)) = item.split_once("flops ") else {
                continue;
            };
            for count in counts.split(" + ") {
                let (precision, n) = count.split_once(' ').expect("a precision and a count");
                let n = n.trim_start_matches("<= ").replace(',', "");
                let n: i64 = n.parse().expect("a bound count");
                all += n;
                if pipe.is_empty() {
                    cuda_core += n;
                    f32 += if precision == Precision::F32.key() {
                        n
                    } else {
                        0
                    };
                }
            }
        }
        (all, cuda_core, f32)
    }

    fn files(dir: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("directory readable") {
            let path = entry.expect("entry readable").path();
            if path.is_dir() {
                files(&path, suffix, out);
            } else if path.to_string_lossy().ends_with(suffix) {
                out.push(path);
            }
        }
    }

    /// Every fixture unbound, the bound cases of 3, and every Nsight Compute
    /// record with its PTX.
    fn every_run() -> Vec<(PathBuf, AnalyzeOptions)> {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut runs: Vec<(PathBuf, AnalyzeOptions)> = Vec::new();
        let mut ptx = Vec::new();
        files(&repo.join("tests/fixtures"), ".ptx", &mut ptx);
        runs.extend(ptx.into_iter().map(|p| (p, AnalyzeOptions::default())));
        runs.extend(
            (BOUND.iter()).map(|&(f, param)| (repo.join("tests/fixtures").join(f), bound(param))),
        );
        let mut records = Vec::new();
        files(&repo.join("tests/fixtures"), ".ncu.json", &mut records);
        files(&repo.join("tests/lit/Inputs"), ".ncu.json", &mut records);
        for path in records {
            let text = fs::read_to_string(&path).expect("record readable");
            let fields: serde_json::Value = serde_json::from_str(&text).expect("record is JSON");
            let record: NcuRecord = serde_json::from_str(&text).expect("record parses");
            let opts = AnalyzeOptions {
                bindings: record.bindings(),
                launch: Some(record.block),
                grid: Some(record.grid),
                ncu: Some(record),
                ..Default::default()
            };
            runs.push((
                repo.join(fields["ptx"].as_str().expect("record names its ptx")),
                opts,
            ));
        }
        assert!(runs.len() > 40, "{} runs", runs.len());
        runs
    }

    #[test]
    fn annotated_ptx_is_the_input_with_a_comment_on_some_lines() {
        for (path, opts) in &every_run() {
            let src = fs::read_to_string(path).expect("ptx readable");
            let out = annotate(&src, opts).expect("annotates");
            let (input, output): (Vec<&str>, Vec<&str>) =
                (src.split('\n').collect(), out.split('\n').collect());
            assert_eq!(input.len(), output.len(), "{}", path.display());
            for (i, o) in input.iter().zip(&output) {
                let rest = o.strip_prefix(i).expect("the input line");
                assert!(
                    rest.is_empty() || rest.starts_with(" // "),
                    "{}: {o}",
                    path.display()
                );
            }
        }
    }
}
