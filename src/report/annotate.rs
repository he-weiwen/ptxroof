//! The input PTX with what the analysis says of each line appended to
//! it as a ` // ` comment.

use crate::analysis::affine::affine_expr::{AffineExpr, Cmp, Var};
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::unknown_reason::UnknownReason;
use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::execution_counts::{CtaSize, Participation, Per, Resolution};
use crate::analysis::instruction_counts::collect::BlockMeasurements;
use crate::analysis::memory_footprint::CountRange;
use crate::analysis::register_effects::writes;
use crate::analysis::thread_participation::{Atom, AtomKey, Pred};
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
    /// label's line, else on its first statement's after `[<block name>] `,
    /// then each statement's on its own line, `unused` alone where nothing
    /// with an effect reads its results. Counts are over the launch given
    /// the grid and the block shape, else per CTA given the shape, else
    /// per thread.
    fn annotations(&self, notes: &mut BTreeMap<u32, Vec<String>>) {
        if let Some(launch) = self.launch() {
            let line = self.lines.line(self.kernel.offset);
            notes.entry(line).or_default().push(launch);
        }
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
        for bm in &self.blocks {
            let (b, start) = (bm.block, self.cfg.block(bm.block).start);
            let header = self.forest.block_loop[b.0 as usize]
                .filter(|&l| self.forest.get(l).header == b)
                .map(|l| format!("loop: trips {}", self.trips(l)));
            let (exec, mut parts) = self.execs(bm, per);
            for s in bm.instructions.iter().map(|i| i.provenance) {
                if self.counts.unused[s] {
                    parts.insert(s, vec!["unused".to_owned()]);
                } else {
                    parts.entry(s).or_default().extend(self.operands(s));
                }
            }
            parts.retain(|_, p| !p.is_empty());
            let when = self.runs_when(b);
            let runs = self.counts.sets.blocks[b.0 as usize] != Pred::False;
            let nesting = runs.then(|| self.nesting(b, when.is_some())).flatten();
            let mut own: Vec<String> = header
                .into_iter()
                .chain(self.threads(b))
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

    /// What the counts take of the launch, those given: `grid X×Y×Z`,
    /// `block X×Y×Z` and each parameter's value.
    fn launch(&self) -> Option<String> {
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
        ]
        .into_iter()
        .flatten()
        .collect();
        (!parts.is_empty()).then(|| parts.join(" | "))
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
    /// the scale `per`, `<=` for a bound, unless no thread runs it; and
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
        (Some(own), parts)
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
                    format!(" {what}/req {bound}{} (ideal {})", range(n), range(ideal))
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

    /// The threads that run block `b`, by warps and lanes; without a block
    /// shape, the condition that selects them, if one does. Where they
    /// read the CTA index and the grid is known: the lanes and the CTAs'
    /// condition as `when` prints it, or where the lanes differ by CTA
    /// the most CTAs' lanes.
    fn threads(&self, b: BlockId) -> Option<String> {
        let set = &self.counts.sets.blocks[b.0 as usize];
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
        let sets = &self.counts.sets;
        let has = |d: BlockId| {
            let i = d.0 as usize;
            sets.blocks[i] != Pred::False && !sets.relative[i].atoms().is_empty()
        };
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
    /// dominator, by `atom`, joined by ` and `, a conjunct without one
    /// left to the threads shown; an `or` or a `not` holding one with all
    /// its parts (`whole`), the `not` of an `and` or an `or` as the `or` or
    /// the `and` of its parts' negations, a thread-index comparison by
    /// `formula` and any other unknown as `?`; an `or` among the parts of an `and`, and an
    /// `and` among those of an `or`, parenthesized. Each unknown adds the
    /// line of the branch deciding it and its scope to `tags`.
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
            Pred::Or(..) | Pred::Not(_) if !whole && p.atoms().is_empty() => None,
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
            Pred::Cmp(c) if whole => Some(self.formula(&c.form, c.cmp)),
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
    /// its count per thread.
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
        let out = annotate(src, &AnalyzeOptions::default()).expect("annotates");
        let notes: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split_once(" // "))
            .map(|(_, n)| n)
            .collect();
        assert_eq!(
            notes,
            [
                "[<block 0>] exec 1/thread",
                "loop: trips 4 | exec 4/thread",
                "[<block 2>] %tid.x - (iv[$L__BB0_1] + 1) < 0 | exec <= 4/thread | \
                 global store [param_0] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
                "exec 4/thread",
                "[<block 4>] exec 1/thread",
                "[<block 5>] %tid.x <= 31 | exec <= 1/thread | \
                 global store [param_0] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
                "exec 1/thread",
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
}
