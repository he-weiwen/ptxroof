//! Build the result tree for one module.
//!
//! Aggregation model: a block's contribution to an aggregate is its
//! per-execution tally times the product of the trip counts of the
//! loops between it and the aggregation root (exclusive). A loop with
//! unresolved trips contributes through a *named opaque symbol*
//! `trips(<loop>)` — totals stay symbolic, never silently zero (S9.1).
//!
//! `at_least` propagation: a scope holding an instruction the classifier
//! does not model marks every flop table and the global and shared byte
//! totals as lower bounds (an unmodelled family may compute on any pipe
//! or move bytes in either space); an unquantified byte count marks its
//! own space and direction.
//!
//! `at_most` propagation: a tally is an upper bound if any contributing
//! block is conditional within its innermost scope (PR 09 rule), any
//! contributing instruction is predicated, or any loop on the
//! multiplier chain is *conditionally entered* (its header does not
//! dominate the enclosing scope's latch/exits — e.g. the whole k2 body
//! behind the bounds guard). Per-iteration views of a loop only look
//! below that loop, which is why a guarded kernel still has exact
//! per-iteration numbers — the altitude where the verdict lives.

use crate::cfg::loops::{LoopForest, LoopId};
use crate::cfg::naming::{LoopName, basename, demangle, loop_names};
use crate::cfg::{BlockId, Cfg, build_cfg, loop_forest};
use crate::classify::{ArithKind, Direction, OpClass, Pipe, Precision, Space};
use crate::core::measurement::MeasureKind;
use crate::core::symexpr::SymExpr;
use crate::core::{Instr, Kernel, Module, Stmt};
use crate::parse::parser::{ParseError, parse};
use crate::report::collect::{BlockMeasurements, CountQualifier, collect};
use crate::report::tree::*;
use crate::trips::{TripInfo, trip_counts};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, thiserror::Error)]
pub enum AnalyzeError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("bad --bind: {0}")]
    Binding(String),
}

/// One `--bind` argument: `name=value` or `idx:name=value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingSpec {
    pub index: Option<usize>,
    pub name: String,
    pub value: i64,
}

pub fn parse_bind(text: &str) -> Result<BindingSpec, String> {
    let (lhs, value) = text
        .split_once('=')
        .ok_or_else(|| format!("`{text}`: expected name=value or idx:name=value"))?;
    let value: i64 = value
        .parse()
        .map_err(|_| format!("`{text}`: value `{value}` is not an integer"))?;
    let (index, name) = match lhs.split_once(':') {
        Some((idx, name)) => {
            let idx = idx
                .parse()
                .map_err(|_| format!("`{text}`: index `{idx}` is not a number"))?;
            (Some(idx), name)
        }
        None => (None, lhs),
    };
    if name.is_empty() {
        return Err(format!("`{text}`: empty parameter name"));
    }
    Ok(BindingSpec {
        index,
        name: name.to_owned(),
        value,
    })
}

/// Caller-supplied analysis inputs (bet 4: launch config and
/// bindings are inputs, never guesses).
#[derive(Debug, Default)]
pub struct AnalyzeOptions {
    pub bindings: Vec<BindingSpec>,
    /// `--launch x,y,z`; `None` = default to `.reqntid`/`.maxntid`
    /// when the kernel carries one.
    pub launch: Option<[u32; 3]>,
}

pub fn analyze(
    source: &str,
    input_name: &str,
    opts: &AnalyzeOptions,
) -> Result<Report, AnalyzeError> {
    let module = parse(source)?;

    let mut kernels = Vec::new();
    let mut classified = Fraction { num: 0, den: 0 };
    let mut trips_resolved = Fraction { num: 0, den: 0 };
    let mut bindings_echo = Vec::new();

    for kernel in &module.kernels {
        let bind_map = resolve_bindings(&module, kernel, &opts.bindings, &mut bindings_echo)?;
        let k = KernelBuilder::new(&module, kernel, &bind_map).build(opts.launch);
        classified.num += k.instruction_classes.total - k.instruction_classes.unknown;
        classified.den += k.instruction_classes.total;
        let mut count_loops = |nodes: &[LoopNode]| {
            fn walk(nodes: &[LoopNode], f: &mut Fraction) {
                for n in nodes {
                    f.den += 1;
                    f.num += u64::from(n.trips.expr.is_some());
                    walk(&n.loops, f);
                }
            }
            walk(nodes, &mut trips_resolved);
        };
        count_loops(&k.loops);
        kernels.push(k);
    }

    let mut coverage = BTreeMap::new();
    coverage.insert("instructions_classified".to_owned(), classified);
    coverage.insert("loop_trips_resolved".to_owned(), trips_resolved);

    Ok(Report {
        input: input_name.to_owned(),
        bindings: bindings_echo,
        kernels,
        coverage,
    })
}

fn resolve_bindings(
    module: &Module,
    kernel: &Kernel,
    binds: &[BindingSpec],
    echo: &mut Vec<Binding>,
) -> Result<HashMap<String, i64>, AnalyzeError> {
    let mut map = HashMap::new();
    for spec in binds {
        let index = match spec.index {
            Some(i) => i,
            None => {
                // Name-only form: positional `param_N`, or the PTX
                // parameter name itself.
                if let Some(n) = spec
                    .name
                    .strip_prefix("param_")
                    .and_then(|s| s.parse().ok())
                {
                    n
                } else if let Some(i) = kernel
                    .params
                    .iter()
                    .position(|p| module.interner.resolve(p.name) == spec.name)
                {
                    i
                } else {
                    return Err(AnalyzeError::Binding(format!(
                        "`{}` names no parameter; use idx:name=value (params are positional)",
                        spec.name
                    )));
                }
            }
        };
        if index >= kernel.params.len() {
            return Err(AnalyzeError::Binding(format!(
                "param index {index} out of range ({} params)",
                kernel.params.len()
            )));
        }
        map.insert(format!("param_{index}"), spec.value);
        if !echo.iter().any(|b: &Binding| b.param == index) {
            echo.push(Binding {
                param: index,
                name: spec.name.clone(),
                value: spec.value,
            });
        }
    }
    Ok(map)
}

/// The direction a count is known in; `None` when bounded in neither.
fn direction(at_most: bool, at_least: bool) -> Option<Bound> {
    match (at_most, at_least) {
        (false, false) => Some(Bound::Exact),
        (true, false) => Some(Bound::AtMost),
        (false, true) => Some(Bound::AtLeast),
        (true, true) => None,
    }
}

/// Direction of `flops / bytes`: the byte bound inverts, and the two
/// sides must agree.
fn ratio_bound(flops: Option<Bound>, bytes: Option<Bound>) -> Option<Bound> {
    let inverted = match bytes? {
        Bound::Exact => Bound::Exact,
        Bound::AtMost => Bound::AtLeast,
        Bound::AtLeast => Bound::AtMost,
    };
    match (flops?, inverted) {
        (a, Bound::Exact) => Some(a),
        (Bound::Exact, b) => Some(b),
        (a, b) if a == b => Some(a),
        _ => None,
    }
}

/// Symbolic sum that collects like terms: contributions are grouped
/// by their constant-free multiplier, so two blocks under the same
/// trip chain merge into one term ("4 * param_1", never
/// "2 * param_1 + 2 * param_1"). `at_most` is the OR over
/// contributions — an upper bound on any term makes the sum one.
#[derive(Default)]
struct TermSum {
    groups: Vec<(SymExpr, i64)>,
    at_most: bool,
    at_least: bool,
    touched: bool,
}

impl TermSum {
    fn add(&mut self, count: i64, mult: &SymExpr, at_most: bool) {
        let (coeff, rest) = split_const(mult.clone());
        match self.groups.iter_mut().find(|(m, _)| *m == rest) {
            Some((_, n)) => *n += coeff * count,
            None => self.groups.push((rest, coeff * count)),
        }
        self.at_most |= at_most;
        self.touched = true;
    }

    fn add_unknown(&mut self) {
        self.at_least = true;
        self.touched = true;
    }

    fn expr(&self) -> SymExpr {
        self.groups
            .iter()
            .fold(SymExpr::Const(0), |acc, (mult, n)| {
                SymExpr::add(acc, SymExpr::mul(SymExpr::Const(*n), mult.clone()))
            })
    }

    fn count(&self) -> Count {
        Count {
            expr: self.expr().to_string(),
            at_most: self.at_most,
            at_least: self.at_least,
        }
    }
}

/// The report's name for an instruction class (see
/// [`InstructionCounts::by_kind`]).
fn instruction_kind(class: OpClass) -> String {
    let dir = |d: Direction| match d {
        Direction::Load => "load",
        Direction::Store => "store",
    };
    let width = |b: Option<u32>| b.map_or("? B".to_owned(), |b| format!("{b} B"));
    match class {
        OpClass::Flop {
            pipe, precision, ..
        } => format!("{} {}", pipe.key(), precision.key()),
        OpClass::Memory {
            space,
            direction,
            bytes,
        } => format!("{} {} {}", space.key(), dir(direction), width(bytes)),
        OpClass::Copy {
            from,
            to,
            written_bytes,
            ..
        } if from == to => format!("{} atomic {}", from.key(), width(written_bytes)),
        OpClass::Copy {
            from,
            to,
            written_bytes,
            ..
        } => format!(
            "{} -> {} copy {}",
            from.key(),
            to.key(),
            width(written_bytes)
        ),
        OpClass::NonFlopArith { kind } => match kind {
            ArithKind::Integer => "integer arithmetic",
            ArithKind::Predicate => "compare / select",
            ArithKind::Conversion => "conversion",
            ArithKind::Move => "register move",
        }
        .to_owned(),
        OpClass::Sync => "synchronization".to_owned(),
        OpClass::Communication => "warp communication".to_owned(),
        OpClass::Control => "control".to_owned(),
        OpClass::Ignore => "hint / no-op".to_owned(),
        OpClass::Unknown => "unknown".to_owned(),
    }
}

struct FlopTable {
    by_precision: BTreeMap<Precision, TermSum>,
    total: TermSum,
}

impl FlopTable {
    fn new() -> Self {
        FlopTable {
            by_precision: Precision::ALL
                .iter()
                .map(|&p| (p, TermSum::default()))
                .collect(),
            total: TermSum::default(),
        }
    }

    fn add(&mut self, precision: Precision, n: i64, mult: &SymExpr, at_most: bool) {
        self.by_precision
            .get_mut(&precision)
            .expect("every precision pre-inserted")
            .add(n, mult, at_most);
        self.total.add(n, mult, at_most);
    }

    fn add_unknown(&mut self) {
        self.total.add_unknown();
    }

    fn counts(&self) -> BTreeMap<String, Count> {
        let mut out: BTreeMap<String, Count> = self
            .by_precision
            .iter()
            .map(|(p, v)| (p.key().to_owned(), v.count()))
            .collect();
        out.insert("total".to_owned(), self.total.count());
        out
    }
}

/// `(constant coefficient, the rest)` of a product.
fn split_const(e: SymExpr) -> (i64, SymExpr) {
    match e {
        SymExpr::Const(c) => (c, SymExpr::Const(1)),
        SymExpr::Prod(fs) => match fs.first() {
            Some(&SymExpr::Const(c)) => {
                let rest = fs[1..]
                    .iter()
                    .cloned()
                    .reduce(SymExpr::mul)
                    .unwrap_or(SymExpr::Const(1));
                (c, rest)
            }
            _ => (1, SymExpr::Prod(fs)),
        },
        other => (1, other),
    }
}

struct KernelBuilder<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    cfg: Cfg,
    forest: LoopForest,
    names: Vec<LoopName>,
    trip_info: TripInfo,
    blocks: Vec<BlockMeasurements>,
    bind_map: &'a HashMap<String, i64>,
    /// Trip expression per loop for aggregation (opaque symbol when
    /// unresolved), already bound.
    trip_exprs: Vec<SymExpr>,
    /// Loop is conditionally entered within its parent scope.
    cond_entry: Vec<bool>,
    /// Display names with the remainder suffix applied.
    display: Vec<String>,
}

impl<'a> KernelBuilder<'a> {
    fn new(module: &'a Module, kernel: &'a Kernel, bind_map: &'a HashMap<String, i64>) -> Self {
        let cfg = build_cfg(module, kernel);
        let forest = loop_forest(&cfg);
        let names = loop_names(module, kernel, &cfg, &forest);
        let trip_info = trip_counts(module, kernel, &cfg, &forest, &names);
        let blocks = collect(module, kernel, &cfg, &forest);

        let mut display: Vec<String> = names.iter().map(|n| n.display.clone()).collect();
        for pair in &trip_info.unroll_pairs {
            let r = pair.remainder.0 as usize;
            display[r] = format!("{} (remainder)", display[r]);
        }

        let trip_exprs: Vec<SymExpr> = trip_info
            .trips
            .iter()
            .enumerate()
            .map(|(i, t)| match t {
                Ok(e) => e.bind(bind_map),
                Err(_) => SymExpr::sym(format!("trips({})", display[i])),
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
                targets.is_empty() || !targets.iter().all(|&t| forest.doms.dominates(l.header, t))
            })
            .collect();

        KernelBuilder {
            module,
            kernel,
            cfg,
            forest,
            names,
            trip_info,
            blocks,
            bind_map,
            trip_exprs,
            cond_entry,
            display,
        }
    }

    fn blocks(&self) -> Vec<BlockInfo> {
        let name = |b: BlockId| self.cfg.block_name(self.module, b);
        (0..self.cfg.blocks.len() as u32)
            .map(BlockId)
            .map(|b| {
                let instrs: Vec<&Instr> = self.cfg.instrs(self.kernel, b).collect();
                let in_loop = self.forest.block_loop[b.0 as usize].map(|l| BlockLoop {
                    name: self.display[l.0 as usize].clone(),
                    header: self.forest.get(l).header == b,
                    latch: self.forest.get(l).latches.contains(&b),
                });
                BlockInfo {
                    name: name(b),
                    lines: self.line_span(&instrs),
                    instructions: instrs.len() as u64,
                    successors: self.cfg.block(b).succs.iter().map(|&s| name(s)).collect(),
                    r#loop: in_loop,
                }
            })
            .collect()
    }

    fn line_span(&self, instrs: &[&Instr]) -> Option<String> {
        let first = instrs.iter().filter_map(|i| i.loc).find(|l| l.line != 0)?;
        let (lo, hi) = instrs
            .iter()
            .filter_map(|i| i.loc)
            .filter(|l| l.file == first.file && l.line != 0)
            .fold((first.line, first.line), |(lo, hi), l| {
                (lo.min(l.line), hi.max(l.line))
            });
        let file = basename(
            self.module
                .file_path(first.file)
                .unwrap_or("<unknown file>"),
        );
        Some(if lo == hi {
            format!("{file}:{lo}")
        } else {
            format!("{file}:{lo}-{hi}")
        })
    }

    /// Loop chain of a block, innermost first.
    fn chain(&self, b: BlockId) -> Vec<LoopId> {
        let mut out = Vec::new();
        let mut cur = self.forest.block_loop[b.0 as usize];
        while let Some(l) = cur {
            out.push(l);
            cur = self.forest.get(l).parent;
        }
        out
    }

    /// Aggregate over blocks; `below` = aggregate within this loop
    /// (multipliers stop there), `None` = whole kernel. With
    /// `cta`, every per-thread count additionally scales by the CTA's
    /// thread count, and is an upper bound when that count is one.
    fn aggregates(&self, below: Option<LoopId>, cta: Option<&LaunchInfo>) -> Aggregates {
        let mut flops: BTreeMap<Pipe, FlopTable> =
            Pipe::ALL.iter().map(|&p| (p, FlopTable::new())).collect();
        let mut bytes: BTreeMap<&'static str, (TermSum, TermSum)> = BTreeMap::new();
        let mut conversions = TermSum::default();
        let mut instructions: BTreeMap<String, (TermSum, BTreeMap<String, TermSum>)> =
            BTreeMap::new();
        let mut instruction_total = TermSum::default();
        for s in ["global", "shared", "local"] {
            bytes.insert(s, Default::default());
        }

        for bm in &self.blocks {
            let chain = self.chain(bm.block);
            // Inside `below`? (kernel root: always.)
            let cut = match below {
                Some(l) => {
                    let Some(pos) = chain.iter().position(|&x| x == l) else {
                        continue;
                    };
                    pos
                }
                None => chain.len(),
            };
            let mut mult = SymExpr::Const(1);
            let mut chain_at_most = false;
            for &l in &chain[..cut] {
                mult = SymExpr::mul(mult, self.trip_exprs[l.0 as usize].clone());
                chain_at_most |= self.cond_entry[l.0 as usize];
            }
            let block_at_most = bm.qualifier == CountQualifier::AtMost
                || chain_at_most
                || cta.is_some_and(|c| !c.exact);
            let threads = cta.map_or(1, |c| c.threads as i64);
            for ((class, opcode), &n) in &bm.instructions {
                let n = n as i64 * threads;
                let kind = instructions.entry(instruction_kind(*class)).or_default();
                kind.0.add(n, &mult, block_at_most);
                kind.1
                    .entry(opcode.clone())
                    .or_default()
                    .add(n, &mult, block_at_most);
                instruction_total.add(n, &mult, block_at_most);
            }
            for m in &bm.measurements {
                let at_most = bm.qualifier == CountQualifier::AtMost
                    || m.predicated
                    || chain_at_most
                    || cta.is_some_and(|c| !c.exact);
                let n = m.count as i64 * cta.map_or(1, |c| c.threads as i64);
                match m.kind {
                    MeasureKind::Flops { pipe, precision } => {
                        flops
                            .get_mut(&pipe)
                            .expect("every pipe pre-inserted")
                            .add(precision, n, &mult, at_most);
                    }
                    MeasureKind::Bytes { space, direction } => {
                        let entry = bytes.entry(space.key()).or_default();
                        match direction {
                            Direction::Load => entry.0.add(n, &mult, at_most),
                            Direction::Store => entry.1.add(n, &mult, at_most),
                        }
                    }
                    MeasureKind::Conversions => conversions.add(n, &mult, at_most),
                    MeasureKind::UnquantifiedBytes { space, direction } => {
                        let entry = bytes.entry(space.key()).or_default();
                        match direction {
                            Direction::Load => entry.0.add_unknown(),
                            Direction::Store => entry.1.add_unknown(),
                        }
                    }
                    MeasureKind::UnknownOps { .. } => {
                        flops.values_mut().for_each(FlopTable::add_unknown);
                        for space in ["global", "shared"] {
                            let (l, s) = bytes.get_mut(space).expect("pre-inserted");
                            l.add_unknown();
                            s.add_unknown();
                        }
                    }
                    // Op-count kinds appear in unknowns/classes, not in
                    // the workload aggregates.
                    MeasureKind::NonFlopOps { .. }
                    | MeasureKind::SyncOps
                    | MeasureKind::CommunicationOps
                    | MeasureKind::ControlOps => {}
                }
            }
        }

        let all_flops = flops
            .values()
            .map(|t| t.total.expr())
            .fold(SymExpr::Const(0), SymExpr::add);
        let bytes_out: BTreeMap<String, DirectionCounts> = bytes
            .iter()
            .map(|(k, (l, s))| {
                (
                    k.to_string(),
                    DirectionCounts {
                        load: l.count(),
                        store: s.count(),
                    },
                )
            })
            .collect();

        let flops_bound = direction(
            flops.values().any(|t| t.total.at_most),
            flops.values().any(|t| t.total.at_least),
        );
        let ai_global = match (all_flops.as_const(), bytes.get("global")) {
            (Some(f), Some((l, s))) => match (l.expr().as_const(), s.expr().as_const()) {
                (Some(lb), Some(sb)) if lb + sb > 0 => {
                    let bytes_bound = direction(l.at_most || s.at_most, l.at_least || s.at_least);
                    ratio_bound(flops_bound, bytes_bound).map(|bound| Intensity {
                        value: f as f64 / (lb + sb) as f64,
                        bound,
                    })
                }
                _ => None,
            },
            _ => None,
        };

        Aggregates {
            flops: flops[&Pipe::CudaCore].counts(),
            tensor_flops: flops[&Pipe::Tensor].counts(),
            sfu_flops: flops[&Pipe::Sfu].counts(),
            bytes: bytes_out,
            conversions: conversions.count(),
            ai_global,
            unrolled_source_lines: self.unrolled_lines(below),
            instructions: InstructionCounts {
                total: instruction_total.count(),
                by_kind: instructions
                    .iter()
                    .map(|(k, (total, opcodes))| {
                        let counts = KindCounts {
                            total: total.count(),
                            opcodes: opcodes
                                .iter()
                                .map(|(o, v)| (o.clone(), v.count()))
                                .collect(),
                        };
                        (k.clone(), counts)
                    })
                    .collect(),
            },
        }
    }

    /// Workload ops (flops + memory) per effective source line over the
    /// blocks DIRECTLY in this scope — the line-aggregation view that
    /// recovers fully-unrolled source loops. Entries with ≥ 2 copies.
    fn unrolled_lines(&self, scope: Option<LoopId>) -> BTreeMap<String, u64> {
        let mut per_line: BTreeMap<String, u64> = BTreeMap::new();
        for bm in &self.blocks {
            if self.forest.block_loop[bm.block.0 as usize] != scope {
                continue;
            }
            let blk = self.cfg.block(bm.block);
            for stmt in &self.kernel.stmts[blk.start..blk.end] {
                let Stmt::Instr(instr) = stmt else { continue };
                let Some(loc) = instr.loc.filter(|l| l.line != 0) else {
                    continue;
                };
                let is_workload = matches!(
                    crate::classify::classify(self.module, instr),
                    crate::classify::OpClass::Flop { .. }
                        | crate::classify::OpClass::Memory { .. }
                        | crate::classify::OpClass::Copy { .. }
                );
                if !is_workload {
                    continue;
                }
                let file = self
                    .module
                    .file_path(loc.file)
                    .map(|p| p.rsplit('/').next().unwrap_or(p).to_owned())
                    .unwrap_or_else(|| "<unknown file>".to_owned());
                *per_line.entry(format!("{file}:{}", loc.line)).or_default() += 1;
            }
        }
        per_line.retain(|_, &mut v| v >= 2);
        per_line
    }

    fn loop_node(&self, id: LoopId) -> LoopNode {
        let name = &self.names[id.0 as usize];
        let trips = match &self.trip_info.trips[id.0 as usize] {
            Ok(e) => {
                let bound = e.bind(self.bind_map);
                Trips {
                    expr: Some(bound.to_string()),
                    unknown: None,
                }
            }
            Err(reason) => Trips {
                expr: None,
                unknown: Some(reason.clone()),
            },
        };
        let unroll = self
            .trip_info
            .unroll_pairs
            .iter()
            .find(|p| p.main == id)
            .map(|p| Unroll {
                factor: p.factor,
                remainder: self.display[p.remainder.0 as usize].clone(),
            });
        LoopNode {
            name: self.display[id.0 as usize].clone(),
            label: name.label.clone(),
            line: name.line,
            depth: self.forest.get(id).depth,
            trips,
            unroll,
            per_iteration: self.aggregates(Some(id), None),
            loops: self
                .forest
                .children_of(id)
                .into_iter()
                .map(|c| self.loop_node(c))
                .collect(),
        }
    }

    /// Weight = executed instructions per kernel invocation; ranking
    /// compares weights at all-symbols = 2^20 + 3 (large, with nonzero
    /// residues mod small powers of two so remainder loops keep their
    /// share). Deliberately a comparison heuristic, not a claim.
    fn ranking(&self) -> Vec<(LoopId, RankEntry)> {
        let mut entries: Vec<(usize, String, SymExpr, i64)> = (0..self.forest.loops.len())
            .map(|i| {
                let l = &self.forest.loops[i];
                let mut weight = TermSum::default();
                for &b in &l.blocks {
                    let instrs = self.blocks[b.0 as usize].class_counts.total as i64;
                    let mut mult = SymExpr::Const(1);
                    for &cl in &self.chain(b) {
                        mult = SymExpr::mul(mult, self.trip_exprs[cl.0 as usize].clone());
                    }
                    weight.add(instrs, &mult, false);
                }
                let weight = weight.expr();
                let approx: HashMap<String, i64> = weight
                    .symbols()
                    .into_iter()
                    .map(|s| (s, (1 << 20) + 3))
                    .collect();
                let key = weight.bind(&approx).as_const().unwrap_or(i64::MAX);
                (i, self.display[i].clone(), weight, key)
            })
            .collect();
        entries.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.1.cmp(&b.1)));
        entries
            .into_iter()
            .map(|(i, name, w, _)| {
                (
                    LoopId(i as u32),
                    RankEntry {
                        loop_name: name,
                        instructions: w.to_string(),
                    },
                )
            })
            .collect()
    }

    fn unknowns(&self) -> Vec<UnknownEntry> {
        let mut out = Vec::new();
        let mut unknown_ops: BTreeMap<String, u64> = BTreeMap::new();
        let mut unquantified: BTreeMap<String, u64> = BTreeMap::new();
        for bm in &self.blocks {
            for m in &bm.measurements {
                match m.kind {
                    MeasureKind::UnknownOps { mnemonic } => {
                        *unknown_ops
                            .entry(self.module.interner.resolve(mnemonic).to_owned())
                            .or_default() += m.count;
                    }
                    MeasureKind::UnquantifiedBytes { space, direction } => {
                        let dir = match direction {
                            Direction::Load => "load",
                            Direction::Store => "store",
                        };
                        *unquantified
                            .entry(format!("{} {dir}", Space::key(space)))
                            .or_default() += m.count;
                    }
                    _ => {}
                }
            }
        }
        let unparsed: u64 = self
            .blocks
            .iter()
            .map(|b| b.class_counts.unparsed as u64)
            .sum();
        if unparsed > 0 {
            out.push(UnknownEntry {
                what: "unparsed statement".to_owned(),
                count: Some(unparsed),
                reason: "the parser could not read it — whatever it does is not counted".to_owned(),
            });
        }
        for (mnemonic, count) in unknown_ops {
            out.push(UnknownEntry {
                what: format!("instruction `{mnemonic}`"),
                count: Some(count),
                reason: "not classified — its flops/bytes are not counted".to_owned(),
            });
        }
        for (what, count) in unquantified {
            out.push(UnknownEntry {
                what: format!("{what} with statically unknown byte count"),
                count: Some(count),
                reason: "counted as an op; bytes missing from every byte total".to_owned(),
            });
        }
        for (i, t) in self.trip_info.trips.iter().enumerate() {
            if let Err(reason) = t {
                out.push(UnknownEntry {
                    what: format!("loop {}", self.display[i]),
                    count: None,
                    reason: reason.clone(),
                });
            }
        }
        for (src, dst) in &self.forest.irreducible_edges {
            let label = |b: BlockId| {
                self.cfg
                    .block(b)
                    .label
                    .map(|s| self.module.interner.resolve(s).to_owned())
                    .unwrap_or_else(|| format!("<block {}>", b.0))
            };
            out.push(UnknownEntry {
                what: format!(
                    "irreducible control flow {} -> {}",
                    label(*src),
                    label(*dst)
                ),
                count: None,
                reason: "cycle with multiple entries — execution multiplicity unknown".to_owned(),
            });
        }
        for (block, target) in &self.cfg.unresolved_branches {
            let from = self
                .cfg
                .block(*block)
                .label
                .map(|s| self.module.interner.resolve(s).to_owned())
                .unwrap_or_else(|| format!("<block {}>", block.0));
            out.push(UnknownEntry {
                what: format!(
                    "branch to `{}` from {from}",
                    self.module.interner.resolve(*target)
                ),
                count: None,
                reason: "branch target matched no label in this kernel; its edge \
                         was dropped, so the control-flow graph — and any loop \
                         structure derived from it — may be incomplete. Unexpected \
                         for compiler-produced PTX"
                    .to_owned(),
            });
        }
        if !self.cfg.call_sites.is_empty() {
            out.push(UnknownEntry {
                what: "call".to_owned(),
                count: Some(self.cfg.call_sites.len() as u64),
                reason: "non-inlined callee — its cost is not included".to_owned(),
            });
        }
        out
    }

    /// Shared memory the kernel reserves per CTA: the sum over its
    /// `.shared` array declarations of `element_count × element_width`,
    /// plus whether it declares a launch-sized `.extern .shared` array.
    /// See [`SharedMemory`]. Element width reuses the classifier's PTX
    /// type table; an unrecognized type falls back to byte sizing, the
    /// `.b8` form nvcc emits for shared tiles.
    fn shared_memory(&self) -> SharedMemory {
        let mut static_bytes = 0u64;
        let mut dynamic = false;
        for decl in self
            .kernel
            .shared_decls
            .iter()
            .chain(&self.module.shared_decls)
        {
            match decl.size {
                Some(count) => {
                    let ty = self.module.interner.resolve(decl.ty);
                    let width = crate::classify::type_width(ty).unwrap_or(1) as u64;
                    static_bytes += count * width;
                }
                None => dynamic = true,
            }
        }
        SharedMemory {
            static_bytes,
            dynamic,
        }
    }

    fn build(self, launch_flag: Option<[u32; 3]>) -> KernelReport {
        let name = self.module.interner.resolve(self.kernel.name).to_owned();
        let mut classes = InstructionClasses::default();
        for bm in &self.blocks {
            let c = bm.class_counts;
            classes.total += c.total as u64;
            classes.flop += c.flop as u64;
            classes.non_flop_arith += c.non_flop_arith as u64;
            classes.memory += c.memory as u64;
            classes.sync += c.sync as u64;
            classes.communication += c.communication as u64;
            classes.control += c.control as u64;
            classes.ignore += c.ignore as u64;
            classes.unknown += c.unknown as u64;
            classes.unparsed += c.unparsed as u64;
        }
        let ranking = self.ranking();

        // Launch config: explicit flag, else the PTX's own directives.
        let launch = launch_flag
            .map(|block| (block, "flag"))
            .or(self.kernel.reqntid.map(|b| (b, ".reqntid")))
            .or(self.kernel.maxntid.map(|b| (b, ".maxntid")))
            .map(|(block, source)| LaunchInfo {
                block,
                threads: block.iter().map(|&d| d as u64).product(),
                source: source.to_owned(),
                exact: source != ".maxntid",
            });
        let totals_per_cta = launch.as_ref().map(|l| self.aggregates(None, Some(l)));

        let unknowns = self.unknowns();

        KernelReport {
            demangled: demangle(&name),
            name,
            params: self
                .kernel
                .params
                .iter()
                .enumerate()
                .map(|(i, p)| ParamInfo {
                    index: i,
                    ty: self.module.interner.resolve(p.ty).to_owned(),
                    name: self.module.interner.resolve(p.name).to_owned(),
                })
                .collect(),
            blocks: self.blocks(),
            shared_memory: self.shared_memory(),
            instruction_classes: classes,
            most_instructions_loop: ranking.first().map(|(_, r)| r.loop_name.clone()),
            launch,
            totals_per_cta,
            ranking: ranking.into_iter().map(|(_, r)| r).collect(),
            loops: self
                .forest
                .top_level()
                .into_iter()
                .map(|t| self.loop_node(t))
                .collect(),
            totals: self.aggregates(None, None),
            unknowns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_parsing_grammar() {
        assert_eq!(
            parse_bind("K=4096"),
            Ok(BindingSpec {
                index: None,
                name: "K".into(),
                value: 4096
            })
        );
        assert_eq!(
            parse_bind("2:K=4096"),
            Ok(BindingSpec {
                index: Some(2),
                name: "K".into(),
                value: 4096
            })
        );
        assert!(parse_bind("K").is_err());
        assert!(parse_bind("K=x").is_err());
        assert!(parse_bind("=4").is_err());
        assert!(parse_bind("a:K=4").is_err());
    }

    /// A scope holding an unclassified instruction or an unquantified
    /// byte count marks its totals as lower bounds; AI(global) then has
    /// no direction when the sides disagree.
    #[test]
    fn unknowns_in_scope_make_totals_lower_bounds() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_90a\n.address_size 64\n\
                   .visible .entry k()\n{\n\
                   ld.global.f32 %f1, [%rd1];\nfma.rn.f32 %f2, %f1, %f1, %f1;\n\
                   wgmma.fence.sync.aligned;\n\
                   cp.async.cg.shared.global [%r1], [%rd2], %r2;\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        let t = &r.kernels[0].totals;
        assert_eq!(t.flops["total"].expr, "2");
        assert!(t.flops["total"].at_least && !t.flops["total"].at_most);
        assert!(!t.flops["f32"].at_least);
        assert!(t.bytes["global"].load.at_least && t.bytes["shared"].store.at_least);
        assert!(t.ai_global.is_none());

        let src = src.replace("wgmma.fence.sync.aligned;\n", "");
        let r = analyze(&src, "t", &opts).expect("analyzes");
        let t = &r.kernels[0].totals;
        assert!(!t.flops["total"].at_least);
        assert!(t.bytes["global"].load.at_least && !t.bytes["global"].store.at_least);
        assert_eq!(t.ai_global.map(|ai| ai.bound), Some(Bound::AtMost));
    }

    /// Static shared memory per CTA = Σ (element count × element width)
    /// over `.shared` decls; `.extern .shared` is dynamic, not counted.
    #[test]
    fn shared_memory_static_and_dynamic() {
        let opts = AnalyzeOptions::default();
        let body = |decls: &str| {
            format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k()\n{{\n{decls}ret;\n}}\n"
            )
        };
        let sm = |decls: &str| {
            let r = analyze(&body(decls), "t", &opts).expect("analyzes");
            let s = &r.kernels[0].shared_memory;
            (s.static_bytes, s.dynamic)
        };

        // The k5 shape: two byte arrays, 1024 + 1024 = 2048 (matches the
        // `2048 bytes smem` ptxas reports for the k5 fixture).
        assert_eq!(
            sm(".shared .align 2 .b8 As[1024];\n.shared .align 2 .b8 Bs[1024];\n"),
            (2048, false)
        );
        // Typed array: per-element width applies, 256 × 4 = 1024.
        assert_eq!(sm(".shared .align 4 .f32 buf[256];\n"), (1024, false));
        // Width coverage: b64 → 8, b128 → 16.
        assert_eq!(
            sm(".shared .b64 w[4];\n.shared .b128 q[2];\n"),
            (4 * 8 + 2 * 16, false)
        );
        // The k1 shape: no shared memory at all.
        assert_eq!(sm(""), (0, false));
        // Dynamic extern shared: out of the static total, flagged.
        assert_eq!(sm(".extern .shared .align 8 .b8 dsm[];\n"), (0, true));
        // Mixed: static counted, dynamic flagged.
        assert_eq!(
            sm(".shared .align 2 .b8 As[1024];\n.extern .shared .align 8 .b8 dsm[];\n"),
            (1024, true)
        );
        // LLVM's module-scope form counts for every kernel in the module.
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .extern .shared .align 16 .b8 global_smem[];\n\
                   .visible .entry k()\n{\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        assert!(r.kernels[0].shared_memory.dynamic);
    }

    #[test]
    fn unparsed_statements_surface_as_a_counted_unknown() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k()\n{\n@@ not ptx;\n.bogus 1;\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        let k = &r.kernels[0];
        assert_eq!(
            (k.instruction_classes.total, k.instruction_classes.unparsed),
            (1, 2)
        );
        let entry = k.unknowns.iter().find(|e| e.what == "unparsed statement");
        assert_eq!(entry.map(|e| e.count), Some(Some(2)), "{:?}", k.unknowns);
    }

    /// A `bra` whose target matches no label surfaces as a report
    /// unknown (the dropped edge is reported, not hidden).
    #[test]
    fn unresolved_branch_surfaces_as_unknown() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k()\n{\nbra $L__MISSING;\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes despite the dangling branch");
        let u = &r.kernels[0].unknowns;
        assert!(
            u.iter()
                .any(|e| e.what.contains("$L__MISSING") && e.what.contains("branch")),
            "expected an unknown naming the unresolved branch target, got {u:?}"
        );
    }
}
