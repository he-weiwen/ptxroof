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

use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::control_flow::{dominators, loop_forest};
use crate::analysis::dead_code::unused;
use crate::analysis::instruction_counts::classify::{
    ArithKind, ClassifiedInstruction, Direction, InstructionCategory, Pipe, Precision, Space,
};
use crate::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier, collect};
use crate::analysis::instruction_counts::count_constraint::count_constraint;
use crate::analysis::instruction_counts::measurement::{Contribution, MeasureKind};
use crate::analysis::loop_names::{LoopName, loop_names};
use crate::analysis::memory_footprint::{
    AccessForm, CountRange, at_trip, loop_bytes, warp_footprint,
};
use crate::analysis::scalar::affine::{Affine, Var};
use crate::analysis::scalar::symexpr::SymExpr;
use crate::analysis::scalar::trace::AffineValueTracer;
use crate::analysis::scalar::trip_counts::{TripCountResults, trip_counts};
use crate::analysis::thread_participation::{Pred, ThreadSet, ThreadSets, thread_sets};
use crate::analysis::uniformity::Uniformity;
use crate::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
use crate::ptx::ir::Operand;
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Register, SpecialVector, Stmt};
use crate::ptx::literal::parse_int;
use crate::ptx::parse::parser::{ParseError, parse};
use crate::report::measured::{LineIndex, NcuRecord, Prediction, Sectors};
use crate::report::names::demangle;
use crate::report::schema::*;
use crate::ssa::Value;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, thiserror::Error)]
pub enum AnalyzeError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("bad --bind: {0}")]
    Binding(String),
    #[error("--ncu record does not match: {0}")]
    Ncu(String),
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
    /// `--grid x,y,z`, the launch's CTAs per axis.
    pub grid: Option<[u32; 3]>,
    pub ncu: Option<NcuRecord>,
}

pub fn analyze(
    source: &str,
    input_name: &str,
    opts: &AnalyzeOptions,
) -> Result<Report, AnalyzeError> {
    let module = parse(source)?;
    if let Some(record) = &opts.ncu {
        record.check(&module, source).map_err(AnalyzeError::Ncu)?;
    }

    let lines = LineIndex::new(source);
    let mut kernels = Vec::new();
    let mut classified = Fraction { num: 0, den: 0 };
    let mut trips_resolved = Fraction { num: 0, den: 0 };
    let mut bindings_echo = Vec::new();

    for kernel in &module.kernels {
        let mut bind_map = resolve_bindings(&module, kernel, &opts.bindings, &mut bindings_echo)?;
        // The block shape, when it is the launch's: --launch, else
        // .reqntid; the grid, when given.
        let shapes = [
            (SpecialVector::Ntid, opts.launch.or(kernel.reqntid)),
            (SpecialVector::Nctaid, opts.grid),
        ];
        for (vector, dims) in shapes {
            for (reg, n) in vector
                .components()
                .into_iter()
                .zip(dims.into_iter().flatten())
            {
                bind_map.insert(reg.name(), i64::from(n));
            }
        }
        let ncu = opts.ncu.as_ref().map(|r| (r, &lines));
        let k =
            KernelReportBuilder::new(&module, kernel, &bind_map, &lines).build(opts.launch, ncu);
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
        grid: opts.grid,
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

/// `a`, unless it counts the iterations of a loop that does not hold
/// `pos` and that lanes leave at different iterations, or holds a
/// uniform value the lanes executing `pos` do not see alike: then each
/// lane reads its own.
fn per_lane(
    a: Affine,
    tracer: &AffineValueTracer,
    uniformity: &Uniformity,
    pos: usize,
    name: &dyn Fn(&Var) -> String,
) -> Result<Affine, String> {
    for v in a.terms.keys() {
        if let Some(l) = v.counter().filter(|&l| !tracer.in_loop(l, pos)) {
            let why = uniformity.exits_uniform(tracer, l).err();
            if let Some(why) = why {
                let k = name(&Var::Iter(l));
                return Err(format!(
                    "{k} differs between lanes after its loop, whose exit {why}"
                ));
            }
        }
        if let Some(u) = v.uniform()
            && let Err(why) = uniformity.value_uniform_at(tracer, u, pos)
        {
            let u = name(&Var::Uniform(u));
            return Err(format!("{u} differs between lanes: {why}"));
        }
    }
    Ok(a)
}

impl KernelReportBuilder<'_> {
    /// Every memory operand of every memory instruction, traced to an
    /// affine address at its own position, grouped by the innermost loop
    /// of its block. `cp.async` gives a store row for its shared
    /// destination and a load row for its global source; an atomic gives
    /// one `load+store` row. Parameter loads are not accesses.
    fn accesses_by_scope(
        &self,
        tracer: &AffineValueTracer,
        uniformity: &Uniformity,
    ) -> (ByScope<Access>, ByScope<AccessForm>, PerRequests) {
        let (module, kernel, cfg, forest, bind_map) = (
            self.module,
            self.kernel,
            &self.cfg,
            &self.forest,
            self.bind_map,
        );
        let display = &self.display;
        let mut forms: ByScope<AccessForm> = HashMap::new();
        let mut per_request = PerRequests::new();
        let name = |v: &Var| match v {
            Var::Iter(l) => format!("k[{}]", display[l.0 as usize]),
            Var::Uniform(u) => format!("uniform({})", self.value_name(*u)),
            other => other.to_string(),
        };
        // A uniform value defined in loop `l` may differ between its trips.
        let defined_in = |u: Value, l: LoopId| {
            let b = match u {
                Value::Result { op, .. } => tracer.block_of(op),
                Value::Argument { block, .. } => block,
                Value::Undef(_) => return false,
            };
            forest.get(l).blocks.binary_search(&b).is_ok()
        };
        // A shared array's mangled name reads as its source name.
        let arrays: Vec<(String, String)> = kernel
            .shared_decls
            .iter()
            .chain(&module.shared_decls)
            .map(|d| {
                let raw = module.interner.resolve(d.name).to_owned();
                let short = demangle(&raw)
                    .rsplit("::")
                    .next()
                    .unwrap_or(&raw)
                    .to_owned();
                (raw, short)
            })
            .collect();
        let shorten = |mut text: String| {
            for (raw, short) in &arrays {
                text = text.replace(raw, short);
            }
            text
        };
        let shape = SpecialVector::Ntid
            .components()
            .map(|reg| bind_map.get(&reg.name()).map(|&v| v as u32).unwrap_or(0));
        let mut out: HashMap<Option<LoopId>, Vec<Access>> = HashMap::new();
        for bm in &self.blocks {
            let bid = bm.block;
            let scope = forest.block_loop[bid.0 as usize];
            for record in &bm.instructions {
                let pos = record.provenance;
                let Stmt::Instr(instr) = &kernel.stmts[pos] else {
                    continue;
                };
                let set = self.sets.of(bid, pos);
                let guard_exact = self.sets.guards.get(&pos).is_none_or(|g| {
                    let alone = ThreadSet { pred: g.clone() };
                    alone
                        .count(shape)
                        .map_or(g.exact(), |c| c.certain == c.possible)
                });
                let conditional = self.blocks[bid.0 as usize].qualifier == CountQualifier::AtMost
                    && set.count(shape).is_none_or(|c| c.certain != c.possible);
                let threads = self
                    .sets
                    .guards
                    .contains_key(&pos)
                    .then(|| self.threads_info(&set))
                    .flatten();
                let mem_ops: Vec<_> = module
                    .operand_ids(instr.operands)
                    .iter()
                    .copied()
                    .filter(|&id| matches!(module.operand(id), Operand::Memory { .. }))
                    .collect();
                let classified = &record.classified;
                // Pair memory contributions by their explicit operand association.
                // An atomic has one address with read+write; a copy has two addresses.
                let mut accesses: BTreeMap<(usize, Space, Option<u32>), (bool, bool)> =
                    BTreeMap::new();
                for c in &classified.contributions {
                    let Some(operand) = c.memory_operand.filter(|&i| i < mem_ops.len()) else {
                        continue;
                    };
                    let (space, direction, bytes) = match c.kind {
                        MeasureKind::Bytes { space, direction } => {
                            (space, direction, Some(c.count as u32))
                        }
                        MeasureKind::UnquantifiedBytes { space, direction } => {
                            (space, direction, None)
                        }
                        _ => continue,
                    };
                    let sides = accesses.entry((operand, space, bytes)).or_default();
                    match direction {
                        Direction::Load => sides.0 = true,
                        Direction::Store => sides.1 = true,
                    }
                }
                let rows = accesses
                    .into_iter()
                    .map(|((i, space, bytes), (load, store))| {
                        let direction = match (load, store) {
                            (true, true) => "load+store",
                            (true, false) => "load",
                            _ => "store",
                        };
                        (i, space, direction, bytes)
                    });
                for (i, space, direction, bytes) in rows {
                    if space == Space::Param {
                        continue;
                    }
                    let Operand::Memory { base, offset, .. } = module.operand(mem_ops[i]) else {
                        continue;
                    };
                    let (address, unknown, form) = match module.operand(*base) {
                        Operand::SymbolRef(s) => {
                            let sym = shorten(module.interner.resolve(*s).to_owned());
                            let addr = if *offset == 0 {
                                sym
                            } else {
                                format!("{sym} + {offset}")
                            };
                            (Some(addr), None, None)
                        }
                        Operand::Register(_) => match (tracer.trace_address(*base, pos, scope))
                            .map(|a| {
                                (a + Affine::invariant(SymExpr::Const(*offset))).bind(bind_map)
                            })
                            .and_then(|a| per_lane(a, tracer, uniformity, pos, &name))
                        {
                            Ok(a) => (Some(shorten(a.render(name))), None, Some(a)),
                            Err(reason) => (
                                None,
                                Some(
                                    reason
                                        .strip_prefix("latch condition ")
                                        .unwrap_or(&reason)
                                        .to_owned(),
                                ),
                                None,
                            ),
                        },
                        _ => (
                            None,
                            Some("address operand form not traced".to_owned()),
                            None,
                        ),
                    };
                    let align = match instr.mnemonic {
                        Opcode::Cp => module.operand_ids(instr.operands).iter().find_map(|&id| {
                            match module.operand(id) {
                                Operand::Immediate(text) => {
                                    parse_int(module.interner.resolve(*text))
                                        .and_then(|v| u32::try_from(v).ok())
                                }
                                _ => None,
                            }
                        }),
                        _ => None,
                    };
                    let footprint = match (&form, bytes) {
                        (Some(a), Some(b)) if matches!(space, Space::Global | Space::Generic) => {
                            Some(warp_footprint(a, b, align.unwrap_or(b), shape, |tid| {
                                set.contains(tid, shape)
                            }))
                        }
                        _ => None,
                    };
                    let sectors_at_most = !set.lane_exact();
                    let top = |r: CountRange| CountRange {
                        min: r.max,
                        max: r.max,
                    };
                    let (sectors_per_request, ideal, footprint_unknown) = match footprint {
                        Some(Ok(f)) if sectors_at_most => {
                            (Some(top(f.sectors)), Some(top(f.ideal)), None)
                        }
                        Some(Ok(f)) => (Some(f.sectors), Some(f.ideal), None),
                        Some(Err(why)) => (None, None, Some(why)),
                        None => (None, None, None),
                    };
                    if space == Space::Global
                        && let (Some(b), Some(sectors), Some(ideal)) =
                            (bytes, sectors_per_request, ideal)
                    {
                        let r = PerRequest {
                            bits: b * 8,
                            sectors,
                            ideal,
                            at_most: sectors_at_most,
                        };
                        per_request.insert(pos, r);
                    }
                    if let Some(b) = bytes.filter(|_| !self.unused[pos]) {
                        let uniform_varies = (form.iter()).flat_map(|a| a.terms.keys()).any(|v| {
                            v.uniform()
                                .is_some_and(|u| scope.is_some_and(|l| defined_in(u, l)))
                        });
                        forms.entry(scope).or_default().push(AccessForm {
                            form: form.clone(),
                            bytes: b,
                            align: align.unwrap_or(b),
                            space,
                            set: set.clone(),
                            guard_exact,
                            conditional,
                            uniform_varies,
                        });
                    }
                    let mut reuse = Vec::new();
                    if let Some(a) = &form {
                        let mut cur = scope;
                        while let Some(l) = cur {
                            let stepped = (a.terms.keys()).any(|v| {
                                (*v != Var::Iter(l) && at_trip(v, l, 0).is_some())
                                    || v.uniform().is_some_and(|u| defined_in(u, l))
                            });
                            reuse.push(Reuse {
                                r#loop: display[l.0 as usize].clone(),
                                stride: match stepped {
                                    true => Some("varies".to_owned()),
                                    false => a.terms.get(&Var::Iter(l)).map(ToString::to_string),
                                },
                            });
                            cur = forest.get(l).parent;
                        }
                    }
                    let site = match instr.loc.filter(|l| l.line != 0) {
                        Some(loc) => format!(
                            "{}:{}",
                            module.file_name(loc.file).unwrap_or("<unknown file>"),
                            loc.line
                        ),
                        None => cfg.block_name(module, bid),
                    };
                    out.entry(scope).or_default().push(Access {
                        site,
                        opcode: module.opcode(instr),
                        space: space.key().to_owned(),
                        direction: direction.to_owned(),
                        bytes,
                        predicated: instr.predicate.is_some(),
                        threads: threads.clone(),
                        address,
                        unknown,
                        sectors_per_request,
                        sectors_at_most,
                        ideal_sectors_per_request: ideal,
                        footprint_unknown,
                        reuse,
                    });
                }
            }
        }
        (out, forms, per_request)
    }
}

/// A global access's width in bits and its sectors and ideal per warp
/// request, bounds when `at_most`.
#[derive(Clone, Copy)]
struct PerRequest {
    bits: u32,
    sectors: CountRange,
    ideal: CountRange,
    at_most: bool,
}

/// Per statement, its global access's sectors.
type PerRequests = HashMap<usize, PerRequest>;

/// The threads that run a statement, its warps with whether they are a
/// bound, and whether the threads are.
type Running<N = u64> = (N, Option<(N, bool)>, bool);

/// A `Running` as the i64 counts the totals sum, when it fits.
fn narrow((threads, warps, at_most): Running) -> Option<Running<i64>> {
    let warps = match warps {
        Some((w, bound)) => Some((i64::try_from(w).ok()?, bound)),
        None => None,
    };
    Some((i64::try_from(threads).ok()?, warps, at_most))
}

/// Rows per scope: a block's innermost loop, or `None` outside every loop.
type ByScope<T> = HashMap<Option<LoopId>, Vec<T>>;

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
struct CountAccumulator {
    groups: Vec<(SymExpr, i64)>,
    at_most: bool,
    at_least: bool,
    touched: bool,
}

impl CountAccumulator {
    /// `count` times `mult`; `None` when the count is not known, and
    /// unknown when it, or the sum, does not fit in `i64`.
    fn add(&mut self, count: i64, mult: Option<&SymExpr>, at_most: bool) {
        let Some(mult) = mult else {
            return self.add_unknown();
        };
        let (coeff, rest) = SymExpr::split_const(mult.clone());
        let Some(term) = coeff.checked_mul(count) else {
            return self.add_unknown();
        };
        match self.groups.iter_mut().find(|(m, _)| *m == rest) {
            Some((_, n)) => match n.checked_add(term) {
                Some(sum) => *n = sum,
                None => return self.add_unknown(),
            },
            None => self.groups.push((rest, term)),
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
fn instruction_kind(instruction: &ClassifiedInstruction) -> String {
    match instruction.category {
        InstructionCategory::Flop => {
            let Some(Contribution {
                kind: MeasureKind::Flops { pipe, precision },
                ..
            }) = instruction.contributions.first()
            else {
                return "floating-point arithmetic".to_owned();
            };
            format!("{} {}", pipe.key(), precision.key())
        }
        InstructionCategory::Memory => {
            let memory: Vec<_> = instruction
                .contributions
                .iter()
                .filter_map(|c| match c.kind {
                    MeasureKind::Bytes { space, direction } => {
                        Some((c, space, direction, format!("{} B", c.count)))
                    }
                    MeasureKind::UnquantifiedBytes { space, direction } => {
                        Some((c, space, direction, "? B".to_owned()))
                    }
                    _ => None,
                })
                .collect();
            match memory.as_slice() {
                [
                    (read, from, Direction::Load, _),
                    (write, to, Direction::Store, width),
                ] => {
                    if read.memory_operand == write.memory_operand {
                        format!("{} atomic {width}", from.key())
                    } else {
                        format!("{} -> {} copy {width}", from.key(), to.key())
                    }
                }
                [(_, space, direction, width)] => {
                    format!("{} {} {width}", space.key(), direction_key(*direction))
                }
                _ => "memory".to_owned(),
            }
        }
        InstructionCategory::NonFlopArith { kind } => arith_key(kind).to_owned(),
        InstructionCategory::Sync => "synchronization".to_owned(),
        InstructionCategory::Communication => "warp communication".to_owned(),
        InstructionCategory::Control => "control".to_owned(),
        InstructionCategory::Ignore => "hint / no-op".to_owned(),
        InstructionCategory::Unknown => "unknown".to_owned(),
    }
}

fn direction_key(direction: Direction) -> &'static str {
    match direction {
        Direction::Load => "load",
        Direction::Store => "store",
    }
}

fn arith_key(kind: ArithKind) -> &'static str {
    match kind {
        ArithKind::Conversion => "conversion",
        ArithKind::Integer => "integer arithmetic",
        ArithKind::Predicate => "compare / select",
        ArithKind::Move => "register move",
    }
}

fn contribution_details(c: &Contribution) -> ContributionDetails {
    let count = c.count;
    match c.kind {
        MeasureKind::Flops { pipe, precision } => ContributionDetails::Flops {
            pipe: pipe.key().to_owned(),
            precision: precision.key().to_owned(),
            count,
        },
        MeasureKind::Bytes { space, direction } => ContributionDetails::Bytes {
            space: space.key().to_owned(),
            direction: direction_key(direction).to_owned(),
            count,
        },
        MeasureKind::UnquantifiedBytes { space, direction } => {
            ContributionDetails::UnquantifiedBytes {
                space: space.key().to_owned(),
                direction: direction_key(direction).to_owned(),
            }
        }
        MeasureKind::Conversions => ContributionDetails::Conversions { count },
        MeasureKind::NonFlopOps { kind } => ContributionDetails::NonFlopOps {
            operation: arith_key(kind).to_owned(),
            count,
        },
        MeasureKind::SyncOps => ContributionDetails::SyncOps { count },
        MeasureKind::CommunicationOps => ContributionDetails::CommunicationOps { count },
        MeasureKind::ControlOps => ContributionDetails::ControlOps { count },
        MeasureKind::UnknownOps { mnemonic } => ContributionDetails::UnknownOps {
            mnemonic: mnemonic.name().to_owned(),
        },
    }
}

/// Thread-level issues, and warp-level issues when the totals are per
/// CTA: a warp issues an instruction once for every thread it holds.
#[derive(Default)]
struct IssueAccumulator {
    threads: CountAccumulator,
    warps: CountAccumulator,
}

impl IssueAccumulator {
    fn add(
        &mut self,
        threads: i64,
        warps: Option<(i64, bool)>,
        mult: Option<&SymExpr>,
        at_most: bool,
    ) {
        self.threads.add(threads, mult, at_most);
        if let Some((warps, warps_at_most)) = warps {
            self.warps.add(warps, mult, warps_at_most);
        }
    }

    fn warps(&self) -> Option<Count> {
        self.warps.touched.then(|| self.warps.count())
    }
}

#[derive(Default)]
struct InstructionAccumulator {
    total: IssueAccumulator,
    opcodes: BTreeMap<String, IssueAccumulator>,
    variants: BTreeMap<(String, Vec<Contribution>), IssueAccumulator>,
}

struct FlopAccumulator {
    by_precision: BTreeMap<Precision, CountAccumulator>,
    total: CountAccumulator,
}

impl FlopAccumulator {
    fn new() -> Self {
        FlopAccumulator {
            by_precision: Precision::ALL
                .iter()
                .map(|&p| (p, CountAccumulator::default()))
                .collect(),
            total: CountAccumulator::default(),
        }
    }

    fn add(&mut self, precision: Precision, n: i64, mult: Option<&SymExpr>, at_most: bool) {
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
struct KernelReportBuilder<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    cfg: ControlFlowGraph,
    forest: LoopForest,
    names: Vec<LoopName>,
    trip_info: TripCountResults,
    blocks: Vec<BlockMeasurements>,
    bind_map: &'a HashMap<String, i64>,
    /// Trip expression per loop for aggregation (opaque symbol when
    /// unresolved), already bound.
    trip_exprs: Vec<SymExpr>,
    /// Loop is conditionally entered within its parent scope.
    cond_entry: Vec<bool>,
    /// Display names with the remainder suffix applied.
    display: Vec<String>,
    /// Memory operands per scope: the innermost loop of their block, or
    /// `None` outside every loop.
    accesses: ByScope<Access>,
    /// The same rows' bound affine forms, where known.
    access_forms: ByScope<AccessForm>,
    per_request: PerRequests,
    /// Which threads run each block and each guarded instruction.
    sets: ThreadSets,
    /// Per statement: it runs, but nothing with an effect reads its results.
    unused: Vec<bool>,
    /// The block shape when known (`--launch` or `.reqntid`), else zeros.
    shape: [u32; 3],
    /// With the grid bound, the threads and warps over the launch of each
    /// block's set (`None`) and of each guarded statement's, where
    /// `ThreadSet::launch` counts them.
    launches: HashMap<(BlockId, Option<usize>), (u64, u64)>,
    /// The PTX file's lines, naming a uniform value by its statement's.
    lines: &'a LineIndex,
}

impl<'a> KernelReportBuilder<'a> {
    fn new(
        module: &'a Module,
        kernel: &'a Kernel,
        bind_map: &'a HashMap<String, i64>,
        lines: &'a LineIndex,
    ) -> Self {
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

        let shape = SpecialVector::Ntid
            .components()
            .map(|reg| bind_map.get(&reg.name()).map(|&v| v as u32).unwrap_or(0));
        let mut b = KernelReportBuilder {
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
            accesses: HashMap::new(),
            access_forms: HashMap::new(),
            per_request: HashMap::new(),
            sets: ThreadSets::default(),
            unused: Vec::new(),
            shape,
            launches: HashMap::new(),
            lines,
        };
        let tracer =
            AffineValueTracer::new(module, kernel, &b.cfg, &b.forest).with_bindings(bind_map);
        let uniformity = tracer.uniformity();
        b.sets = thread_sets(module, kernel, &b.cfg, &b.forest, &tracer, uniformity);
        if let Some(grid) = b.grid() {
            for (block, blk) in b.cfg.blocks.iter_enumerated() {
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
        }
        let doms = dominators(&b.cfg.without(&b.sets.dead_edges));
        let always = |x: BlockId, targets: &[BlockId]| {
            let mut live = (targets.iter())
                .filter(|&&t| doms.is_reachable(t))
                .peekable();
            live.peek().is_some() && live.all(|&t| doms.dominates(x, t))
        };
        for bm in &mut b.blocks {
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
            b.cond_entry[i] &= !always(l.header, targets);
        }
        let compile_time: HashMap<String, i64> = (SpecialVector::Ntid.components().into_iter())
            .zip(kernel.reqntid.into_iter().flatten())
            .map(|(reg, n)| (reg.name(), i64::from(n)))
            .collect();
        let static_sets;
        let sets = match compile_time == *bind_map {
            true => &b.sets,
            false => {
                let t = AffineValueTracer::new(module, kernel, &b.cfg, &b.forest)
                    .with_bindings(&compile_time);
                static_sets = thread_sets(module, kernel, &b.cfg, &b.forest, &t, t.uniformity());
                &static_sets
            }
        };
        b.unused = unused(module, kernel, &tracer.ssa, |s| {
            match sets.never(tracer.block_of(s), s) {
                true => Some(false),
                false => (sets.guards.get(&s) == Some(&Pred::True)).then_some(true),
            }
        });
        let (accesses, access_forms, per_request) = b.accesses_by_scope(&tracer, uniformity);
        b.accesses = accesses;
        b.access_forms = access_forms;
        b.per_request = per_request;
        b
    }

    /// `%r83@106`: a value's register and the PTX line of its statement,
    /// with the statement's index on the line when it shares it;
    /// `%r34@$L__BB0_5` for a φ, its block.
    fn value_name(&self, v: Value) -> String {
        let reg = |r: Register| match r {
            Register::Named(s) => self.module.interner.resolve(s).to_owned(),
            Register::Cc => "CC.CF".to_owned(),
        };
        let line = |i: usize| match &self.kernel.stmts[i] {
            Stmt::Instr(instr) => Some(self.lines.line(instr.offset)),
            Stmt::Unparsed { offset, .. } => Some(self.lines.line(*offset)),
            _ => None,
        };
        match v {
            Value::Result { op, reg: r } => {
                let Some(at) = line(op) else {
                    return reg(r);
                };
                let on: Vec<usize> = (0..self.kernel.stmts.len())
                    .filter(|&i| line(i) == Some(at))
                    .collect();
                match on.iter().position(|&i| i == op) {
                    Some(i) if on.len() > 1 => format!("{}@{at}.{}", reg(r), i + 1),
                    _ => format!("{}@{at}", reg(r)),
                }
            }
            Value::Argument { block, reg: r } => {
                format!("{}@{}", reg(r), self.cfg.block_name(self.module, block))
            }
            Value::Undef(r) => reg(r),
        }
    }

    fn loop_bytes(&self, id: LoopId) -> Option<LoopBytes> {
        let trips = self.trip_exprs[id.0 as usize]
            .as_const()
            .filter(|&t| t > 0)?;
        let forms = self.access_forms.get(&Some(id))?;
        loop_bytes(id, trips, self.shape, forms, |key| self.pointer(key))
    }

    /// The pointer a loop footprint's base sits on: the one 64-bit
    /// parameter it names, with coefficient 1 in its symbolic part, when
    /// it names no other address (a parameter in a coefficient, a
    /// module-scope array) and no 64-bit uniform value. An offset from a
    /// pointer is taken to stay in its allocation.
    fn pointer(&self, key: &Affine) -> Option<String> {
        let pointer = |name: &str| {
            (name
                .strip_prefix("param_")
                .and_then(|i| i.parse::<usize>().ok()))
            .and_then(|i| self.kernel.params.get(i))
            .is_some_and(|p| p.ty.bytes() == Some(8))
        };
        let address =
            |name: &String| pointer(name) || !(name.starts_with("param_") || name.starts_with('%'));
        let mut coefficients = Vec::new();
        for (v, c) in &key.terms {
            let wide = v.uniform().is_some_and(|u| {
                let (Value::Result { reg, .. } | Value::Argument { reg, .. } | Value::Undef(reg)) =
                    u;
                !matches!(self.register_bytes(reg), Some(1 | 2 | 4))
            });
            if wide {
                return None;
            }
            coefficients.extend(c.symbols());
        }
        if coefficients.iter().any(address) {
            return None;
        }
        let [p] = &key
            .base
            .symbols()
            .into_iter()
            .filter(address)
            .collect::<Vec<_>>()[..]
        else {
            return None;
        };
        let direct = match &key.base {
            SymExpr::Sym(n) => n == p,
            SymExpr::Sum(ts) => ts.iter().any(|t| matches!(t, SymExpr::Sym(n) if n == p)),
            _ => false,
        };
        (direct && pointer(p)).then(|| p.clone())
    }

    /// The bytes a register's declaration gives it.
    fn register_bytes(&self, reg: Register) -> Option<u32> {
        let Register::Named(reg) = reg else {
            return None;
        };
        let name = self.module.interner.resolve(reg);
        (self.kernel.reg_decls.iter())
            .find(|d| {
                let prefix = self.module.interner.resolve(d.prefix);
                match d.count {
                    Some(n) => crate::ptx::parse::parser::member(name, prefix, n).is_some(),
                    None => name == prefix,
                }
            })
            .and_then(|d| d.class.bytes())
    }

    /// The threads a set selects, when a thread-index condition does.
    fn threads_info(&self, set: &ThreadSet) -> Option<ThreadsInfo> {
        let counted = set.count(self.shape);
        let warps = set.warps(self.shape);
        Some(ThreadsInfo {
            condition: set.render()?,
            count: counted.map(|c| u64::from(c.possible)),
            at_most: counted.map_or(!set.exact(), |c| c.certain != c.possible),
            warps: warps.map(|w| u64::from(w.possible)),
            warps_at_most: warps.map_or(!set.exact(), |w| w.certain != w.possible),
        })
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
                    threads: self.threads_info(&self.sets.blocks[b.0 as usize]),
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
        let file = self
            .module
            .file_name(first.file)
            .unwrap_or("<unknown file>");
        Some(if lo == hi {
            format!("{file}:{lo}")
        } else {
            format!("{file}:{lo}-{hi}")
        })
    }

    /// Loop chain of a block, innermost first.
    /// Whether statement `stmt` of block `b` counts: some thread runs it,
    /// and something with an effect reads its results.
    fn counted(&self, b: BlockId, stmt: usize) -> bool {
        !self.sets.never(b, stmt) && !self.unused[stmt]
    }

    fn chain(&self, b: BlockId) -> Vec<LoopId> {
        let mut out = Vec::new();
        let mut cur = self.forest.block_loop[b.0 as usize];
        while let Some(l) = cur {
            out.push(l);
            cur = self.forest.get(l).parent;
        }
        out
    }

    fn multiplier(&self, b: BlockId, below: Option<LoopId>) -> Option<(SymExpr, bool)> {
        let chain = self.chain(b);
        let cut = match below {
            Some(l) => chain.iter().position(|&x| x == l)?,
            None => chain.len(),
        };
        let mut mult = SymExpr::Const(1);
        let mut at_most = false;
        for &l in &chain[..cut] {
            mult = SymExpr::mul(mult, self.trip_exprs[l.0 as usize].clone());
            at_most |= self.cond_entry[l.0 as usize];
        }
        Some((mult, at_most))
    }

    // A block selected by thread-index branches runs on exactly
    // its threads: per CTA that count, and not a bound.
    fn executing(
        &self,
        bm: &BlockMeasurements,
        set: &ThreadSet,
        chain_at_most: bool,
        cta: Option<&LaunchInfo>,
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

    /// The threads and warps that run block `bm`, or with `guard` the
    /// statement it guards, and whether they are a bound: per thread,
    /// per CTA with `cta`, and over the launch with `ctas` too, at the
    /// launch's own count where the grid gives one. `None` when a count
    /// does not fit in u64.
    fn running(
        &self,
        bm: &BlockMeasurements,
        guard: Option<usize>,
        chain_at_most: bool,
        cta: Option<&LaunchInfo>,
        ctas: Option<u64>,
    ) -> Option<Running> {
        let b = bm.block;
        let set = match guard {
            Some(s) => self.sets.of(b, s),
            None => self.sets.blocks[b.0 as usize].clone(),
        };
        let (threads, warps, at_most, explained) = match (ctas, self.launches.get(&(b, guard))) {
            (Some(_), Some(&(t, w))) => {
                let bound = chain_at_most || cta.is_some_and(|c| !c.exact);
                (t, Some((w, bound)), bound, true)
            }
            (n, _) => {
                let n = n.unwrap_or(1);
                let (t, w, at_most, explained) = self.executing(bm, &set, chain_at_most, cta);
                let w = match w {
                    Some((w, bound)) => Some((u64::try_from(w).ok()?.checked_mul(n)?, bound)),
                    None => None,
                };
                (
                    u64::try_from(t).ok()?.checked_mul(n)?,
                    w,
                    at_most,
                    explained,
                )
            }
        };
        Some((threads, warps, at_most || guard.is_some() && !explained))
    }

    /// Statement `stmt` of block `bm`: it runs as the block's `running`
    /// unless a guard not every thread reads as true selects its own
    /// threads, and a predicated branch is a bound.
    fn statement(
        &self,
        bm: &BlockMeasurements,
        stmt: usize,
        block: Option<Running>,
        chain_at_most: bool,
        cta: Option<&LaunchInfo>,
        ctas: Option<u64>,
    ) -> Option<Running> {
        match self.sets.guards.get(&stmt) {
            Some(g) if *g != Pred::True => self.running(bm, Some(stmt), chain_at_most, cta, ctas),
            Some(_) => block,
            None => {
                let branch =
                    matches!(&self.kernel.stmts[stmt], Stmt::Instr(i) if i.predicate.is_some());
                block.map(|(t, w, at_most)| (t, w, at_most || branch))
            }
        }
    }

    /// Aggregate over blocks; `below` = aggregate within this loop
    /// (multipliers stop there), `None` = whole kernel. With
    /// `cta`, every per-thread count additionally scales by the CTA's
    /// thread count, and is an upper bound when that count is one. With
    /// `ctas` too, over the launch: a block or guarded instruction at its
    /// threads and warps over the grid where they are counted, else at
    /// its count per CTA times the CTAs; one under a loop whose trip
    /// count is not read is unknown, its trips perhaps differing by CTA.
    fn aggregates(
        &self,
        below: Option<LoopId>,
        cta: Option<&LaunchInfo>,
        ctas: Option<u64>,
    ) -> Aggregates {
        let mut flops: BTreeMap<Pipe, FlopAccumulator> = Pipe::ALL
            .iter()
            .map(|&p| (p, FlopAccumulator::new()))
            .collect();
        let mut bytes: BTreeMap<&'static str, (CountAccumulator, CountAccumulator)> =
            BTreeMap::new();
        let mut conversions = CountAccumulator::default();
        let mut instructions: BTreeMap<String, InstructionAccumulator> = BTreeMap::new();
        let mut instruction_total = IssueAccumulator::default();
        let mut instruction_all = IssueAccumulator::default();
        for s in ["global", "shared", "local"] {
            bytes.insert(s, Default::default());
        }

        for bm in &self.blocks {
            let Some((mult, chain_at_most)) = self.multiplier(bm.block, below) else {
                continue;
            };
            if self.sets.blocks[bm.block.0 as usize].pred == Pred::False {
                continue;
            }
            let unread = ctas.is_some()
                && (self.chain(bm.block).iter())
                    .any(|l| self.trip_info.trips[l.0 as usize].is_err());
            let trips = (!unread).then_some(mult);
            let trips = trips.as_ref();
            // A block whose count does not fit is unknown, its warps too.
            let block = self.running(bm, None, chain_at_most, cta, ctas);
            let narrow_block = block.and_then(narrow);
            let mult = trips.filter(|_| narrow_block.is_some());
            let (threads, warps, block_at_most) =
                narrow_block.unwrap_or((0, Some((0, false)), false));
            for instruction in &bm.instructions {
                instruction_all.add(threads, warps, mult, block_at_most);
                if !self.counted(bm.block, instruction.provenance) {
                    continue;
                }
                let kind = instructions
                    .entry(instruction_kind(&instruction.classified))
                    .or_default();
                kind.total.add(threads, warps, mult, block_at_most);
                kind.opcodes
                    .entry(instruction.opcode.clone())
                    .or_default()
                    .add(threads, warps, mult, block_at_most);
                kind.variants
                    .entry((
                        instruction.opcode.clone(),
                        instruction.classified.contributions.clone(),
                    ))
                    .or_default()
                    .add(threads, warps, mult, block_at_most);
                instruction_total.add(threads, warps, mult, block_at_most);
            }
            for m in &bm.measurements {
                if !self.counted(bm.block, m.provenance) {
                    continue;
                }
                let run = self.statement(bm, m.provenance, block, chain_at_most, cta, ctas);
                let (threads, at_most, mult) = match run.and_then(narrow) {
                    Some((threads, _, at_most)) => (threads, at_most, trips),
                    None => (0, false, None),
                };
                let n = (m.count as i64).checked_mul(threads);
                let (n, mult) = (n.unwrap_or(0), mult.filter(|_| n.is_some()));
                match m.kind {
                    MeasureKind::Flops { pipe, precision } => {
                        flops
                            .get_mut(&pipe)
                            .expect("every pipe pre-inserted")
                            .add(precision, n, mult, at_most);
                    }
                    MeasureKind::Bytes { space, direction } => {
                        let entry = bytes.entry(space.key()).or_default();
                        match direction {
                            Direction::Load => entry.0.add(n, mult, at_most),
                            Direction::Store => entry.1.add(n, mult, at_most),
                        }
                    }
                    MeasureKind::Conversions => conversions.add(n, mult, at_most),
                    MeasureKind::UnquantifiedBytes { space, direction } => {
                        let entry = bytes.entry(space.key()).or_default();
                        match direction {
                            Direction::Load => entry.0.add_unknown(),
                            Direction::Store => entry.1.add_unknown(),
                        }
                    }
                    MeasureKind::UnknownOps { .. } => {
                        flops.values_mut().for_each(FlopAccumulator::add_unknown);
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
                (Some(lb), Some(sb)) if i128::from(lb) + i128::from(sb) > 0 => {
                    let bytes_bound = direction(l.at_most || s.at_most, l.at_least || s.at_least);
                    ratio_bound(flops_bound, bytes_bound).map(|bound| Intensity {
                        value: f as f64 / (i128::from(lb) + i128::from(sb)) as f64,
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
            atomic_flops: flops[&Pipe::Atomic].counts(),
            bytes: bytes_out,
            conversions: conversions.count(),
            ai_global,
            unrolled_source_lines: self.unrolled_lines(below),
            instructions: InstructionCounts {
                all: Some(instruction_all.threads.count())
                    .filter(|all| *all != instruction_total.threads.count()),
                all_warps: (instruction_all.threads.count() != instruction_total.threads.count())
                    .then(|| instruction_all.warps())
                    .flatten(),
                total: instruction_total.threads.count(),
                warps: instruction_total.warps(),
                by_kind: instructions
                    .iter()
                    .map(|(k, acc)| {
                        let mut contribution_variants: BTreeMap<String, Vec<InstructionVariant>> =
                            BTreeMap::new();
                        for ((opcode, contributions), issued) in &acc.variants {
                            contribution_variants
                                .entry(opcode.clone())
                                .or_default()
                                .push(InstructionVariant {
                                    issued: issued.threads.count(),
                                    warps: issued.warps(),
                                    contributions_per_execution: contributions
                                        .iter()
                                        .map(contribution_details)
                                        .collect(),
                                });
                        }
                        (
                            k.clone(),
                            KindCounts {
                                total: acc.total.threads.count(),
                                warps: acc.total.warps(),
                                opcodes: acc
                                    .opcodes
                                    .iter()
                                    .map(|(o, v)| (o.clone(), v.threads.count()))
                                    .collect(),
                                opcode_warps: acc
                                    .opcodes
                                    .iter()
                                    .filter_map(|(o, v)| v.warps().map(|w| (o.clone(), w)))
                                    .collect(),
                                contribution_variants,
                            },
                        )
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
                let is_workload =
                    crate::analysis::instruction_counts::classify::classify(self.module, instr)
                        .contributions
                        .iter()
                        .any(|c| {
                            matches!(
                                c.kind,
                                MeasureKind::Flops { .. }
                                    | MeasureKind::Bytes { .. }
                                    | MeasureKind::UnquantifiedBytes { .. }
                            )
                        });
                if !is_workload {
                    continue;
                }
                let file = self.module.file_name(loc.file).unwrap_or("<unknown file>");
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
            per_iteration: self.aggregates(Some(id), None, None),
            accesses: self.accesses.get(&Some(id)).cloned().unwrap_or_default(),
            global_bytes_per_cta: self.loop_bytes(id),
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
                let mut weight = CountAccumulator::default();
                for &b in &l.blocks {
                    let instrs = (self.blocks[b.0 as usize].instructions.iter())
                        .filter(|i| self.counted(b, i.provenance))
                        .count() as i64;
                    let mut mult = SymExpr::Const(1);
                    for &cl in &self.chain(b) {
                        mult = SymExpr::mul(mult, self.trip_exprs[cl.0 as usize].clone());
                    }
                    weight.add(instrs, Some(&mult), false);
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
            .filter(|e| e.2.as_const() != Some(0))
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
            for m in (bm.measurements.iter()).filter(|m| self.counted(bm.block, m.provenance)) {
                match m.kind {
                    MeasureKind::UnknownOps { mnemonic } => {
                        *unknown_ops.entry(mnemonic.name().to_owned()).or_default() += m.count;
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
            .filter(|b| self.sets.blocks[b.block.0 as usize].pred != Pred::False)
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
            let header = self.forest.loops[i].header;
            if let Err(reason) = t
                && self.sets.blocks[header.0 as usize].pred != Pred::False
            {
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
                    target.map_or("?", |t| self.module.interner.resolve(t))
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
    /// See [`SharedMemory`].
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
                Some(count) => static_bytes += count * decl.ty.bytes().map_or(0, u64::from),
                None => dynamic = true,
            }
        }
        SharedMemory {
            static_bytes,
            dynamic,
        }
    }

    /// The grid, when bound.
    fn grid(&self) -> Option<[u32; 3]> {
        let [x, y, z] = (SpecialVector::Nctaid.components()).map(|reg| {
            self.bind_map
                .get(&reg.name())
                .and_then(|&n| u32::try_from(n).ok())
        });
        Some([x?, y?, z?])
    }

    fn measured(
        &self,
        record: &NcuRecord,
        lines: &LineIndex,
        cta: &LaunchInfo,
    ) -> Vec<MeasuredBlock> {
        let ctas = (record.grid.iter()).try_fold(1u64, |n, &d| n.checked_mul(d.into()));
        let mut per_line: HashMap<u32, usize> = HashMap::new();
        for stmt in &self.kernel.stmts {
            if let Stmt::Instr(i) = stmt {
                *per_line.entry(lines.line(i.offset)).or_default() += 1;
            }
        }
        let reducible = self.forest.irreducible_edges.is_empty();
        let mut blocks = Vec::new();
        for bm in &self.blocks {
            let Some((mult, chain_at_most)) = self.multiplier(bm.block, None) else {
                continue;
            };
            let block =
                ctas.and_then(|n| self.running(bm, None, chain_at_most, Some(cta), Some(n)));
            let mut predictions = Vec::new();
            for i in &bm.instructions {
                let Stmt::Instr(instr) = &self.kernel.stmts[i.provenance] else {
                    continue;
                };
                let run = ctas.and_then(|n| {
                    self.statement(bm, i.provenance, block, chain_at_most, Some(cta), Some(n))
                });
                let line = lines.line(instr.offset);
                let alone = reducible && per_line[&line] == 1;
                let never = !self.counted(bm.block, i.provenance);
                let trips = mult.as_const().filter(|_| alone);
                let at_most = !never && run.is_some_and(|(_, _, at_most)| at_most);
                // Warps with a lane that runs it: each requests its sectors.
                let requests = match never {
                    true => alone.then_some((0, false)),
                    false => trips.zip(run).and_then(|(m, (_, w, _))| {
                        let (w, bound) = w?;
                        Some((u64::try_from(m).ok()?.checked_mul(w)?, bound))
                    }),
                };
                let sectors = (self.per_request.get(&i.provenance).copied())
                    .zip(requests)
                    .and_then(|(r, (n, bound))| {
                        let total = |c: CountRange| {
                            Some(SectorCount {
                                min: u64::from(c.min).checked_mul(n)?,
                                max: u64::from(c.max).checked_mul(n)?,
                            })
                        };
                        Some(Sectors {
                            bits: r.bits,
                            requested: total(r.sectors)?,
                            ideal: total(r.ideal)?,
                            at_most: at_most || bound || r.at_most,
                        })
                    });
                predictions.push(Prediction {
                    line,
                    constraint: count_constraint(self.module, instr),
                    executions: match never {
                        true => alone.then_some(0),
                        false => trips
                            .zip(run)
                            .and_then(|(m, (t, _, _))| u64::try_from(m).ok()?.checked_mul(t)),
                    },
                    at_most,
                    sectors,
                });
            }
            let name = self.cfg.block_name(self.module, bm.block);
            blocks.push(record.compare(&name, &predictions));
        }
        blocks
    }

    fn build(
        self,
        launch_flag: Option<[u32; 3]>,
        ncu: Option<(&NcuRecord, &LineIndex)>,
    ) -> KernelReport {
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
                warps: block
                    .iter()
                    .map(|&d| d as u64)
                    .product::<u64>()
                    .div_ceil(32),
                source: source.to_owned(),
                exact: source != ".maxntid",
            });
        let totals_per_cta = launch
            .as_ref()
            .map(|l| self.aggregates(None, Some(l), None));
        let ctas = self
            .grid()
            .and_then(|g| g.iter().try_fold(1u64, |n, &d| n.checked_mul(d.into())));
        let totals_per_launch =
            (launch.as_ref().zip(ctas)).map(|(l, n)| self.aggregates(None, Some(l), Some(n)));

        let unknowns = self.unknowns();
        let measured = (ncu.filter(|(r, _)| r.kernel == name).zip(launch.as_ref()))
            .map(|((record, lines), cta)| self.measured(record, lines, cta));

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
                    ty: p.ty.name().to_owned(),
                    name: self.module.interner.resolve(p.name).to_owned(),
                })
                .collect(),
            blocks: self.blocks(),
            shared_memory: self.shared_memory(),
            instruction_classes: classes,
            most_instructions_loop: ranking.first().map(|(_, r)| r.loop_name.clone()),
            launch,
            accesses: self.accesses.get(&None).cloned().unwrap_or_default(),
            totals_per_cta,
            totals_per_launch,
            ranking: ranking.into_iter().map(|(_, r)| r).collect(),
            loops: self
                .forest
                .top_level()
                .into_iter()
                .map(|t| self.loop_node(t))
                .collect(),
            totals: self.aggregates(None, None, None),
            unknowns,
            unused: (self.kernel.stmts.iter().enumerate())
                .filter_map(|(s, stmt)| match stmt {
                    Stmt::Instr(i) if self.unused[s] => Some(UnusedInstruction {
                        line: self.lines.line(i.offset),
                        opcode: self.module.opcode(i),
                    }),
                    _ => None,
                })
                .collect(),
            measured,
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

    /// Blocks selected by thread-index branches: the two halves of a
    /// warp-specialized CTA and an elected thread. Per CTA their counts
    /// are exact; per thread they stay bounds.
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
        let r = analyze(src, "t", &opts).expect("analyzes");
        let k = &r.kernels[0];
        let threads: Vec<Option<String>> = k
            .blocks
            .iter()
            .map(|b| b.threads.as_ref().map(ThreadsInfo::render))
            .collect();
        assert_eq!(
            threads.iter().map(Option::as_deref).collect::<Vec<_>>(),
            [
                None,
                Some("128 in 4 warps (⌊%tid.x/32⌋ >= 4)"),
                Some("128 in 4 warps (⌊%tid.x/32⌋ < 4)"),
                None,
                Some("1 in 1 warp (%tid.x == 0)"),
                None,
            ]
        );
        // 128 threads add once, 128 multiply twice: 384 flops per CTA, exact.
        let cta = k.totals_per_cta.as_ref().expect("reqntid gives a launch");
        assert_eq!(cta.flops["f32"].expr, "384");
        assert!(!cta.flops["f32"].at_most);
        assert_eq!(cta.bytes["global"].store.expr, "4");
        assert!(!cta.bytes["global"].store.at_most);
        // Per thread the same counts are bounds: a thread runs one path.
        assert_eq!(k.totals.flops["f32"].expr, "3");
        assert!(k.totals.flops["f32"].at_most);
        // Triton broadcasts the warp index from lane 0 before comparing.
        let src = src.replace(
            "setp.lt.u32 %p1, %r2, 4;",
            "shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;\nsetp.lt.u32 %p1, %r3, 4;",
        );
        let r = analyze(&src, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].blocks[2]
                .threads
                .as_ref()
                .map(ThreadsInfo::render)
                .as_deref(),
            Some("128 in 4 warps (⌊%tid.x/32⌋ < 4)")
        );
    }

    #[test]
    fn guarded_predicate_definitions_and_deep_chains_stay_bounds() {
        let opts = AnalyzeOptions::default();
        let head = ".version 8.7\n.target sm_80\n.address_size 64\n\
                    .visible .entry k(\n.param .u64 k_param_0\n)\n\
                    .reqntid 64, 1, 1\n{\n\
                    ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n\
                    setp.lt.u32 %p0, %r1, 32;\nsetp.eq.u32 %p9, %r1, 0;\n";
        // A guarded redefinition of the guard register is not read.
        let src = format!(
            "{head}@%p9 setp.lt.u32 %p0, %r1, 64;\n@%p0 st.global.u32 [%rd1], %r1;\nret;\n}}\n"
        );
        let r = analyze(&src, "t", &opts).expect("analyzes");
        let cta = r.kernels[0].totals_per_cta.as_ref().expect("launch");
        assert_eq!(cta.bytes["global"].store.expr, "256");
        assert!(cta.bytes["global"].store.at_most);
        // Nine and.pred steps exceed the reading depth: a bound; the
        // base predicate read directly afterwards is still exact.
        let chain: String = (1..=9)
            .map(|i| format!("and.pred %p{i}, %p{}, %p{};\n", i - 1, i - 1))
            .collect();
        let src = format!(
            "{head}{chain}@%p9 st.global.u32 [%rd1], %r1;\n@%p0 st.global.u16 [%rd1+4], %rs1;\nret;\n}}\n"
        );
        let r = analyze(&src, "t", &opts).expect("analyzes");
        let cta = r.kernels[0].totals_per_cta.as_ref().expect("launch");
        // 64 × 4 B as a bound, plus 32 × 2 B exact.
        assert_eq!(cta.bytes["global"].store.expr, "320");
        assert!(cta.bytes["global"].store.at_most);
    }

    #[test]
    fn a_guard_every_thread_reads_as_true_keeps_counts_exact() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nsetp.eq.u32 %p1, 1, 1;\n\
                   @%p1 ld.global.u32 %r1, [%rd1];\nst.global.u32 [%rd1], %r1;\nret;\n}\n";
        let r = analyze(src, "t", &AnalyzeOptions::default()).expect("analyzes");
        let cta = r.kernels[0]
            .totals_per_cta
            .as_ref()
            .expect("reqntid gives a launch");
        assert_eq!(cta.bytes["global"].load.expr, "128");
        assert!(!cta.bytes["global"].load.at_most);
    }

    #[test]
    fn a_launch_shape_leaves_the_unused_instructions_alone() {
        // The store's block is reached only through two branches no
        // thread takes, so the load it alone reads is unused, with or
        // without a launch shape ptxas never sees.
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
            let r = analyze(src, "t", &opts).expect("analyzes");
            (r.kernels[0].unused.iter().map(|u| u.opcode.clone())).collect::<Vec<_>>()
        };
        assert!(unused(None).contains(&"ld.global.u32".to_owned()));
        assert_eq!(unused(None), unused(Some([32, 1, 1])));
    }

    #[test]
    fn laneid_is_the_lane_in_a_one_dimensional_block() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n\
                   .reqntid 64, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %laneid;\n\
                   setp.eq.u32 %p1, %r1, 0;\n@%p1 st.global.u32 [%rd1], %r1;\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        let cta = r.kernels[0].totals_per_cta.as_ref().expect("launch");
        assert_eq!(cta.bytes["global"].store.expr, "8");
        assert!(!cta.bytes["global"].store.at_most);
        // Two rows of 32: the lane is not a function of %tid.x alone.
        let src = src.replace(".reqntid 64, 1, 1", ".reqntid 32, 2, 1");
        let r = analyze(&src, "t", &opts).expect("analyzes");
        let cta = r.kernels[0].totals_per_cta.as_ref().expect("launch");
        assert_eq!(cta.bytes["global"].store.expr, "256");
        assert!(cta.bytes["global"].store.at_most);
    }

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
        let r = analyze(src, "t", &opts).expect("analyzes");
        // 32 lanes × 4 B × 8 trips per load; the first sweeps 1024 B in 4
        // or 5 sectors a trip, the second is one word in one sector.
        assert_eq!(
            r.kernels[0].loops[0].global_bytes_per_cta,
            Some(LoopBytes {
                requested: 2048,
                unique: Some(ByteRange {
                    min: 1028,
                    max: 1028
                }),
                sector_bytes: Some(ByteRange {
                    min: 1280,
                    max: 1536
                }),
                at_most: false,
            })
        );
        // Guarded on %tid.x != 0: the second load runs on 31 lanes.
        let guarded = src.replace("ld.global.f32 %f2", "@%p2 ld.global.f32 %f2");
        let r = analyze(&guarded, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].loops[0].global_bytes_per_cta,
            Some(LoopBytes {
                requested: 2016,
                unique: Some(ByteRange {
                    min: 1028,
                    max: 1028
                }),
                sector_bytes: Some(ByteRange {
                    min: 1280,
                    max: 1536
                }),
                at_most: false,
            })
        );
        // Guarded on a loaded value: every lane, as a bound.
        let loaded = guarded.replace(
            "setp.ne.s32 %p2, %r1, 0;",
            "ld.global.u32 %r9, [%rd5];\nsetp.ne.s32 %p2, %r9, 0;",
        );
        let r = analyze(&loaded, "t", &opts).expect("analyzes");
        let b = r.kernels[0].loops[0]
            .global_bytes_per_cta
            .clone()
            .expect("still enumerable");
        assert!(b.at_most);
        assert_eq!(
            (b.requested, b.unique),
            (
                2048,
                Some(ByteRange {
                    min: 1028,
                    max: 1028
                })
            )
        );
        // The second load in a block a loaded value skips: a bound even
        // though it is not guarded.
        let branched = src.replace(
            "ld.global.f32 %f2, [%rd5];",
            "ld.global.u32 %r9, [%rd5];\nsetp.ne.s32 %p3, %r9, 0;\n@%p3 bra $L__SKIP;\n\
             ld.global.f32 %f2, [%rd5];\n$L__SKIP:",
        );
        let r = analyze(&branched, "t", &opts).expect("analyzes");
        let b = r.kernels[0].loops[0]
            .global_bytes_per_cta
            .clone()
            .expect("still enumerable");
        assert!(b.at_most);
        assert_eq!(
            (b.requested, b.unique),
            (
                3072,
                Some(ByteRange {
                    min: 1028,
                    max: 1028
                })
            )
        );
        let src = src.replace(".reqntid 32, 1, 1\n", "");
        let r = analyze(&src, "t", &opts).expect("analyzes");
        assert_eq!(r.kernels[0].loops[0].global_bytes_per_cta, None);
    }

    #[test]
    fn a_floor_of_the_counter_over_too_many_trips_has_no_footprint() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, 0;\n\
                   $L__LOOP:\nadd.u32 %r1, %r1, 1;\nshr.u32 %r3, %r1, 1;\n\
                   mul.wide.u32 %rd2, %r3, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                   st.global.u32 [%rd3], 0;\nsetp.lt.u32 %p1, %r1, 1000000000;\n\
                   @%p1 bra $L__LOOP;\nret;\n}\n";
        let r = analyze(src, "t", &AnalyzeOptions::default()).expect("analyzes");
        assert_eq!(r.kernels[0].loops[0].global_bytes_per_cta, None);
    }

    #[test]
    fn a_counter_whose_update_leaves_its_type_is_no_counter() {
        // %r1 steps by 2^30 for 8 trips: its update reaches 2^33, past
        // 32 bits, though the store reads it before the update. It is no
        // counter, but one value for every thread on each trip.
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
        let address = |step: &str| {
            let r = analyze(&src(step), "t", &AnalyzeOptions::default()).expect("analyzes");
            r.kernels[0].loops[0].accesses[0].address.clone()
        };
        assert_eq!(
            address("1"),
            Some("4 * k[$L__LOOP] + param_0 - 4".to_owned())
        );
        assert_eq!(
            address("1073741824"),
            Some("4 * uniform(%r1@$L__LOOP) + param_0".to_owned())
        );
    }

    /// `%r10` is a 64-bit `selp` of 0 and `q − p`, declared after a
    /// 32-bit `%r<2>` that does not hold it: `p + %r10` may be `q`.
    #[test]
    fn a_wide_uniform_offset_names_no_pointer() {
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
        let r = analyze(src, "t", &AnalyzeOptions::default()).expect("analyzes");
        let b = r.kernels[0].loops[0].global_bytes_per_cta.clone().unwrap();
        assert_eq!(
            b.unique,
            Some(ByteRange {
                min: 512,
                max: 1024
            })
        );
    }

    #[test]
    fn an_access_of_no_bytes_leaves_a_loop_footprint_whole() {
        // A zero-source cp.async through a loaded pointer: its address is
        // unknown, but it reads no global bytes.
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   .shared .align 16 .b8 dst[512];\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, dst;\nmov.u32 %r2, 0;\n\
                   $L__LOOP:\nld.global.u64 %rd4, [%rd1];\n\
                   cp.async.cg.shared.global [%r1], [%rd4], 16, 0;\n\
                   add.s32 %r2, %r2, 1;\nsetp.lt.u32 %p1, %r2, 2;\n@%p1 bra $L__LOOP;\n\
                   cp.async.wait_all;\nret;\n}\n";
        let r = analyze(src, "t", &AnalyzeOptions::default()).expect("analyzes");
        let b = (r.kernels[0].loops[0].global_bytes_per_cta.clone()).expect("enumerable");
        assert_eq!(
            (b.requested, b.unique),
            (512, Some(ByteRange { min: 8, max: 8 }))
        );
    }

    #[test]
    fn accesses_are_affine_addresses() {
        let opts = AnalyzeOptions::default();
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
        let r = analyze(src, "t", &opts).expect("analyzes");
        let k = &r.kernels[0];
        assert_eq!(k.accesses.len(), 1);
        assert_eq!(k.accesses[0].opcode, "ld.shared.u16");
        assert_eq!(
            k.accesses[0].address.as_deref(),
            Some("16 * ⌊%tid.x/8⌋ + 2 * (%tid.x mod 8) + As + 2")
        );
        let inner = &k.loops[0].accesses;
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].direction, "load");
        assert_eq!(inner[0].bytes, Some(4));
        assert_eq!(
            inner[0].address.as_deref(),
            Some("4 * k[$L__L] + param_0 - 4")
        );
        assert_eq!(
            inner[0].reuse,
            [Reuse {
                r#loop: "$L__L".to_owned(),
                stride: Some("4".to_owned())
            }]
        );
        assert!(k.accesses[0].reuse.is_empty());
    }

    /// A guard `%tid.x < 16 and min(p, q) < 7`: the second half is not
    /// read but uniform, so each warp's lanes are still known.
    /// §10.32: `%dynamic_smem_size` is set "for the CTA … at launch time",
    /// so a branch on it keeps a warp whole; §10.23: `%clock` is a cycle
    /// counter, so a branch on it may split one.
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
            let r = analyze(&src, "t", &opts).expect("analyzes");
            let a = &r.kernels[0].accesses[0];
            (a.sectors_at_most, a.sectors_per_request)
        };
        assert_eq!(
            sectors("%dynamic_smem_size"),
            (false, Some(CountRange { min: 4, max: 5 }))
        );
        assert_eq!(
            sectors("%clock"),
            (true, Some(CountRange { min: 5, max: 5 }))
        );
    }

    #[test]
    fn a_uniform_half_of_an_and_keeps_the_lanes_exact() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0,\n.param .u32 k_param_1\n)\n\
                   .reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nld.param.u32 %r3, [k_param_1];\n\
                   mov.u32 %r1, %tid.x;\nsetp.lt.u32 %p1, %r1, 16;\n\
                   max.s32 %r5, %r3, 9;\nmin.s32 %r6, %r5, %r3;\n\
                   setp.lt.s32 %p2, %r6, 7;\nand.pred %p3, %p1, %p2;\n\
                   mul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd3, %rd1, %rd2;\n\
                   @%p3 ld.global.f32 %f1, [%rd3];\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        let a = &r.kernels[0].accesses[0];
        assert!(!a.sectors_at_most);
        assert_eq!(a.sectors_per_request, Some(CountRange { min: 2, max: 3 }));
    }

    /// `mov.b64 {%r2, %r3}, %rd1` writes %r2 as half of %rd1, not %rd1:
    /// an address through it is a uniform variable, not param_0.
    #[test]
    fn a_vector_destination_is_not_the_whole_source() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.b64 {%r2, %r3}, %rd1;\n\
                   cvt.u64.u32 %rd2, %r2;\nld.global.f32 %f1, [%rd2];\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].accesses[0].address.as_deref(),
            Some("uniform(%r2@9)")
        );
    }

    /// The base at depth 0, `%r30` at 2 and `%r0` at 32, the deepest
    /// definition the tracer follows: `%tid.x` below it is a leaf.
    #[test]
    fn a_special_register_at_the_depth_limit_is_read() {
        let opts = AnalyzeOptions::default();
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
        let r = analyze(&src, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].accesses[0].address.as_deref(),
            Some("4 * %tid.x + param_0 + 120")
        );
    }

    /// PTX ISA §6.4.3: `{Ra, Rb, Rc, Rd}` take `V.x`, `V.y`, `V.z`, `V.w`;
    /// §10.1: `%tid`'s fourth element "always returns zero".
    #[test]
    fn a_special_vector_is_read_by_element() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\n\
                   mov.v4.u32 {%r1, %r2, %r3, %r4}, %tid;\nmov.u32 %r5, %tid.a;\n\
                   mul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd3, %rd1, %rd2;\nld.global.f32 %f1, [%rd3];\n\
                   mul.wide.u32 %rd4, %r4, 4;\nadd.s64 %rd5, %rd1, %rd4;\nld.global.f32 %f2, [%rd5];\n\
                   mul.wide.u32 %rd6, %r5, 4;\nadd.s64 %rd7, %rd1, %rd6;\nld.global.f32 %f3, [%rd7];\n\
                   ret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        let addresses: Vec<_> = (r.kernels[0].accesses.iter())
            .map(|a| a.address.as_deref())
            .collect();
        assert_eq!(
            addresses,
            [
                Some("4 * %tid.x + param_0"),
                Some("param_0"),
                Some("param_0")
            ]
        );
    }

    /// `ld.param.v2.u32 {%r1, %r2}` loads two words of the parameter: %r2
    /// is the second, a uniform variable, not param_0.
    #[test]
    fn a_vector_parameter_load_element_is_not_the_parameter() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .align 8 .b8 k_param_0[8]\n)\n{\n\
                   ld.param.v2.u32 {%r1, %r2}, [k_param_0];\n\
                   cvt.u64.u32 %rd2, %r2;\nld.global.f32 %f1, [%rd2];\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].accesses[0].address.as_deref(),
            Some("uniform(%r2@8)")
        );
    }

    /// Both paths write `{%r2, %r3}` from one 64-bit immediate: %r3 is its
    /// upper half, not the immediate, the same for every thread.
    #[test]
    fn merged_vector_writes_of_one_immediate_are_not_that_immediate() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u32 k_param_0\n)\n{\n\
                   ld.param.u32 %r1, [k_param_0];\nsetp.eq.u32 %p1, %r1, 0;\n\
                   @%p1 bra $L1;\nmov.b64 {%r2, %r3}, 1;\nbra.uni $L2;\n\
                   $L1:\nmov.b64 {%r2, %r3}, 1;\n$L2:\n\
                   cvt.u64.u32 %rd2, %r3;\nld.global.f32 %f1, [%rd2];\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        let access = &r.kernels[0].accesses[0];
        assert_eq!(
            (access.address.as_deref(), access.unknown.as_deref()),
            (Some("uniform(%r3@$L2)"), None)
        );
    }

    /// An `or` whose constant is the high side: the lane part fits below it.
    #[test]
    fn an_or_above_the_other_sides_largest_value_is_an_add() {
        let opts = AnalyzeOptions::default();
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\n\
                   and.b32 %r2, %r1, 31;\nor.b32 %r3, %r2, 256;\nmul.wide.u32 %rd2, %r3, 4;\n\
                   add.s64 %rd3, %rd1, %rd2;\nld.global.f32 %f1, [%rd3];\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].accesses[0].address.as_deref(),
            Some("4 * (%tid.x mod 32) + param_0 + 1024")
        );
        // No power of two fits above 2^62 in an i64: that order is refused
        // and the other, the 0 below 1, still read.
        let wide = src.replace(
            "mov.u32 %r1, %tid.x;\n",
            "mov.u64 %rd8, 0x4000000000000000;\nor.b64 %rd9, %rd8, 0;\nadd.s64 %rd1, %rd1, %rd9;\nmov.u32 %r1, %tid.x;\n",
        );
        let r = analyze(&wide, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].accesses[0].address.as_deref(),
            Some("4 * (%tid.x mod 32) + param_0 + 4611686018427388928")
        );
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
                   st.global.f32 [%rd1], %f2;\nwgmma.fence.sync.aligned;\n\
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
        let line = crate::report::measured::NcuLine {
            ptx: "st.global.f32 [%rd1], %f1;".into(),
            sass: [("STG.E".to_owned(), 1)].into(),
            executions: 32,
            l2_theoretical_sectors_global: Some(1),
            l2_theoretical_sectors_global_ideal: Some(1),
            global_access_sizes: vec![32],
            global_executions: Some(32),
        };
        let record = NcuRecord {
            kernel: "k".into(),
            grid: [2147483647, 65535, 65535],
            block: [32, 1, 1],
            params: vec!["ptr:4".into()],
            lines: [(9, line)].into(),
        };
        let opts = AnalyzeOptions {
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        let report = analyze(src, "k.ptx", &opts).expect("analyzes");
        let blocks = report.kernels[0].measured.as_ref().expect("measured");
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
        let line = crate::report::measured::NcuLine {
            ptx: "st.global.f32 [%rd1], %f1;".into(),
            sass: [("STG.E".to_owned(), 1)].into(),
            executions: 1 << 63,
            l2_theoretical_sectors_global: Some(1),
            l2_theoretical_sectors_global_ideal: Some(1),
            global_access_sizes: vec![32],
            global_executions: Some(32),
        };
        let record = NcuRecord {
            kernel: "k".into(),
            grid: [1 << 30, 32768, 8192],
            block: [32, 1, 1],
            params: vec!["ptr:4".into()],
            lines: [(9, line)].into(),
        };
        let opts = AnalyzeOptions {
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        let report = analyze(src, "k.ptx", &opts).expect("analyzes");
        let blocks = report.kernels[0].measured.as_ref().expect("measured");
        assert_eq!(blocks[0].with_sass, 1);
        assert_eq!(blocks[0].agreeing, 1);
        assert!(blocks[0].differing.is_empty());
    }

    /// A loop of 2^62 trips whose 1024 threads each read 4 bytes: the
    /// bytes it requests pass 2^64, so it has no byte counts.
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
        let opts = |n| AnalyzeOptions {
            bindings: vec![parse_bind(&format!("1:N={n}")).expect("binding")],
            launch: Some([1024, 1, 1]),
            ..Default::default()
        };
        let bytes = |n: i64| {
            let report = analyze(src, "k.ptx", &opts(n)).expect("analyzes");
            report.kernels[0].loops[0].global_bytes_per_cta.clone()
        };
        assert_eq!(bytes(4).map(|b| b.requested), Some(32768));
        assert_eq!(bytes(1 << 62), None);
    }
}
