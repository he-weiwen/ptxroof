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

use crate::analysis::control_flow::loop_forest;
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::instruction_counts::classify::{
    ArithKind, ClassifiedInstruction, Direction, InstructionCategory, Pipe, Precision, Space,
};
use crate::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier, collect};
use crate::analysis::instruction_counts::measurement::{Contribution, MeasureKind};
use crate::analysis::loop_names::{LoopName, loop_names};
use crate::analysis::memory_footprint::warp_footprint;
use crate::analysis::scalar::affine::{Affine, Var};
use crate::analysis::scalar::symexpr::SymExpr;
use crate::analysis::scalar::trace::AffineValueTracer;
use crate::analysis::scalar::trip_counts::{TripCountResults, trip_counts};
use crate::analysis::thread_participation::{ThreadSet, thread_sets};
use crate::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
use crate::ptx::ir::Operand;
use crate::ptx::ir::{Instr, Kernel, Module, Stmt};
use crate::ptx::parse::parser::{ParseError, parse};
use crate::report::names::demangle;
use crate::report::schema::*;
use crate::support::paths::basename;
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
        let mut bind_map = resolve_bindings(&module, kernel, &opts.bindings, &mut bindings_echo)?;
        // The block shape, when it is the launch's: --launch, else .reqntid.
        if let Some([x, y, z]) = opts.launch.or(kernel.reqntid) {
            for (name, n) in [("%ntid.x", x), ("%ntid.y", y), ("%ntid.z", z)] {
                bind_map.insert(name.to_owned(), i64::from(n));
            }
        }
        let k = KernelReportBuilder::new(&module, kernel, &bind_map).build(opts.launch);
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

/// Every memory operand of every memory instruction, traced to an
/// affine address at its own position, grouped by the innermost loop
/// of its block. `cp.async` gives a store row for its shared
/// destination and a load row for its global source; an atomic gives
/// one `load+store` row. Parameter loads are not accesses.
fn accesses_by_scope(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    display: &[String],
    bind_map: &HashMap<String, i64>,
    tracer: &AffineValueTracer,
) -> (ByScope<Access>, ByScope<AccessForm>) {
    let mut forms: ByScope<AccessForm> = HashMap::new();
    let name = |v: &Var| match v {
        Var::Iter(l) => format!("k[{}]", display[l.0 as usize]),
        other => other.to_string(),
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
    let shape =
        ["%ntid.x", "%ntid.y", "%ntid.z"].map(|n| bind_map.get(n).map(|&v| v as u32).unwrap_or(0));
    let mut out: HashMap<Option<LoopId>, Vec<Access>> = HashMap::new();
    for (bid, block) in cfg.blocks.iter_enumerated() {
        let scope = forest.block_loop[bid.0 as usize];
        for (si, stmt) in kernel.stmts[block.start..block.end].iter().enumerate() {
            let Stmt::Instr(instr) = stmt else { continue };
            let pos = block.start + si;
            let mem_ops: Vec<_> = module
                .operand_ids(instr.operands)
                .iter()
                .copied()
                .filter(|&id| matches!(module.operand(id), Operand::Memory { .. }))
                .collect();
            let classified = crate::analysis::instruction_counts::classify::classify(module, instr);
            // Pair memory contributions by their explicit operand association.
            // An atomic has one address with read+write; a copy has two addresses.
            let mut accesses: BTreeMap<(usize, Space, Option<u32>), (bool, bool)> = BTreeMap::new();
            for c in &classified.contributions {
                let Some(operand) = c.memory_operand.filter(|&i| i < mem_ops.len()) else {
                    continue;
                };
                let (space, direction, bytes) = match c.kind {
                    MeasureKind::Bytes { space, direction } => {
                        (space, direction, Some(c.count as u32))
                    }
                    MeasureKind::UnquantifiedBytes { space, direction } => (space, direction, None),
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
                let Operand::Memory { base, offset } = module.operand(mem_ops[i]) else {
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
                    Operand::Register(_) => match tracer.trace_operand(*base, pos, scope, 0) {
                        Ok(a) => {
                            let a = (a + Affine::invariant(SymExpr::Const(*offset))).bind(bind_map);
                            (Some(shorten(a.render(name))), None, Some(a))
                        }
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
                let footprint = match (&form, bytes) {
                    (Some(a), Some(b)) if matches!(space, Space::Global | Space::Generic) => {
                        Some(warp_footprint(a, b, shape))
                    }
                    _ => None,
                };
                let (sectors_per_request, lines_per_request, footprint_unknown) = match footprint {
                    Some(Ok(f)) => (Some(f.sectors), Some(f.lines), None),
                    Some(Err(why)) => (None, None, Some(why)),
                    None => (None, None, None),
                };
                if let (Some(a), Some(b)) = (&form, bytes) {
                    forms.entry(scope).or_default().push(AccessForm {
                        form: a.clone(),
                        bytes: b,
                        space,
                        block: bid,
                        predicated: instr.predicate.is_some(),
                    });
                }
                let mut reuse = Vec::new();
                if let Some(a) = &form {
                    let mut cur = scope;
                    while let Some(l) = cur {
                        reuse.push(Reuse {
                            r#loop: display[l.0 as usize].clone(),
                            stride: a.terms.get(&Var::Iter(l)).map(ToString::to_string),
                        });
                        cur = forest.get(l).parent;
                    }
                }
                let site = match instr.loc.filter(|l| l.line != 0) {
                    Some(loc) => format!(
                        "{}:{}",
                        basename(module.file_path(loc.file).unwrap_or("<unknown file>")),
                        loc.line
                    ),
                    None => cfg.block_name(module, bid),
                };
                let mods: Vec<&str> = module
                    .modifiers(instr)
                    .iter()
                    .map(|&m| module.interner.resolve(m))
                    .collect();
                out.entry(scope).or_default().push(Access {
                    site,
                    opcode: module.opcode(instr),
                    space: space.key().to_owned(),
                    direction: direction.to_owned(),
                    bytes,
                    predicated: instr.predicate.is_some(),
                    path: cache_path(
                        module.interner.resolve(instr.mnemonic),
                        &mods,
                        space,
                        direction,
                    ),
                    address,
                    unknown,
                    sectors_per_request,
                    lines_per_request,
                    footprint_unknown,
                    reuse,
                });
            }
        }
    }
    (out, forms)
}

/// Rows per scope: a block's innermost loop, or `None` outside every loop.
type ByScope<T> = HashMap<Option<LoopId>, Vec<T>>;

/// A memory operand's bound affine address, for the per-loop footprint.
struct AccessForm {
    form: Affine,
    bytes: u32,
    space: Space,
    block: BlockId,
    predicated: bool,
}

/// The cache level an access can hit at. PTX ISA §9.7.9.1, Tables 30
/// and 31: loads default to `.ca` (all levels), `.cg` caches in L2
/// only, `.cs` and `.lu` allocate evict-first, `.cv` fetches again;
/// stores default to `.wb`, with `.cg`, `.cs` and `.wt`. `ld.global.nc`
/// is the read-only data path; `cp.async` takes `.ca` or `.cg`.
fn cache_path(mnemonic: &str, mods: &[&str], space: Space, direction: &str) -> String {
    let has = |m: &str| mods.contains(&m);
    let base = match space {
        Space::Shared | Space::SharedCluster => return "shared memory".to_owned(),
        Space::Const => return "constant cache".to_owned(),
        _ if mnemonic == "atom" || mnemonic == "red" => "L2 (atomics)",
        _ if has("nc") => "read-only path (.nc)",
        _ if has("cg") => "L2 only (.cg)",
        _ if has("cs") => "evict-first streaming (.cs)",
        _ if has("lu") => "evict-first streaming (.lu)",
        _ if has("cv") => "no cache (.cv)",
        _ if has("wt") => "write-through (.wt)",
        _ if direction == "store" => "write-back (.wb)",
        _ => "L1 and L2",
    };
    let mut path = base.to_owned();
    if let Some(hint) = mods.iter().find(|m| m.starts_with("L2::")) {
        path.push_str(&format!(" with .{hint}"));
    }
    path
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
struct CountAccumulator {
    groups: Vec<(SymExpr, i64)>,
    at_most: bool,
    at_least: bool,
    touched: bool,
}

impl CountAccumulator {
    fn add(&mut self, count: i64, mult: &SymExpr, at_most: bool) {
        let (coeff, rest) = SymExpr::split_const(mult.clone());
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

fn contribution_details(c: &Contribution, module: &Module) -> ContributionDetails {
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
            mnemonic: module.interner.resolve(mnemonic).to_owned(),
        },
    }
}

#[derive(Default)]
struct InstructionAccumulator {
    total: CountAccumulator,
    opcodes: BTreeMap<String, CountAccumulator>,
    variants: BTreeMap<(String, Vec<Contribution>), CountAccumulator>,
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
    /// Which threads run each block.
    thread_sets: Vec<ThreadSet>,
    /// Which threads run each guarded instruction, by statement index.
    instr_sets: HashMap<usize, ThreadSet>,
    /// The block shape when known (`--launch` or `.reqntid`), else zeros.
    shape: [u32; 3],
}

impl<'a> KernelReportBuilder<'a> {
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

        let tracer = AffineValueTracer::new(module, kernel, &cfg, &forest).with_bindings(bind_map);
        let (accesses, access_forms) =
            accesses_by_scope(module, kernel, &cfg, &forest, &display, bind_map, &tracer);
        let (thread_sets, instr_sets) = thread_sets(module, kernel, &cfg, &forest, &tracer);
        let shape = ["%ntid.x", "%ntid.y", "%ntid.z"]
            .map(|n| bind_map.get(n).map(|&v| v as u32).unwrap_or(0));
        KernelReportBuilder {
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
            accesses,
            access_forms,
            thread_sets,
            instr_sets,
            shape,
        }
    }

    /// Global bytes one CTA requests over one execution of loop `id`'s
    /// own blocks, and the distinct bytes they touch: every lane of every
    /// executing thread at every iteration, as intervals merged per
    /// base (the form without its lane, counter and constant parts:
    /// forms on different pointers are assumed disjoint). Needs a
    /// numeric trip count, the block shape, and every global address's
    /// lane and counter coefficients as constants.
    fn loop_bytes(&self, id: LoopId) -> Option<LoopBytes> {
        let trips = self.trip_exprs[id.0 as usize]
            .as_const()
            .filter(|&t| t > 0)?;
        let [nx, ny, nz] = self.shape.map(i64::from);
        let threads = nx * ny * nz;
        if threads == 0 {
            return None;
        }
        let forms = self.access_forms.get(&Some(id))?;
        let global: Vec<&AccessForm> = forms
            .iter()
            .filter(|f| matches!(f.space, Space::Global | Space::Generic))
            .collect();
        if global.is_empty() {
            return None;
        }
        let mut requested = 0u64;
        let mut at_most = false;
        let mut per_base: HashMap<Affine, Vec<(i64, i64)>> = HashMap::new();
        for f in global {
            let stride = match f.form.terms.get(&Var::Iter(id)) {
                Some(c) => c.as_const()?,
                None => 0,
            };
            if f.form.terms.iter().any(|(v, c)| {
                crate::analysis::scalar::lane_eval::depends_on_lane(v) && c.as_const().is_none()
            }) {
                return None;
            }
            let set = &self.thread_sets[f.block.0 as usize];
            let bytes = i64::from(f.bytes);
            at_most |= f.predicated;
            let mut key = f.form.clone();
            key.terms.retain(|v, _| {
                !crate::analysis::scalar::lane_eval::depends_on_lane(v) && *v != Var::Iter(id)
            });
            key.base = SymExpr::sub(key.base.clone(), SymExpr::Const(key.base.const_part()));
            let intervals = per_base.entry(key).or_default();
            for t in 0..threads {
                let tid = [t % nx, (t / nx) % ny, t / (nx * ny)];
                if !set.contains(tid, self.shape) {
                    continue;
                }
                requested += f.bytes as u64 * trips as u64;
                let base = crate::analysis::scalar::lane_eval::eval_lane(&f.form, tid);
                if intervals.len() as i64 + trips > 1 << 22 {
                    return None;
                }
                for k in 0..trips {
                    let start = base + stride * k;
                    intervals.push((start, start + bytes));
                }
            }
        }
        let mut unique = 0i64;
        for mut intervals in per_base.into_values() {
            intervals.sort_unstable();
            let mut cur: Option<(i64, i64)> = None;
            for (s, e) in intervals {
                match cur {
                    Some((cs, ce)) if s <= ce => cur = Some((cs, ce.max(e))),
                    Some((cs, ce)) => {
                        unique += ce - cs;
                        cur = Some((s, e));
                    }
                    None => cur = Some((s, e)),
                }
            }
            if let Some((cs, ce)) = cur {
                unique += ce - cs;
            }
        }
        Some(LoopBytes {
            requested,
            unique: unique as u64,
            at_most,
        })
    }

    /// The block's selected threads as text, when a branch selects them.
    fn thread_set_text(&self, b: BlockId) -> Option<String> {
        let set = &self.thread_sets[b.0 as usize];
        let cond = set.render()?;
        let bound = if set.exact() { "" } else { "<= " };
        Some(match set.count(self.shape) {
            Some(n) => format!("{bound}{n} ({cond})"),
            None => format!("{bound}({cond})"),
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
                    threads: self.thread_set_text(b),
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
        let mut flops: BTreeMap<Pipe, FlopAccumulator> = Pipe::ALL
            .iter()
            .map(|&p| (p, FlopAccumulator::new()))
            .collect();
        let mut bytes: BTreeMap<&'static str, (CountAccumulator, CountAccumulator)> =
            BTreeMap::new();
        let mut conversions = CountAccumulator::default();
        let mut instructions: BTreeMap<String, InstructionAccumulator> = BTreeMap::new();
        let mut instruction_total = CountAccumulator::default();
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
            // A block selected by thread-index branches runs on exactly
            // its threads: per CTA that count, and not a bound.
            let resolve = |set: &ThreadSet| {
                let selected = cta.and(set.count(self.shape));
                let explained = selected.is_some() && set.exact();
                let conditional = bm.qualifier == CountQualifier::AtMost && !explained;
                let at_most = conditional || chain_at_most || cta.is_some_and(|c| !c.exact);
                let threads = match (cta, selected) {
                    (Some(_), Some(n)) => i64::from(n),
                    (Some(c), None) => c.threads as i64,
                    (None, _) => 1,
                };
                (threads, at_most, explained)
            };
            let (threads, block_at_most, _) = resolve(&self.thread_sets[bm.block.0 as usize]);
            for instruction in &bm.instructions {
                let kind = instructions
                    .entry(instruction_kind(&instruction.classified))
                    .or_default();
                kind.total.add(threads, &mult, block_at_most);
                kind.opcodes
                    .entry(instruction.opcode.clone())
                    .or_default()
                    .add(threads, &mult, block_at_most);
                kind.variants
                    .entry((
                        instruction.opcode.clone(),
                        instruction.classified.contributions.clone(),
                    ))
                    .or_default()
                    .add(threads, &mult, block_at_most);
                instruction_total.add(threads, &mult, block_at_most);
            }
            for m in &bm.measurements {
                let (threads, at_most) = match self.instr_sets.get(&m.provenance) {
                    Some(set) => {
                        let (threads, at_most, explained) = resolve(set);
                        (threads, at_most || !explained)
                    }
                    None => (threads, block_at_most || m.predicated),
                };
                let n = m.count as i64 * threads;
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
            atomic_flops: flops[&Pipe::Atomic].counts(),
            bytes: bytes_out,
            conversions: conversions.count(),
            ai_global,
            unrolled_source_lines: self.unrolled_lines(below),
            instructions: InstructionCounts {
                total: instruction_total.count(),
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
                                    issued: issued.count(),
                                    contributions_per_execution: contributions
                                        .iter()
                                        .map(|c| contribution_details(c, self.module))
                                        .collect(),
                                });
                        }
                        (
                            k.clone(),
                            KindCounts {
                                total: acc.total.count(),
                                opcodes: acc
                                    .opcodes
                                    .iter()
                                    .map(|(o, v)| (o.clone(), v.count()))
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
                    let width = crate::analysis::instruction_counts::classify::type_width(ty)
                        .unwrap_or(1) as u64;
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
            accesses: self.accesses.get(&None).cloned().unwrap_or_default(),
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
        let threads: Vec<Option<&str>> = k.blocks.iter().map(|b| b.threads.as_deref()).collect();
        assert_eq!(
            threads,
            [
                None,
                Some("128 (⌊%tid.x/32⌋ >= 4)"),
                Some("128 (⌊%tid.x/32⌋ < 4)"),
                None,
                Some("1 (%tid.x == 0)"),
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
            r.kernels[0].blocks[2].threads.as_deref(),
            Some("128 (⌊%tid.x/32⌋ < 4)")
        );
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
                   @%p1 bra $L__LOOP;\nret;\n}\n";
        let r = analyze(src, "t", &opts).expect("analyzes");
        // 32 lanes × 4 B × 8 trips per load; the first sweeps 1024 B, the
        // second is one word.
        assert_eq!(
            r.kernels[0].loops[0].global_bytes_per_cta,
            Some(LoopBytes {
                requested: 2048,
                unique: 1028,
                at_most: false,
            })
        );
        let src = src.replace("ld.global.f32 %f2", "@%p2 ld.global.f32 %f2");
        let r = analyze(&src, "t", &opts).expect("analyzes");
        let b = r.kernels[0].loops[0]
            .global_bytes_per_cta
            .clone()
            .expect("still enumerable");
        assert!(b.at_most);
        assert_eq!((b.requested, b.unique), (2048, 1028));
        let src = src.replace(".reqntid 32, 1, 1\n", "");
        let r = analyze(&src, "t", &opts).expect("analyzes");
        assert_eq!(r.kernels[0].loops[0].global_bytes_per_cta, None);
    }

    #[test]
    fn cache_paths_follow_the_operators() {
        let p = |m: &str, mods: &[&str], s, d| cache_path(m, mods, s, d);
        assert_eq!(
            p("ld", &["global", "u16"], Space::Global, "load"),
            "L1 and L2"
        );
        assert_eq!(
            p("ld", &["global", "nc", "v4", "f32"], Space::Global, "load"),
            "read-only path (.nc)"
        );
        assert_eq!(
            p(
                "cp",
                &["async", "cg", "shared", "global"],
                Space::Global,
                "load"
            ),
            "L2 only (.cg)"
        );
        assert_eq!(
            p("st", &["global", "cs", "f32"], Space::Global, "store"),
            "evict-first streaming (.cs)"
        );
        assert_eq!(
            p("st", &["global", "b32"], Space::Global, "store"),
            "write-back (.wb)"
        );
        assert_eq!(
            p("ld", &["global", "L2::128B", "f32"], Space::Global, "load"),
            "L1 and L2 with .L2::128B"
        );
        assert_eq!(
            p("red", &["global", "add", "f32"], Space::Global, "store"),
            "L2 (atomics)"
        );
        assert_eq!(
            p(
                "ldmatrix",
                &["sync", "aligned", "m8n8", "x4", "shared", "b16"],
                Space::Shared,
                "load"
            ),
            "shared memory"
        );
    }

    /// Each memory operand's address as an affine form: a 2D tile's
    /// shared row and column from the thread index, and a pointer that
    /// steps through a loop, read before its increment.
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
