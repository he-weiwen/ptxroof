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

use crate::analysis::affine::affine_analysis::AffineAnalysis;
use crate::analysis::affine::affine_expr::{AffineExpr, Var};
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::trip_counts::trip_counts_from;
use crate::analysis::affine::unknown_reason::UnknownReason;
use crate::analysis::control_flow::loop_forest;
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::dead_code::unused;
use crate::analysis::execution_counts::{
    CountAccumulator, CtaSize, ExecutionCounts, FlopAccumulator, Participation, Per, Resolution,
    Workload,
};
use crate::analysis::instruction_counts::classify::{ClassifiedInstruction, InstructionCategory};
use crate::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier, collect};
use crate::analysis::instruction_counts::count_constraint::count_constraint;
use crate::analysis::instruction_counts::measurement::{
    ArithKind, Contribution, Direction, MeasureKind, Pipe, Space,
};
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::loop_names::{LoopName, loop_names};
use crate::analysis::memory_footprint::{
    AccessForm, CountRange, SharedRequest, Wavefronts, loop_bytes, warp_footprint, warp_wavefronts,
};
use crate::analysis::thread_participation::{Pred, thread_sets};
use crate::analysis::uniformity_info::{Uniform, UniformityInfo};
use crate::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
use crate::ptx::ir::Operand;
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Register, SharedSize, SpecialVector, Stmt};
use crate::ptx::literal::parse_int;
use crate::ptx::parse::parser::{ParseError, parse};
use crate::report::bindings::BindingSpec;
use crate::report::measured::{
    Comparison, LineIndex, MeasuredBlock, MeasuredTotals, NcuRecord, Prediction, SectorCount,
    Sectors, SharedWaves,
};
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
    let mut kernels = Vec::new();
    let mut classified = Fraction { num: 0, den: 0 };
    let mut trips_resolved = Fraction { num: 0, den: 0 };
    let bindings = each_kernel(source, opts, |builder| {
        let ncu = opts.ncu.as_ref().map(|r| (r, builder.lines));
        let k = builder.build(opts.launch, ncu);
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
    })?;

    let mut coverage = BTreeMap::new();
    coverage.insert("instructions_classified".to_owned(), classified);
    coverage.insert("loop_trips_resolved".to_owned(), trips_resolved);

    Ok(Report {
        input: input_name.to_owned(),
        bindings,
        grid: opts.grid,
        kernels,
        coverage,
    })
}

/// Parse `source` and run `each` on every kernel's report builder, under
/// the options' bindings, block shape and grid; the bindings it applied.
/// The record's kernel against the record: per block, its predicted
/// counts, sectors and wavefronts and the measured ones, and the launch's
/// global sectors and L1TEX's; `None` without a record, a block shape or
/// that kernel.
pub fn compare(source: &str, opts: &AnalyzeOptions) -> Result<Option<Comparison>, AnalyzeError> {
    let mut out = None;
    each_kernel(source, opts, |builder| {
        let Some(record) = &opts.ncu else {
            return;
        };
        let threads: u64 = builder.counts.shape.iter().map(|&n| u64::from(n)).product();
        if builder.module.interner.resolve(builder.kernel.name) != record.kernel || threads == 0 {
            return;
        }
        let cta = CtaSize {
            threads,
            warps: threads.div_ceil(32),
            exact: true,
        };
        let (blocks, launch) = builder.measured(record, builder.lines, &cta);
        out = Some(Comparison { blocks, launch });
    })?;
    Ok(out)
}

pub(super) fn each_kernel(
    source: &str,
    opts: &AnalyzeOptions,
    mut each: impl FnMut(KernelReportBuilder<'_>),
) -> Result<Vec<Binding>, AnalyzeError> {
    let module = parse(source)?;
    if let Some(record) = &opts.ncu {
        record.check(&module, source).map_err(AnalyzeError::Ncu)?;
    }
    let lines = LineIndex::new(source);
    let mut bindings = Vec::new();
    for kernel in &module.kernels {
        let mut bind_map = resolve_bindings(&module, kernel, &opts.bindings, &mut bindings)?;
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
                bind_map.insert(SymbolicConstant::SpecialReg(reg), i64::from(n));
            }
        }
        let cfg = build_cfg(&module, kernel);
        let forest = loop_forest(&cfg);
        let data = KernelAnalyses::new(&module, kernel, &cfg, &forest);
        each(KernelReportBuilder::new(&data, &bind_map, &lines));
    }
    Ok(bindings)
}

fn resolve_bindings(
    module: &Module,
    kernel: &Kernel,
    binds: &[BindingSpec],
    echo: &mut Vec<Binding>,
) -> Result<HashMap<SymbolicConstant, i64>, AnalyzeError> {
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
        map.insert(SymbolicConstant::KernelParam(index), spec.value);
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

impl KernelReportBuilder<'_> {
    /// Every memory operand of every memory instruction, traced to an
    /// affine address at its own position, grouped by the innermost loop
    /// of its block. `cp.async` gives a store row for its shared
    /// destination and a load row for its global source; an atomic gives
    /// one `load+store` row. Parameter loads are not accesses.
    fn accesses_by_scope(
        &self,
        affine: &AffineAnalysis,
    ) -> (
        ByScope<Access>,
        ByScope<AccessForm>,
        PerRequests,
        PerWavefronts,
        Operands,
    ) {
        let (module, kernel, cfg, forest) = (self.module, self.kernel, self.cfg, self.forest);
        let display = &self.display;
        let mut forms: ByScope<AccessForm> = HashMap::new();
        let mut operands = Operands::new();
        let mut per_request = PerRequests::new();
        let mut per_wavefronts = PerWavefronts::new();
        let name = |v: &Var| match v {
            Var::Iteration(l) => format!("k[{}]", self.loop_ref(*l)),
            Var::Unknown(u) => format!(
                "uniform({})",
                Self::value_name(self.module, self.kernel, self.cfg, self.lines, *u)
            ),
            other => other.to_string(),
        };
        let shape = self.counts.shape;
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
                let classified = crate::analysis::instruction_counts::classify::classify_in(
                    module, kernel, instr,
                );
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
                    let set = self.counts.sets.of_access(bid, pos, i);
                    let guard = self.counts.sets.guard(pos, i);
                    let guard_exact = guard.as_ref().is_none_or(|g| {
                        g.count(shape)
                            .map_or(g.exact(), |c| c.certain == c.possible)
                    });
                    let conditional = (self.blocks[bid.0 as usize].qualifier
                        == CountQualifier::AtMost
                        || !set.lane_exact())
                        && set.count(shape).is_none_or(|c| c.certain != c.possible);
                    let threads = guard.is_some().then(|| self.threads_info(&set)).flatten();
                    let Operand::Memory {
                        base, offset, rest, ..
                    } = module.operand(mem_ops[i])
                    else {
                        continue;
                    };
                    let traced = match module.operand(*base) {
                        Operand::Register(_) if rest.is_empty() => {
                            Some(affine.address(*base, *offset, pos))
                        }
                        _ => None,
                    };
                    let (address, unknown, form) = match (module.operand(*base), &traced) {
                        // PTX ISA §9.7.9.26.5.2: `[tensorMap, tensorCoords]`.
                        _ if !rest.is_empty() => (
                            None,
                            Some("the tensor map holds the address".to_owned()),
                            None,
                        ),
                        (Operand::SymbolRef(s), _) => {
                            let sym = self.shorten(module.interner.resolve(*s).to_owned());
                            let addr = if *offset == 0 {
                                sym
                            } else {
                                format!("{sym} + {offset}")
                            };
                            (Some(addr), None, None)
                        }
                        (_, Some(Ok(a))) => {
                            (Some(self.shorten(a.render(name))), None, Some(a.clone()))
                        }
                        (_, Some(Err(reason))) => (
                            None,
                            Some(reason.render(affine.data, scope, false, &name)),
                            None,
                        ),
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
                    let request = bytes.and_then(|b| shared_request(module, instr, b));
                    let wavefronts = match (&form, bytes) {
                        (Some(a), Some(_)) if space == Space::Shared => Some(
                            request
                                .ok_or_else(|| "wavefronts not modelled".to_owned())
                                .and_then(|r| {
                                    warp_wavefronts(a, r, shape, |tid| set.contains(tid, shape))
                                }),
                        ),
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
                    let (wavefronts_per_request, ideal_wavefronts, wavefronts_unknown) =
                        match wavefronts {
                            Some(Ok(w)) if sectors_at_most => {
                                (Some(top(w.count)), Some(top(w.ideal)), None)
                            }
                            Some(Ok(w)) => (Some(w.count), Some(w.ideal), None),
                            Some(Err(why)) => (None, None, Some(why)),
                            None => (None, None, None),
                        };
                    if let (Some(count), Some(ideal), Some(r)) =
                        (wavefronts_per_request, ideal_wavefronts, request)
                    {
                        let per_request = Wavefronts { count, ideal };
                        let (at_most, bits) = (sectors_at_most, r.access_bits());
                        (per_wavefronts).insert(
                            pos,
                            SharedWaves {
                                per_request,
                                at_most,
                                bits,
                            },
                        );
                    }
                    if space == Space::Global
                        && let (Some(b), Some(sectors), Some(ideal)) =
                            (bytes, sectors_per_request, ideal)
                    {
                        let r = PerRequest {
                            bits: b * 8,
                            store: direction == "store",
                            sectors,
                            ideal,
                            at_most: sectors_at_most,
                        };
                        per_request.insert(pos, r);
                    }
                    if !self.counts.unused[pos] {
                        let unknown_varies = (form.as_ref())
                            .is_some_and(|a| affine.alike(a, pos, Uniform::Cta, scope).is_err());
                        forms.entry(scope).or_default().push(AccessForm {
                            form: form.clone(),
                            bytes,
                            align: align.or(bytes).unwrap_or(1),
                            space,
                            set: set.clone(),
                            guard_exact,
                            conditional,
                            unknown_varies,
                        });
                    }
                    let mut reuse = Vec::new();
                    if let Some(a) = &form {
                        let mut cur = scope;
                        while let Some(l) = cur {
                            let stepped = (a.terms.keys()).any(|v| {
                                (*v != Var::Iteration(l) && v.counter() == Some(l))
                                    || v.unknown().is_some_and(|u| affine.data.defined_in(u, l))
                            });
                            reuse.push(Reuse {
                                r#loop: display[l.0 as usize].clone(),
                                stride: match stepped {
                                    true => Some("varies".to_owned()),
                                    false => {
                                        a.terms.get(&Var::Iteration(l)).map(ToString::to_string)
                                    }
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
                    let row = Access {
                        site,
                        opcode: module.opcode(instr),
                        space: space.key().to_owned(),
                        direction: direction.to_owned(),
                        bytes,
                        bytes_at_most: (classified.contributions.iter())
                            .any(|c| c.at_most && c.memory_operand == Some(i)),
                        predicated: guard.is_some(),
                        threads: threads.clone(),
                        address,
                        unknown,
                        sectors_per_request,
                        sectors_at_most,
                        ideal_sectors_per_request: ideal,
                        wavefronts_per_request,
                        ideal_wavefronts_per_request: ideal_wavefronts,
                        footprint_unknown: footprint_unknown.or(wavefronts_unknown),
                        reuse,
                    };
                    let traced = traced.map(|t| t.map(|a| affine.known_multiples(&a)));
                    operands.entry(pos).or_default().push((row.clone(), traced));
                    out.entry(scope).or_default().push(row);
                }
            }
        }
        (out, forms, per_request, per_wavefronts, operands)
    }

    /// `text` with each shared array's mangled name as its source name.
    pub(super) fn shorten(&self, mut text: String) -> String {
        let decls = (self.kernel.shared_decls.iter()).chain(&self.module.shared_decls);
        for raw in decls.map(|d| self.module.interner.resolve(d.name)) {
            let demangled = demangle(raw);
            text = text.replace(raw, demangled.rsplit("::").next().unwrap_or(raw));
        }
        text
    }
}

/// The shared request of a load, a store, or an `.m8n8` `.b16` matrix
/// move, of `bytes` per lane.
fn shared_request(module: &Module, instr: &Instr, bytes: u32) -> Option<SharedRequest> {
    let mods: Vec<&str> = (module.modifiers(instr).iter())
        .map(|&m| module.interner.resolve(m))
        .collect();
    match instr.mnemonic {
        Opcode::Ld => Some(SharedRequest::Load(bytes)),
        Opcode::St => Some(SharedRequest::Store(bytes)),
        Opcode::Ldmatrix | Opcode::Stmatrix if mods.contains(&"m8n8") && mods.contains(&"b16") => {
            (mods.iter())
                .find_map(|m| m.strip_prefix('x')?.parse().ok())
                .map(SharedRequest::Matrix)
        }
        _ => None,
    }
}

/// A global access's width in bits and its sectors and ideal per warp
/// request, bounds when `at_most`.
#[derive(Clone, Copy)]
struct PerRequest {
    bits: u32,
    store: bool,
    sectors: CountRange,
    ideal: CountRange,
    at_most: bool,
}

/// Per statement, its global access's sectors.
type PerRequests = HashMap<usize, PerRequest>;

/// Per statement, its shared access's wavefronts.
type PerWavefronts = HashMap<usize, SharedWaves>;

/// Rows per scope: a block's innermost loop, or `None` outside every loop.
type ByScope<T> = HashMap<Option<LoopId>, Vec<T>>;

/// Per statement, its memory operands' rows, each with its address as
/// traced where a register holds it: the affine form, an unknown value's
/// known multiple as the value, or why there is none.
pub(super) type Operands = HashMap<usize, Vec<(Access, Option<Result<AffineExpr, UnknownReason>>)>>;

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

/// `acc`'s sum as the report's count, each symbol named by `name`.
fn count(acc: &CountAccumulator, name: &dyn Fn(&SymbolicConstant) -> String) -> Count {
    Count {
        expr: acc.expr().render(name),
        at_most: acc.at_most,
        at_least: acc.at_least,
    }
}

/// `acc`'s FLOPs by precision and in total, as the report's counts.
fn flop_counts(
    acc: &FlopAccumulator,
    name: &dyn Fn(&SymbolicConstant) -> String,
) -> BTreeMap<String, Count> {
    let mut out: BTreeMap<String, Count> = (acc.by_precision.iter())
        .map(|(p, v)| (p.key().to_owned(), count(v, name)))
        .collect();
    out.insert("total".to_owned(), count(&acc.total, name));
    out
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
            at_most: c.at_most,
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

    fn warps(&self, name: &dyn Fn(&SymbolicConstant) -> String) -> Option<Count> {
        self.warps.touched.then(|| count(&self.warps, name))
    }
}

#[derive(Default)]
struct InstructionAccumulator {
    total: IssueAccumulator,
    opcodes: BTreeMap<String, IssueAccumulator>,
    variants: BTreeMap<(String, Vec<Contribution>), IssueAccumulator>,
}

/// `(constant coefficient, the rest)` of a product.
pub(super) struct KernelReportBuilder<'a> {
    pub(super) module: &'a Module,
    pub(super) kernel: &'a Kernel,
    pub(super) cfg: &'a ControlFlowGraph,
    pub(super) forest: &'a LoopForest,
    names: Vec<LoopName>,
    pub(super) blocks: Vec<BlockMeasurements>,
    pub(super) bind_map: &'a HashMap<SymbolicConstant, i64>,
    /// Display names with the remainder suffix applied.
    display: Vec<String>,
    /// Memory operands per scope: the innermost loop of their block, or
    /// `None` outside every loop.
    accesses: ByScope<Access>,
    /// The same rows' bound affine forms, where known.
    access_forms: ByScope<AccessForm>,
    per_request: PerRequests,
    per_wavefronts: PerWavefronts,
    pub(super) operands: Operands,
    pub(super) data: &'a KernelAnalyses<'a>,
    pub(super) counts: ExecutionCounts<'a>,
    /// The PTX file's lines, naming a uniform value by its statement's.
    pub(super) lines: &'a LineIndex,
}

impl<'a> KernelReportBuilder<'a> {
    fn new(
        data: &'a KernelAnalyses<'a>,
        bind_map: &'a HashMap<SymbolicConstant, i64>,
        lines: &'a LineIndex,
    ) -> Self {
        let KernelAnalyses {
            module,
            kernel,
            cfg,
            forest,
            ..
        } = *data;
        let uniformity = UniformityInfo::new(data);
        let affine = AffineAnalysis::new(data, &uniformity, Some(bind_map));
        let names = loop_names(module, kernel, cfg, forest);
        let name = |v: &Var| match v {
            Var::Unknown(u) => format!(
                "uniform({})",
                Self::value_name(module, kernel, cfg, lines, *u)
            ),
            other => other.to_string(),
        };
        let trip_info = trip_counts_from(&affine, &names, &name);
        let mut blocks = collect(module, kernel, cfg, forest);

        let mut display: Vec<String> = names.iter().map(|n| n.display.clone()).collect();
        for pair in &trip_info.unroll_pairs {
            let r = pair.remainder.0 as usize;
            display[r] = format!("{} (remainder)", display[r]);
        }

        let mut sets = thread_sets(&affine);
        for (form, _) in sets.forms.values_mut() {
            *form = affine.known_multiples(form);
        }
        let compile_time: HashMap<SymbolicConstant, i64> =
            (SpecialVector::Ntid.components().into_iter())
                .zip(kernel.reqntid.into_iter().flatten())
                .map(|(reg, n)| (SymbolicConstant::SpecialReg(reg), i64::from(n)))
                .collect();
        let static_sets;
        let compile_time_sets = match compile_time == *bind_map {
            true => &sets,
            false => {
                static_sets =
                    thread_sets(&AffineAnalysis::new(data, &uniformity, Some(&compile_time)));
                &static_sets
            }
        };
        let unused = unused(module, kernel, &data.ssa, |s| {
            match compile_time_sets.never(data.block_of(s), s) {
                true => Some(false),
                false => (compile_time_sets.guards.get(&s) == Some(&Pred::True)).then_some(true),
            }
        });
        let counts = ExecutionCounts::new(data, bind_map, trip_info, sets, unused, &mut blocks);
        let mut b = KernelReportBuilder {
            module,
            kernel,
            cfg,
            forest,
            names,
            blocks,
            bind_map,
            display,
            accesses: HashMap::new(),
            access_forms: HashMap::new(),
            per_request: HashMap::new(),
            per_wavefronts: HashMap::new(),
            operands: HashMap::new(),
            data,
            counts,
            lines,
        };
        let (accesses, access_forms, per_request, per_wavefronts, operands) =
            b.accesses_by_scope(&affine);
        b.accesses = accesses;
        b.access_forms = access_forms;
        b.per_request = per_request;
        b.per_wavefronts = per_wavefronts;
        b.operands = operands;
        b
    }

    /// `%r83@106`: a value's register and the PTX line of its statement,
    /// with the statement's index on the line when it shares it;
    /// `%r34@$L__BB0_5` for a φ, its block.
    pub(super) fn value_name(
        module: &Module,
        kernel: &Kernel,
        cfg: &ControlFlowGraph,
        lines: &LineIndex,
        v: Value,
    ) -> String {
        let reg = |r: Register| match r {
            Register::Named(s) => module.interner.resolve(s).to_owned(),
            Register::Cc => "CC.CF".to_owned(),
        };
        let line = |i: usize| match &kernel.stmts[i] {
            Stmt::Instr(instr) => Some(lines.line(instr.offset)),
            Stmt::Unparsed { offset, .. } => Some(lines.line(*offset)),
            _ => None,
        };
        match v {
            Value::Result { op, reg: r } => {
                let Some(at) = line(op) else {
                    return reg(r);
                };
                let on: Vec<usize> = (0..kernel.stmts.len())
                    .filter(|&i| line(i) == Some(at))
                    .collect();
                match on.iter().position(|&i| i == op) {
                    Some(i) if on.len() > 1 => format!("{}@{at}.{}", reg(r), i + 1),
                    _ => format!("{}@{at}", reg(r)),
                }
            }
            Value::Argument { block, reg: r } => {
                format!("{}@{}", reg(r), cfg.block_name(module, block))
            }
            Value::Undef(r) => reg(r),
        }
    }

    pub(super) fn loop_bytes(&self, id: LoopId) -> Option<LoopBytes> {
        let trips = self.counts.trip_exprs[id.0 as usize]
            .as_const()
            .filter(|&t| t > 0)?;
        let forms = self.access_forms.get(&Some(id))?;
        loop_bytes(id, trips, self.counts.shape, forms, |key| self.pointer(key))
    }

    /// The pointer a loop footprint's base sits on: the one 64-bit
    /// parameter it names, with coefficient 1 in its symbolic part, when
    /// it names no other address (a parameter in a coefficient, a
    /// module-scope array) and no 64-bit uniform value. An offset from a
    /// pointer is taken to stay in its allocation.
    fn pointer(&self, key: &AffineExpr) -> Option<usize> {
        let pointer = |leaf: &SymbolicConstant| match leaf {
            SymbolicConstant::KernelParam(i) => {
                (self.kernel.params.get(*i)).is_some_and(|p| p.ty.bytes() == Some(8))
            }
            _ => false,
        };
        let address = |leaf: &SymbolicConstant| {
            pointer(leaf)
                || !matches!(
                    leaf,
                    SymbolicConstant::KernelParam(_) | SymbolicConstant::SpecialReg(_)
                )
        };
        let mut coefficients = Vec::new();
        for (v, c) in &key.terms {
            let wide = v.unknown().is_some_and(|u| {
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
        match p {
            SymbolicConstant::KernelParam(i) if direct && pointer(p) => Some(*i),
            _ => None,
        }
    }

    /// A loop in an expression: its name, and its header's label where
    /// another loop has the same name.
    fn loop_ref(&self, l: LoopId) -> String {
        let name = &self.display[l.0 as usize];
        match self.display.iter().filter(|d| *d == name).count() {
            1 => name.clone(),
            _ => format!("{name} {}", self.names[l.0 as usize].label),
        }
    }

    /// A symbol as the report prints it: a loop's unknown count by the
    /// loop's name, a uniform value by its register and line.
    pub(super) fn leaf_name(&self, leaf: &SymbolicConstant) -> String {
        match leaf {
            SymbolicConstant::TripCount(l) => format!("trips({})", self.loop_ref(*l)),
            SymbolicConstant::Unknown(v) => format!(
                "uniform({})",
                Self::value_name(self.module, self.kernel, self.cfg, self.lines, *v)
            ),
            other => other.to_string(),
        }
    }

    /// The bytes a register's declaration gives it.
    fn register_bytes(&self, reg: Register) -> Option<u32> {
        let Register::Named(reg) = reg else {
            return None;
        };
        self.kernel.register_type(self.module, reg)?.bytes()
    }

    /// The threads a set selects, when a thread-index condition does.
    fn threads_info(&self, set: &Pred) -> Option<ThreadsInfo> {
        let counted = set.count(self.counts.shape);
        let warps = set.warps(self.counts.shape);
        Some(ThreadsInfo {
            condition: set.render_named(&|v| match v {
                Var::Iteration(l) => format!("k[{}]", self.loop_ref(*l)),
                other => other.to_string(),
            })?,
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
                    threads: self.threads_info(&self.counts.sets.blocks[b.0 as usize]),
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

    /// Aggregate over blocks; `below` = aggregate within this loop
    /// (multipliers stop there), `None` = whole kernel. With
    /// `cta`, every per-thread count additionally scales by the CTA's
    /// thread count, and is an upper bound when that count is one. With
    /// `ctas` too, over the launch: a block or guarded instruction at its
    /// threads and warps over the grid where they are counted, else at
    /// its count per CTA times the CTAs; one under a loop whose trip
    /// count is not read is unknown, its trips perhaps differing by CTA.
    /// Under loops whose counts read the CTA index, a block or guarded
    /// instruction whose threads do not is counted per CTA times those
    /// counts summed over the grid, and is unknown otherwise.
    fn aggregates(&self, below: Option<LoopId>, per: Per) -> Aggregates {
        let Workload {
            flops,
            bytes,
            conversions,
        } = self.counts.workload(&self.blocks, below, per);
        let mut instructions: BTreeMap<String, InstructionAccumulator> = BTreeMap::new();
        let mut instruction_total = IssueAccumulator::default();
        let mut instruction_all = IssueAccumulator::default();

        for bm in &self.blocks {
            let Some((_, block)) = self.counts.scaled(bm, below, per) else {
                continue;
            };
            let mult = (block.as_ref()).and_then(|p| p.trips.clone());
            let mult = mult.as_ref();
            let ((threads, warps), block_at_most) = (block.as_ref())
                .and_then(|p| Some((p.narrow()?, p.at_most)))
                .unwrap_or(((0, Some((0, false))), false));
            for instruction in &bm.instructions {
                instruction_all.add(threads, warps, mult, block_at_most);
                if !self.counts.counted(bm.block, instruction.provenance) {
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
        }

        let all_flops = flops
            .values()
            .map(|t| t.total.expr())
            .fold(SymExpr::Const(0), SymExpr::add);
        let name = |l: &SymbolicConstant| self.leaf_name(l);
        let bytes_out: BTreeMap<String, DirectionCounts> = bytes
            .iter()
            .map(|(k, (l, s))| {
                (
                    k.to_string(),
                    DirectionCounts {
                        load: count(l, &name),
                        store: count(s, &name),
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
            flops: flop_counts(&flops[&Pipe::CudaCore], &name),
            tensor_flops: flop_counts(&flops[&Pipe::Tensor], &name),
            sfu_flops: flop_counts(&flops[&Pipe::Sfu], &name),
            atomic_flops: flop_counts(&flops[&Pipe::Atomic], &name),
            bytes: bytes_out,
            conversions: count(&conversions, &name),
            ai_global,
            unrolled_source_lines: self.unrolled_lines(below),
            instructions: InstructionCounts {
                all: Some(count(&instruction_all.threads, &name))
                    .filter(|all| *all != count(&instruction_total.threads, &name)),
                all_warps: (count(&instruction_all.threads, &name)
                    != count(&instruction_total.threads, &name))
                .then(|| instruction_all.warps(&name))
                .flatten(),
                total: count(&instruction_total.threads, &name),
                warps: instruction_total.warps(&name),
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
                                    issued: count(&issued.threads, &name),
                                    warps: issued.warps(&name),
                                    contributions_per_execution: contributions
                                        .iter()
                                        .map(contribution_details)
                                        .collect(),
                                });
                        }
                        (
                            k.clone(),
                            KindCounts {
                                total: count(&acc.total.threads, &name),
                                warps: acc.total.warps(&name),
                                opcodes: acc
                                    .opcodes
                                    .iter()
                                    .map(|(o, v)| (o.clone(), count(&v.threads, &name)))
                                    .collect(),
                                opcode_warps: acc
                                    .opcodes
                                    .iter()
                                    .filter_map(|(o, v)| v.warps(&name).map(|w| (o.clone(), w)))
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
        let trips = match &self.counts.trip_info.trips[id.0 as usize] {
            Ok(e) => {
                let bound = e.bind(self.bind_map);
                Trips {
                    expr: Some(bound.render(&|l| self.leaf_name(l))),
                    unknown: None,
                }
            }
            Err(reason) => Trips {
                expr: None,
                unknown: Some(reason.clone()),
            },
        };
        let unroll = self
            .counts
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
            per_iteration: self.aggregates(Some(id), Per::Thread),
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
                        .filter(|i| self.counts.counted(b, i.provenance))
                        .count() as i64;
                    let mut mult = SymExpr::Const(1);
                    for &cl in &self.counts.chain(b) {
                        mult = SymExpr::mul(mult, self.counts.trip_exprs[cl.0 as usize].clone());
                    }
                    weight.add(instrs, Some(&mult), false);
                }
                let weight = weight.expr();
                let approx: HashMap<SymbolicConstant, i64> = weight
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
                        instructions: w.render(&|l| self.leaf_name(l)),
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
            for m in
                (bm.measurements.iter()).filter(|m| self.counts.counted(bm.block, m.provenance))
            {
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
            .filter(|b| self.counts.sets.blocks[b.block.0 as usize] != Pred::False)
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
        for (i, t) in self.counts.trip_info.trips.iter().enumerate() {
            let header = self.forest.loops[i].header;
            if let Err(reason) = t
                && self.counts.sets.blocks[header.0 as usize] != Pred::False
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
    /// `.shared` declarations of `element_count × element_width`, a
    /// scalar's count being 1, plus whether it declares a launch-sized
    /// `.extern .shared` array. See [`SharedMemory`].
    pub(super) fn shared_memory(&self) -> SharedMemory {
        let mut static_bytes = 0u64;
        let mut dynamic = false;
        for decl in self
            .kernel
            .shared_decls
            .iter()
            .chain(&self.module.shared_decls)
        {
            let bytes = decl.ty.bytes().map_or(0, u64::from);
            match decl.size {
                SharedSize::Scalar => static_bytes += bytes,
                SharedSize::Array(count) => static_bytes += count * bytes,
                SharedSize::Dynamic => dynamic = true,
            }
        }
        SharedMemory {
            static_bytes,
            dynamic,
        }
    }

    fn measured(
        &self,
        record: &NcuRecord,
        lines: &LineIndex,
        cta: &CtaSize,
    ) -> (Vec<MeasuredBlock>, Option<MeasuredTotals>) {
        let grid = (record.grid.iter()).try_fold(1u64, |n, &d| n.checked_mul(d.into()));
        let mut per_line: HashMap<u32, usize> = HashMap::new();
        for stmt in &self.kernel.stmts {
            if let Stmt::Instr(i) = stmt {
                *per_line.entry(lines.line(i.offset)).or_default() += 1;
            }
        }
        let reducible = self.forest.irreducible_edges.is_empty();
        let (mut blocks, mut all) = (Vec::new(), Vec::new());
        for bm in &self.blocks {
            let Some(m) = self
                .counts
                .multiplier(bm.block, None, Per::Launch(cta, grid))
            else {
                continue;
            };
            let set = &self.counts.sets.blocks[bm.block.0 as usize];
            let block = self.counts.running(
                bm,
                &m,
                set,
                self.counts.launches.get(&(bm.block, None)).copied(),
            );
            let mut guarded = HashMap::new();
            let mut predictions = Vec::new();
            for i in &bm.instructions {
                let Stmt::Instr(instr) = &self.kernel.stmts[i.provenance] else {
                    continue;
                };
                let line = lines.line(instr.offset);
                let alone = reducible && per_line[&line] == 1;
                let never = !self.counts.counted(bm.block, i.provenance);
                let mut resolve = |operand| {
                    self.counts.statement(
                        bm,
                        &m,
                        i.provenance,
                        operand,
                        block.as_ref(),
                        &mut guarded,
                    )
                };
                let run = match resolve(0) {
                    Resolution::Counted(p) if !never => Some(p),
                    _ => None,
                };
                let trips = |p: &Participation| p.trips.as_ref()?.as_const().filter(|_| alone);
                let at_most = run.as_ref().is_some_and(|p| p.at_most);
                // Warps with a lane that reads its global operand: each
                // requests its sectors; a source no thread reads, none.
                let reads = match self.counts.sets.sources.contains_key(&i.provenance) {
                    true => match resolve(1) {
                        Resolution::Skipped => Some(None),
                        Resolution::Unknown => None,
                        Resolution::Counted(p) => Some(Some(p)),
                    },
                    false => run.clone().map(Some),
                };
                let requests = match (never, reads) {
                    (true, _) | (false, Some(None)) => alone.then_some((0, false)),
                    (false, Some(Some(p))) => p.warps.and_then(|(w, bound)| {
                        Some((
                            u64::try_from(trips(&p)?).ok()?.checked_mul(w)?,
                            bound || p.at_most,
                        ))
                    }),
                    (false, None) => None,
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
                            store: r.store,
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
                        false => run
                            .as_ref()
                            .and_then(|p| u64::try_from(trips(p)?).ok()?.checked_mul(p.threads)),
                    },
                    at_most,
                    sectors,
                    wavefronts: self.per_wavefronts.get(&i.provenance).copied(),
                    atomic: matches!(instr.mnemonic, Opcode::Atom | Opcode::Red),
                });
            }
            let name = self.cfg.block_name(self.module, bm.block);
            blocks.push(record.compare(&name, &predictions));
            all.extend(predictions);
        }
        (blocks, record.compare_totals(&all))
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
        let cta = launch.as_ref().map(|l| CtaSize {
            threads: l.threads,
            warps: l.warps,
            exact: l.exact,
        });
        let totals_per_cta = cta.as_ref().map(|l| self.aggregates(None, Per::Cta(l)));
        let ctas = self
            .counts
            .grid()
            .and_then(|g| g.iter().try_fold(1u64, |n, &d| n.checked_mul(d.into())));
        let totals_per_launch =
            (cta.as_ref().zip(ctas)).map(|(l, n)| self.aggregates(None, Per::Launch(l, Some(n))));

        let unknowns = self.unknowns();
        let (measured, measured_totals) = (ncu.filter(|(r, _)| r.kernel == name).zip(cta.as_ref()))
            .map(|((record, lines), cta)| self.measured(record, lines, cta))
            .unzip();

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
            totals: self.aggregates(None, Per::Thread),
            unknowns,
            unused: (self.kernel.stmts.iter().enumerate())
                .filter_map(|(s, stmt)| match stmt {
                    Stmt::Instr(i) if self.counts.unused[s] => Some(UnusedInstruction {
                        line: self.lines.line(i.offset),
                        opcode: self.module.opcode(i),
                    }),
                    _ => None,
                })
                .collect(),
            measured,
            measured_totals: measured_totals.flatten(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prediction past u64 is none: 32 threads in each of 2 CTAs run
    /// 2^62 + %ctaid.x trips, 2^63 + 1 over the grid.
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
        };
        let opts = AnalyzeOptions {
            grid: Some([2, 1, 1]),
            ncu: Some(record),
            ..Default::default()
        };
        let r = analyze(src, "t", &opts).expect("analyzes");
        assert_eq!(
            r.kernels[0].loops[0].trips.expr.as_deref(),
            Some("%ctaid.x + 4611686018427387904")
        );
        assert!(r.kernels[0].measured.is_some());
    }

    /// A per-trip launch count past u64, 32 threads of 3 CTAs on
    /// 2^58 - 1 of 2^58 trips, has no prediction.
    #[test]
    fn a_per_trip_launch_count_past_u64_has_no_prediction() {
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k(\n.param .u64 k_param_0\n)\n.reqntid 32, 1, 1\n{\n\
                   ld.param.u64 %rd1, [k_param_0];\nmov.u64 %rd2, 0;\n$L__H:\n\
                   add.s64 %rd2, %rd2, 1;\nsetp.ge.u64 %p1, %rd2, 288230376151711744;\n\
                   @%p1 bra $L__T;\nbar.sync 0;\n$L__T:\n\
                   setp.lt.u64 %p2, %rd2, 288230376151711744;\n@%p2 bra $L__H;\nret;\n}\n";
        let line = |ptx: &str| crate::report::measured::NcuLine {
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
        };
        let opts = AnalyzeOptions {
            grid: Some([3, 1, 1]),
            ncu: Some(record),
            ..Default::default()
        };
        let r = analyze(src, "t", &opts).expect("analyzes");
        let measured = r.kernels[0].measured.as_ref().expect("a record");
        let predicted: Vec<u64> = (measured.iter())
            .flat_map(|b| b.differing.iter().map(|l| l.predicted))
            .collect();
        assert_eq!(predicted, Vec::<u64>::new());
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
        // A scalar reserves its type's size.
        assert_eq!(sm(".shared .align 8 .b64 bar;\n"), (8, false));
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
        let line = crate::report::measured::NcuLine {
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
        };
        let opts = AnalyzeOptions {
            bindings: record.bindings(),
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
        };
        let report = analyze(src, "k.ptx", &opts).expect("analyzes");
        let blocks = report.kernels[0].measured.as_ref().expect("measured");
        let inner = blocks
            .iter()
            .find(|b| b.name == "$L__I")
            .expect("inner loop");
        assert_eq!((inner.with_sass, inner.agreeing), (1, 1));
    }
}
