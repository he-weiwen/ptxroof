//! Each kernel's analyses under the caller's bindings and launch, the
//! memory operands' rows the annotated PTX prints, and a kernel's
//! predictions against its ncu record.

use crate::analysis::affine::affine_analysis::AffineAnalysis;
use crate::analysis::affine::affine_expr::{AffineExpr, Var};
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::trip_counts::trip_counts_from;
use crate::analysis::affine::unknown_reason::UnknownReason;
use crate::analysis::control_flow::loop_forest;
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::dead_code::unused;
use crate::analysis::execution_counts::{CtaSize, ExecutionCounts, Participation, Per, Resolution};
use crate::analysis::instruction_counts::collect::{BlockMeasurements, CountQualifier, collect};
use crate::analysis::instruction_counts::measurement::{Direction, MeasureKind, Pipe};
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::loop_names::{LoopName, loop_names};
use crate::analysis::memory_footprint::{
    AccessForm, CountRange, LoopBytes, SharedRequest, Wavefronts, loop_bytes, warp_footprint,
    warp_wavefronts,
};
use crate::analysis::thread_participation::{Pred, thread_sets};
use crate::analysis::uniformity_info::{Uniform, UniformityInfo};
use crate::ptx::cfg::{ControlFlowGraph, build_cfg};
use crate::ptx::count_constraint::count_constraint;
use crate::ptx::ir::Operand;
use crate::ptx::ir::{Instr, Kernel, Module, Opcode, Register, SharedSize, SpecialVector, Stmt};
use crate::ptx::literal::parse_int;
use crate::ptx::modifier::Space;
use crate::ptx::parse::parser::{ParseError, parse};
use crate::ptx::thread_hierarchy::{WARP_SZ, ctas, threads};
use crate::report::bindings::BindingSpec;
use crate::report::measured::{
    Comparison, LineIndex, MeasuredBlock, MeasuredTotals, NcuRecord, Prediction, SectorCount,
    Sectors, SharedWaves,
};
use crate::report::names::demangle;

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

/// Caller-supplied analysis inputs: the launch config and bindings are
/// never guessed; the alignment of the memory pointer parameters point
/// to defaults to `PTR_ALIGN`, and the `.entry` line says so.
#[derive(Debug)]
pub struct AnalyzeOptions {
    pub bindings: Vec<BindingSpec>,
    /// `--launch x,y,z`; `None` = default to `.reqntid`/`.maxntid`
    /// when the kernel carries one.
    pub launch: Option<[u32; 3]>,
    /// `--grid x,y,z`, the launch's CTAs per axis.
    pub grid: Option<[u32; 3]>,
    pub ncu: Option<NcuRecord>,
    /// `--ptr-align N`: the alignment, in bytes, taken for the memory a
    /// pointer parameter points to where its `.ptr .align` declares less.
    pub ptr_align: u32,
}

/// The default `ptr_align`: a 32-byte sector's, which the 256 B
/// alignment of `cudaMalloc`'s memory (CUDA C++ Programming Guide,
/// "Device Memory Accesses") covers but a view into it may not.
pub const PTR_ALIGN: u32 = 32;

impl Default for AnalyzeOptions {
    fn default() -> Self {
        AnalyzeOptions {
            bindings: Vec::new(),
            launch: None,
            grid: None,
            ncu: None,
            ptr_align: PTR_ALIGN,
        }
    }
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
        let threads = threads(builder.counts.shape) as u64;
        if builder.module.interner.resolve(builder.kernel.name) != record.kernel || threads == 0 {
            return;
        }
        let cta = CtaSize {
            threads,
            warps: threads.div_ceil(WARP_SZ as u64),
            exact: true,
        };
        let (blocks, launch) = builder.measured(record, builder.lines, &cta);
        let per = Per::Launch(&cta, ctas(record.grid));
        let workload = builder.counts.workload(&builder.blocks, None, per);
        let tensor = &workload.flops[&Pipe::Tensor].total;
        let tensor_flops = (tensor.expr().as_const())
            .filter(|_| !tensor.at_least)
            .and_then(|n| record.compare_tensor_flops(u64::try_from(n).ok()?, tensor.at_most));
        out = Some(Comparison {
            blocks,
            launch,
            tensor_flops,
        });
    })?;
    Ok(out)
}

/// One memory operand of one instruction: where it points, as an
/// affine form over thread and CTA indices (`%tid.x`), loop counters
/// (`k[loop name]`, the iteration number from 1) and the parameters,
/// plus the instruction's constant offset; or why that is unknown.
#[derive(Clone)]
pub(super) struct Access {
    pub(super) space: String,
    /// `load`, `store`, or `load+store` for an atomic on one location.
    pub(super) direction: String,
    pub(super) address: Option<String>,
    pub(super) unknown: Option<String>,
    /// The 32-byte sectors one warp's request touches, as a range
    /// over the alignment of the address's uniform part; global and
    /// generic spaces with a known address only. When the executing
    /// threads are a bound, so is this: the most any warp touches.
    pub(super) sectors_per_request: Option<crate::analysis::memory_footprint::CountRange>,
    /// `sectors_per_request` and `ideal_sectors_per_request`, or the
    /// wavefronts, are each the most any warp takes.
    pub(super) sectors_at_most: bool,
    /// The fewest sectors the distinct bytes the executing lanes touch
    /// could occupy if packed: the request's coalesced ideal.
    pub(super) ideal_sectors_per_request: Option<crate::analysis::memory_footprint::CountRange>,
    /// The shared memory wavefronts one warp's request takes, as a range
    /// over the warps and the alignment of the address's uniform part;
    /// shared space with a known address only, a bound like the sectors.
    pub(super) wavefronts_per_request: Option<crate::analysis::memory_footprint::CountRange>,
    /// The wavefronts if each group of lanes had its words in the fewest;
    /// fewer than `wavefronts_per_request` is a bank conflict.
    pub(super) ideal_wavefronts_per_request: Option<crate::analysis::memory_footprint::CountRange>,
    pub(super) footprint_unknown: Option<String>,
}

pub(super) struct SharedMemory {
    /// Statically-declared shared memory per CTA, in bytes.
    pub(super) static_bytes: u64,
    /// An `.extern .shared` array is present; its size is set at launch.
    pub(super) dynamic: bool,
}

pub(super) fn each_kernel(
    source: &str,
    opts: &AnalyzeOptions,
    mut each: impl FnMut(KernelReportBuilder<'_>),
) -> Result<(), AnalyzeError> {
    let module = parse(source)?;
    if let Some(record) = &opts.ncu {
        record.check(&module, source).map_err(AnalyzeError::Ncu)?;
    }
    let lines = LineIndex::new(source);
    for kernel in &module.kernels {
        let mut bind_map = resolve_bindings(&module, kernel, &opts.bindings)?;
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
        each(KernelReportBuilder::new(
            &data,
            &bind_map,
            &lines,
            opts.ptr_align,
        ));
    }
    Ok(())
}

fn resolve_bindings(
    module: &Module,
    kernel: &Kernel,
    binds: &[BindingSpec],
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
        let mut forms: ByScope<AccessForm> = HashMap::new();
        let mut operands = Operands::new();
        let mut per_request = PerRequests::new();
        let mut per_wavefronts = PerWavefronts::new();
        let name = |v: &Var| Self::var_text(self.data, self.lines, v);
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
                // Pair memory contributions by their explicit operand association.
                // An atomic has one address with read+write; a copy has two addresses.
                let mut accesses: BTreeMap<(usize, Space, Option<u32>), (bool, bool)> =
                    BTreeMap::new();
                let measured = &self.blocks[bid.0 as usize].measurements;
                let first = measured.partition_point(|m| m.provenance < pos);
                for c in measured[first..].iter().take_while(|m| m.provenance == pos) {
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
                    let Operand::Memory {
                        base, offset, rest, ..
                    } = module.operand(mem_ops[i])
                    else {
                        continue;
                    };
                    let traced = match module.operand(*base) {
                        Operand::Register(_) | Operand::SymbolRef(_) if rest.is_empty() => {
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
                            let pointer = self.pointee(a);
                            Some(warp_footprint(
                                a,
                                b,
                                align.unwrap_or(b),
                                pointer,
                                shape,
                                |tid| set.contains(tid, shape),
                            ))
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
                    let row = Access {
                        space: space.key().to_owned(),
                        direction: direction.to_owned(),
                        address,
                        unknown,
                        sectors_per_request,
                        sectors_at_most,
                        ideal_sectors_per_request: ideal,
                        wavefronts_per_request,
                        ideal_wavefronts_per_request: ideal_wavefronts,
                        footprint_unknown: footprint_unknown.or(wavefronts_unknown),
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
    /// The alignment taken for the memory a pointer parameter points to
    /// past its `.ptr .align`.
    pub(super) ptr_align: u32,
}

impl<'a> KernelReportBuilder<'a> {
    fn new(
        data: &'a KernelAnalyses<'a>,
        bind_map: &'a HashMap<SymbolicConstant, i64>,
        lines: &'a LineIndex,
        ptr_align: u32,
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
        let name = |v: &Var| Self::var_text(data, lines, v);
        let trip_info = trip_counts_from(&affine, &name);
        let mut blocks = collect(&affine);

        let display: Vec<String> = names.iter().map(|n| n.display.clone()).collect();

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
            ptr_align,
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

    /// Loop `l`'s iteration from 0, `iv[<header label>]`, LLVM's canonical
    /// induction variable (llvm/include/llvm/Analysis/LoopInfo.h).
    pub(super) fn iv(data: &KernelAnalyses, l: LoopId) -> String {
        let header = data.forest.get(l).header;
        format!("iv[{}]", data.cfg.block_name(data.module, header))
    }

    /// A variable as the report writes it: a loop's iteration k as
    /// `(iv[<header label>] + 1)`, an unknown value by its register and
    /// line, `uniform(%r83@106)`.
    pub(super) fn var_text(data: &KernelAnalyses, lines: &LineIndex, v: &Var) -> String {
        match v {
            Var::Iteration(l) => format!("({} + 1)", Self::iv(data, *l)),
            Var::Unknown(u) => {
                let name = Self::value_name(data.module, data.kernel, data.cfg, lines, *u);
                format!("uniform({name})")
            }
            other => other.to_string(),
        }
    }

    pub(super) fn loop_bytes(&self, id: LoopId) -> Option<LoopBytes> {
        let trips = self.counts.trip_exprs[id.0 as usize]
            .as_const()
            .filter(|&t| t > 0)?;
        let forms = self.access_forms.get(&Some(id))?;
        loop_bytes(id, trips, self.counts.shape, forms, |key| self.pointee(key))
    }

    /// The parameter `pointer` finds an address sits on, and the
    /// alignment of the memory it points to: its `.ptr .align`, or
    /// `ptr_align` where that is more.
    fn pointee(&self, a: &AffineExpr) -> Option<(usize, u32)> {
        let i = self.pointer(a)?;
        let declared = self.kernel.params[i].ptr_align.unwrap_or(1);
        Some((i, declared.max(self.ptr_align)))
    }

    /// The pointer an address or a loop footprint's base sits on: the one 64-bit
    /// parameter it names, with coefficient 1 in its symbolic part, when
    /// it names no other address (a parameter in a coefficient, a
    /// module-scope array) and no 64-bit uniform value. An offset from a
    /// pointer is taken to stay in its allocation.
    pub(super) fn pointer(&self, key: &AffineExpr) -> Option<usize> {
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
        let grid = ctas(record.grid);
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
            for &s in &bm.instructions {
                let Stmt::Instr(instr) = &self.kernel.stmts[s] else {
                    continue;
                };
                let line = lines.line(instr.offset);
                let alone = reducible && per_line[&line] == 1;
                let never = !self.counts.counted(bm.block, s);
                let mut resolve = |operand| {
                    self.counts
                        .statement(bm, &m, s, operand, block.as_ref(), &mut guarded)
                };
                let run = match resolve(0) {
                    Resolution::Counted(p) if !never => Some(p),
                    _ => None,
                };
                let trips = |p: &Participation| p.trips.as_ref()?.as_const().filter(|_| alone);
                let at_most = run.as_ref().is_some_and(|p| p.at_most);
                // Warps with a lane that reads its global operand: each
                // requests its sectors; a source no thread reads, none.
                let reads = match self.counts.sets.sources.contains_key(&s) {
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
                let sectors = (self.per_request.get(&s).copied()).zip(requests).and_then(
                    |(r, (n, bound))| {
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
                    },
                );
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
                    wavefronts: self.per_wavefronts.get(&s).copied(),
                    atomic: matches!(instr.mnemonic, Opcode::Atom | Opcode::Red),
                });
            }
            let name = self.cfg.block_name(self.module, bm.block);
            blocks.push(record.compare(&name, &predictions));
            all.extend(predictions);
        }
        (blocks, record.compare_totals(&all))
    }
}
