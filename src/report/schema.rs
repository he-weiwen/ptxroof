//! The owned result tree returned by `report::analyze`. Lower-level IR
//! and analysis types are also public. JSON output is the
//! `Serialize` derivation of these structs; the text report renders
//! the same values, so the two views cannot drift.
//!
//! Schema conventions, pinned by the committed scenario expectations:
//! - every count is `{"expr": string, "at_most": bool, "at_least":
//!   bool}` — symbolic expressions print via SymExpr's deterministic
//!   form, `at_most` marks upper bounds (rendered `<=` in text),
//!   `at_least` a count that unclassified instructions or
//!   unquantified bytes in its scope could raise (`+ unknown`);
//! - trips are `{"expr": ...}` or `{"unknown": reason}` — an unknown
//!   is a result, not an error;
//! - the three flop tables (one per pipe) always carry every
//!   precision plus "total", so "0 f16 flops" is assertable (S8);
//! - byte tables always carry global/shared/local; other spaces appear
//!   when touched;
//! - `coverage` is `{metric: {num, den}}` count pairs — the runner
//!   aggregates them corpus-wide (percentages cannot be aggregated).

use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct Report {
    pub input: String,
    /// `--bind` values, echoed (bet 4: inputs are visible).
    pub bindings: Vec<Binding>,
    pub kernels: Vec<KernelReport>,
    pub coverage: BTreeMap<String, Fraction>,
}

#[derive(Debug, Serialize)]
pub struct Binding {
    pub param: usize,
    pub name: String,
    pub value: i64,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
pub struct Fraction {
    pub num: u64,
    pub den: u64,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct Count {
    pub expr: String,
    pub at_most: bool,
    /// Unclassified instructions or unquantified bytes in the scope
    /// could raise it.
    pub at_least: bool,
}

#[derive(Debug, Serialize)]
pub struct KernelReport {
    pub name: String,
    pub demangled: String,
    pub params: Vec<ParamInfo>,
    /// The kernel's basic blocks in program order: the names every
    /// loop below is referred to by, with where each block came from
    /// and where control goes next.
    pub blocks: Vec<BlockInfo>,
    /// Shared memory reserved per CTA. `static_bytes` is the sum of the
    /// kernel's `.shared` array declarations — a `[static]` demand
    /// figure that matches ptxas's `bytes smem` and Nsight Compute's
    /// `launch__shared_mem_per_block_static`; driver-reserved shared
    /// memory (NCU's `_driver`) is not included. `dynamic` is set when
    /// the kernel also declares an `.extern .shared` array whose size is
    /// fixed at launch and so is not statically knowable.
    pub shared_memory: SharedMemory,
    /// Instruction-class tallies; the verifier's accounting identity
    /// (`flop + non_flop_arith + memory + sync + communication +
    /// control + ignore + unknown == total`) runs on these.
    pub instruction_classes: InstructionClasses,
    /// The loop that executes the most instructions per kernel
    /// invocation, by the static count (instructions × iterations).
    pub most_instructions_loop: Option<String>,
    /// Launch configuration, when known (flag or PTX directive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub launch: Option<LaunchInfo>,
    /// Memory operands in blocks outside every loop.
    pub accesses: Vec<Access>,
    /// Kernel totals scaled to one CTA (needs `launch`; upper bounds
    /// when the block size is only a maximum).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub totals_per_cta: Option<Aggregates>,
    /// Loops ranked by instructions executed per invocation (a symbolic
    /// expression), most first.
    pub ranking: Vec<RankEntry>,
    /// Top-level loop nodes, in program order.
    pub loops: Vec<LoopNode>,
    pub totals: Aggregates,
    /// Every named hole in the analysis: unclassified instructions,
    /// unquantifiable bytes, unresolved trips, irreducible regions,
    /// call sites. Never silently empty when something was dropped.
    pub unknowns: Vec<UnknownEntry>,
}

#[derive(Debug, Serialize)]
pub struct ParamInfo {
    pub index: usize,
    #[serde(rename = "type")]
    pub ty: String,
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct BlockInfo {
    /// The block's PTX label, or `<block N>` when it has none.
    pub name: String,
    /// Source lines the block's instructions carry, as
    /// `file:first-last` in the block's first file; absent without
    /// line info.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines: Option<String>,
    pub instructions: u64,
    /// The threads of the CTA that execute the block, when a branch on
    /// the thread index selects them. Absent when every thread runs the
    /// block or no selecting branch is a thread-index condition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<ThreadsInfo>,
    /// Successor block names; empty for a block that ends the kernel.
    pub successors: Vec<String>,
    /// The innermost loop containing the block, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#loop: Option<BlockLoop>,
}

/// The threads that execute a block or a guarded instruction, when a
/// thread-index condition selects them.
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct ThreadsInfo {
    /// `⌊%tid.x/32⌋ < 4 and elected`.
    pub condition: String,
    /// How many threads of the block satisfy it; absent without a
    /// block shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    /// Some selector is not a thread-index condition: the count and
    /// the condition are an upper bound.
    pub at_most: bool,
    /// How many warps of the block have a thread in the set; absent
    /// without a block shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warps: Option<u64>,
    /// The warp count is an upper bound.
    pub warps_at_most: bool,
}

impl ThreadsInfo {
    /// `128 in 4 warps (⌊%tid.x/32⌋ < 4)`, `<= (%tid.x == 0)` without
    /// a shape.
    pub fn render(&self) -> String {
        let bound = if self.at_most { "<= " } else { "" };
        match self.count {
            Some(n) => format!("{bound}{n} in {} ({})", self.warps_phrase(), self.condition),
            None => format!("{bound}({})", self.condition),
        }
    }

    /// `4 warps`, `<= 4 warps`, `1 warp`.
    pub fn warps_phrase(&self) -> String {
        let bound = if self.warps_at_most { "<= " } else { "" };
        match self.warps {
            Some(1) if !self.warps_at_most => "1 warp".to_owned(),
            Some(w) => format!("{bound}{w} warps"),
            None => "warps".to_owned(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct BlockLoop {
    pub name: String,
    pub header: bool,
    pub latch: bool,
}

#[derive(Debug, Serialize, Default, Clone, Copy)]
pub struct InstructionClasses {
    pub total: u64,
    pub flop: u64,
    pub non_flop_arith: u64,
    pub memory: u64,
    pub sync: u64,
    pub communication: u64,
    pub control: u64,
    pub ignore: u64,
    pub unknown: u64,
    /// Statements the parser could not read. Not part of `total` (they
    /// are not instructions), so outside the accounting identity; each
    /// is also an entry in `unknowns`.
    pub unparsed: u64,
}

#[derive(Debug, Serialize)]
pub struct SharedMemory {
    /// Statically-declared shared memory per CTA, in bytes.
    pub static_bytes: u64,
    /// An `.extern .shared` array is present; its size is set at launch.
    pub dynamic: bool,
}

#[derive(Debug, Serialize)]
pub struct LaunchInfo {
    pub block: [u32; 3],
    pub threads: u64,
    pub warps: u64,
    /// "flag", ".reqntid", or ".maxntid".
    pub source: String,
    /// `.maxntid` is a maximum, not the launch: `false` there, and
    /// every per-CTA total is then an upper bound.
    pub exact: bool,
}

#[derive(Debug, Serialize)]
pub struct RankEntry {
    #[serde(rename = "loop")]
    pub loop_name: String,
    /// Instructions executed per invocation: instructions × iterations.
    pub instructions: String,
}

#[derive(Debug, Serialize)]
pub struct LoopNode {
    pub name: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    pub depth: u32,
    pub trips: Trips,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unroll: Option<Unroll>,
    pub per_iteration: Aggregates,
    /// The memory operands in this loop's own blocks, not its nested
    /// loops', in program order.
    pub accesses: Vec<Access>,
    /// Global bytes one CTA requests over one execution of the loop's
    /// own blocks, and the distinct bytes those requests touch, when
    /// the trip count, the block shape and every global address are
    /// known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_bytes_per_cta: Option<LoopBytes>,
    pub loops: Vec<LoopNode>,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct LoopBytes {
    pub requested: u64,
    pub unique: u64,
    /// Some contributing access is predicated: both are upper bounds.
    pub at_most: bool,
}

/// One memory operand of one instruction: where it points, as an
/// affine form over thread and CTA indices (`%tid.x`), loop counters
/// (`k[loop name]`, the iteration number from 1) and the parameters,
/// plus the instruction's constant offset; or why that is unknown.
#[derive(Debug, Serialize, Clone)]
pub struct Access {
    /// `file:line`, or the block's label when there is no line.
    pub site: String,
    pub opcode: String,
    pub space: String,
    /// `load`, `store`, or `load+store` for an atomic on one location.
    pub direction: String,
    /// Bytes per thread per execution, when the instruction states them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u32>,
    /// The instruction is guarded.
    pub predicated: bool,
    /// The threads the guard selects, when it is a thread-index
    /// condition; the bytes and footprint count those threads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<ThreadsInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unknown: Option<String>,
    /// The 32-byte sectors and 128-byte lines one warp's request
    /// touches, as a range over the alignment of the address's uniform
    /// part; global and generic spaces with a known address only. An
    /// upper bound when the instruction is predicated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sectors_per_request: Option<crate::analysis::memory_footprint::CountRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines_per_request: Option<crate::analysis::memory_footprint::CountRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub footprint_unknown: Option<String>,
    /// How the address moves per iteration of each enclosing loop,
    /// innermost first; a loop it does not move in is self-reuse.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reuse: Vec<Reuse>,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct Reuse {
    pub r#loop: String,
    /// Bytes the address advances per iteration; absent when it does not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stride: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Trips {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unknown: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Unroll {
    pub factor: i64,
    pub remainder: String,
}

#[derive(Debug, Serialize)]
pub struct Aggregates {
    /// CUDA-core flops. Keys: "total" and every precision key
    /// ("fp8", "f16", "bf16", "tf32", "f32", "f64") — always present.
    pub flops: BTreeMap<String, Count>,
    /// Tensor-core flops (`wmma.mma`, `mma`), same keys.
    pub tensor_flops: BTreeMap<String, Count>,
    /// Special-function-unit flops (`ex2`, `rsqrt`, ...), same keys.
    pub sfu_flops: BTreeMap<String, Count>,
    /// Atomic/reduction FP work, separate from CUDA-core throughput.
    pub atomic_flops: BTreeMap<String, Count>,
    /// Keys: space names; global/shared/local always present.
    pub bytes: BTreeMap<String, DirectionCounts>,
    pub conversions: Count,
    /// Flops of all accounting pipes per global byte, when both are
    /// constants, bytes > 0, and at most one side is an upper bound
    /// (a bound over a bound bounds nothing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_global: Option<Intensity>,
    /// Straight-line repeated source lines (fully-unrolled loops):
    /// "file:line" → workload-op copies. Empty = omitted.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub unrolled_source_lines: BTreeMap<String, u64>,
    /// Instructions issued, in total and by kind. PTX counts, not
    /// SASS: register moves are mostly removed by ptxas and one
    /// `.rn` divide becomes many machine instructions.
    pub instructions: InstructionCounts,
}

#[derive(Debug, Serialize)]
pub struct InstructionCounts {
    pub total: Count,
    /// Warp-level issues: one per warp with a thread in the block, per
    /// execution. Per-CTA totals only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warps: Option<Count>,
    /// Keys are the report's kind labels: "tensor f16", "cuda-core
    /// f32", "sfu f32", "global load 16 B", "global -> shared copy
    /// 16 B", "global atomic 4 B", "integer arithmetic", "compare /
    /// select", "conversion", "register move", "control",
    /// "synchronization", "warp communication", "hint / no-op",
    /// "unknown".
    pub by_kind: BTreeMap<String, KindCounts>,
}

#[derive(Debug, Serialize)]
pub struct KindCounts {
    pub total: Count,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warps: Option<Count>,
    /// By opcode as PTX spells it (`fma.rn.f32`); sums to `total`.
    pub opcodes: BTreeMap<String, Count>,
    /// Warp-level issues per opcode; per-CTA totals only.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub opcode_warps: BTreeMap<String, Count>,
    /// Different per-thread, per-execution contributions under each opcode.
    /// Variant issued counts sum to the opcode count. Contributions describe
    /// work when the predicate is true, not work per issued instruction.
    pub contribution_variants: BTreeMap<String, Vec<InstructionVariant>>,
}

#[derive(Debug, Serialize)]
pub struct InstructionVariant {
    pub issued: Count,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warps: Option<Count>,
    pub contributions_per_execution: Vec<ContributionDetails>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContributionDetails {
    Flops {
        pipe: String,
        precision: String,
        count: u64,
    },
    Bytes {
        space: String,
        direction: String,
        count: u64,
    },
    UnquantifiedBytes {
        space: String,
        direction: String,
    },
    Conversions {
        count: u64,
    },
    NonFlopOps {
        operation: String,
        count: u64,
    },
    SyncOps {
        count: u64,
    },
    CommunicationOps {
        count: u64,
    },
    ControlOps {
        count: u64,
    },
    UnknownOps {
        mnemonic: String,
    },
}

/// A flop/byte ratio with the direction it is known in: `exact`,
/// `at_least` (exact flops over bytes that are an upper bound) or
/// `at_most` (flops that are an upper bound over exact bytes).
#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
pub struct Intensity {
    pub value: f64,
    pub bound: Bound,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    Exact,
    AtLeast,
    AtMost,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct DirectionCounts {
    pub load: Count,
    pub store: Count,
}

#[derive(Debug, Serialize)]
pub struct UnknownEntry {
    pub what: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    pub reason: String,
}
