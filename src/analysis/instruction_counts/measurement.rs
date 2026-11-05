//! Measurement: the canonical record of one instruction's contribution.
//!
//! `count` is the magnitude contributed by ONE execution of the
//! instruction by ONE thread (flops for `Flops`, bytes for `Bytes`,
//! 1 for the others). A warp-collective instruction (the
//! tensor families) contributes its warp total divided by the 32
//! lanes that issue it, so every count adds up per thread. Loop trip
//! multiplication happens at report aggregation (PR 12); until then
//! everything is per-execution.
//!
//! Honesty is in the kinds: an instruction that moves statically
//! unquantifiable bytes becomes `UnquantifiedBytes` (an op count with
//! a visible hole), an unhandled instruction becomes `UnknownOps`
//! — the v1 `AsyncCopy{bytes unset} → 0` silent-zero bug
//! class is unrepresentable.

use crate::ptx::modifier::Space;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Precision {
    F16,
    BF16,
    /// Tensor-core only (PTX ISA §9.7.15.2): 32-bit storage, 10-bit
    /// mantissa.
    TF32,
    F32,
    F64,
    /// Tensor-core only: the 8-bit `.e4m3` and `.e5m2` multiplicands
    /// (PTX ISA §9.7.15.5.1, `mma` with `.kind::f8f6f4` shapes).
    FP8,
}

impl Precision {
    pub const ALL: [Precision; 6] = [
        Precision::FP8,
        Precision::F16,
        Precision::BF16,
        Precision::TF32,
        Precision::F32,
        Precision::F64,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
            Precision::TF32 => "tf32",
            Precision::F32 => "f32",
            Precision::F64 => "f64",
            Precision::FP8 => "fp8",
        }
    }
}

/// FLOP accounting bucket; each gets its own table in the report.
/// Atomic work does not imply a fixed physical execution unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pipe {
    CudaCore,
    Tensor,
    /// Special function unit: transcendentals, reciprocals, roots.
    Sfu,
    /// Atomic/reduction arithmetic; an accounting bucket, not a fixed hardware unit.
    Atomic,
}

impl Pipe {
    pub const ALL: [Pipe; 4] = [Pipe::CudaCore, Pipe::Tensor, Pipe::Sfu, Pipe::Atomic];

    pub fn key(self) -> &'static str {
        match self {
            Pipe::CudaCore => "cuda-core",
            Pipe::Tensor => "tensor",
            Pipe::Sfu => "sfu",
            Pipe::Atomic => "atomic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Load,
    Store,
}

/// The kind of a non-flop arithmetic instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArithKind {
    /// `cvt` — precision/width conversion.
    Conversion,
    /// Integer/bit arithmetic: address math, masks, shifts.
    Integer,
    /// Predicate computation and selection: `setp`, `selp`, `set`...
    Predicate,
    /// Register/address bookkeeping: `mov`, `cvta`.
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeasureKind {
    Flops {
        pipe: Pipe,
        precision: Precision,
    },
    Bytes {
        space: Space,
        direction: Direction,
    },
    /// A memory op whose byte count is statically unknowable: counted
    /// as an op, surfaced in the unquantified counter — never zero.
    UnquantifiedBytes {
        space: Space,
        direction: Direction,
    },
    /// An instruction the classifier does not handle.
    UnknownOps,
}

/// A contribution before execution context is attached by the collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Contribution {
    pub kind: MeasureKind,
    pub count: u64,
    /// Index among the instruction's memory operands (not all operands).
    /// Read and write contributions of an atomic reference the same operand.
    pub memory_operand: Option<usize>,
}

impl Contribution {
    pub fn new(kind: MeasureKind, count: u64) -> Self {
        Self {
            kind,
            count,
            memory_operand: None,
        }
    }

    pub fn memory(space: Space, direction: Direction, bytes: Option<u32>, operand: usize) -> Self {
        let (kind, count) = match bytes {
            Some(bytes) => (MeasureKind::Bytes { space, direction }, u64::from(bytes)),
            None => (MeasureKind::UnquantifiedBytes { space, direction }, 1),
        };
        Self {
            kind,
            count,
            memory_operand: Some(operand),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Measurement {
    pub kind: MeasureKind,
    pub count: u64,
    /// Statement index into the owning kernel's `stmts` — provenance
    /// for diagnostics and the report verifier.
    pub provenance: usize,
    /// The contribution's memory operand, as in [`Contribution`].
    pub memory_operand: Option<usize>,
}
