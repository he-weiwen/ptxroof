//! Measurement: the canonical record of one instruction's contribution.
//!
//! `count` is the magnitude contributed by ONE execution of the
//! instruction by ONE thread (flops for `Flops`, bytes for `Bytes`,
//! 1 for op-counting kinds). A warp-collective instruction (the
//! tensor families) contributes its warp total divided by the 32
//! lanes that issue it, so every count adds up per thread. Loop trip
//! multiplication happens at report aggregation (PR 12); until then
//! everything is per-execution and the constants stay exact.
//!
//! Honesty is in the kinds: an instruction that moves statically
//! unquantifiable bytes becomes `UnquantifiedBytes` (an op count with
//! a visible hole), an unhandled instruction becomes `UnknownOp` with
//! its mnemonic — the v1 `AsyncCopy{bytes unset} → 0` silent-zero bug
//! class is unrepresentable.

use crate::analysis::instruction_counts::classify::{ArithKind, Direction, Pipe, Precision, Space};
use crate::support::intern::Symbol;

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
    /// `cvt` ops — the precision-conversion overhead (S8).
    Conversions,
    /// Integer/predicate/move bookkeeping ops.
    NonFlopOps {
        kind: ArithKind,
    },
    SyncOps,
    CommunicationOps,
    ControlOps,
    /// An instruction the classifier does not handle: counted by
    /// mnemonic, reported by name.
    UnknownOps {
        mnemonic: Symbol,
    },
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
    /// The instruction itself is `@%p`-guarded: its count is an upper
    /// bound regardless of where its block sits.
    pub predicated: bool,
    /// Statement index into the owning kernel's `stmts` — provenance
    /// for diagnostics and the report verifier.
    pub provenance: usize,
}
