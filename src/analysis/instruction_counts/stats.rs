//! Stats: filter queries over the per-block measurement stream.
//!
//! The soft-filter rule (v1's, documented once here): a `None` filter
//! axis matches everything; a `Some` axis selects only measurements of
//! kinds that *carry* that axis, with equal value. Consequently
//! `bytes(...)` sums only statically-known byte counts — measurements
//! whose bytes are unknowable are a separate query
//! (`unquantified_memory_ops`) that report code must surface alongside
//! every byte total it prints; the pairing is the honesty principle,
//! and the report verifier's accounting identity keeps it true.
//!
//! Every tally carries a [`CountQualifier`]: `Exact` only if every
//! contributing measurement sits in an exact block and is itself
//! unpredicated. An empty selection is exact (a true zero).

use super::collect::{BlockMeasurements, CountQualifier};
use crate::analysis::control_flow::BlockId;
use crate::analysis::instruction_counts::classify::{ArithKind, Direction, Precision, Space};
use crate::analysis::instruction_counts::measurement::MeasureKind;
use crate::support::intern::Symbol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    /// Sum of measurement counts (flops, bytes, or ops by kind).
    pub value: u64,
    /// Number of contributing instructions.
    pub ops: u64,
    pub qualifier: CountQualifier,
}

impl Tally {
    const ZERO: Tally = Tally {
        value: 0,
        ops: 0,
        qualifier: CountQualifier::Exact,
    };
}

pub struct Stats<'a> {
    blocks: &'a [BlockMeasurements],
}

impl<'a> Stats<'a> {
    pub fn new(blocks: &'a [BlockMeasurements]) -> Self {
        Stats { blocks }
    }

    pub fn all_blocks(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|b| b.block).collect()
    }

    fn tally(&self, blocks: &[BlockId], mut select: impl FnMut(&MeasureKind) -> bool) -> Tally {
        let mut t = Tally::ZERO;
        for &bid in blocks {
            let b = &self.blocks[bid.0 as usize];
            for m in &b.measurements {
                if select(&m.kind) {
                    t.value += m.count;
                    t.ops += 1;
                    let q = if m.predicated {
                        CountQualifier::AtMost
                    } else {
                        b.qualifier
                    };
                    t.qualifier = t.qualifier.and(q);
                }
            }
        }
        t
    }

    /// Flop total over `blocks`, optionally restricted to one precision.
    pub fn flops(&self, blocks: &[BlockId], precision: Option<Precision>) -> Tally {
        self.tally(blocks, |k| match k {
            MeasureKind::Flops { precision: p, .. } => precision.is_none_or(|want| *p == want),
            _ => false,
        })
    }

    /// Statically-known byte total over `blocks`, filtered by space
    /// and/or direction. Pair with [`Stats::unquantified_memory_ops`].
    pub fn bytes(
        &self,
        blocks: &[BlockId],
        space: Option<Space>,
        direction: Option<Direction>,
    ) -> Tally {
        self.tally(blocks, |k| match k {
            MeasureKind::Bytes {
                space: s,
                direction: d,
            } => space.is_none_or(|want| *s == want) && direction.is_none_or(|want| *d == want),
            _ => false,
        })
    }

    /// Memory ops whose byte count is statically unknowable.
    pub fn unquantified_memory_ops(&self, blocks: &[BlockId]) -> Tally {
        self.tally(blocks, |k| {
            matches!(k, MeasureKind::UnquantifiedBytes { .. })
        })
    }

    /// `cvt` op count (S8's conversion-overhead column).
    pub fn conversions(&self, blocks: &[BlockId]) -> Tally {
        self.tally(blocks, |k| matches!(k, MeasureKind::Conversions))
    }

    pub fn non_flop_ops(&self, blocks: &[BlockId], kind: Option<ArithKind>) -> Tally {
        self.tally(blocks, |k| match k {
            MeasureKind::NonFlopOps { kind: g } => kind.is_none_or(|want| *g == want),
            _ => false,
        })
    }

    pub fn sync_ops(&self, blocks: &[BlockId]) -> Tally {
        self.tally(blocks, |k| matches!(k, MeasureKind::SyncOps))
    }

    /// Unknown-instruction tallies by mnemonic, sorted by symbol for
    /// deterministic output.
    pub fn unknown_ops(&self, blocks: &[BlockId]) -> Vec<(Symbol, u64)> {
        let mut map: std::collections::BTreeMap<Symbol, u64> = Default::default();
        for &bid in blocks {
            for m in &self.blocks[bid.0 as usize].measurements {
                if let MeasureKind::UnknownOps { mnemonic } = m.kind {
                    *map.entry(mnemonic).or_default() += m.count;
                }
            }
        }
        map.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::instruction_counts::classify::Pipe;
    use crate::analysis::instruction_counts::measurement::Measurement;

    fn block(
        id: u32,
        qualifier: CountQualifier,
        ms: Vec<(MeasureKind, u64, bool)>,
    ) -> BlockMeasurements {
        BlockMeasurements {
            block: BlockId(id),
            qualifier,
            measurements: ms
                .into_iter()
                .map(|(kind, count, predicated)| Measurement {
                    kind,
                    count,
                    predicated,
                    provenance: 0,
                })
                .collect(),
            class_counts: Default::default(),
            instructions: Default::default(),
        }
    }

    fn fixture() -> Vec<BlockMeasurements> {
        use MeasureKind::*;
        vec![
            block(
                0,
                CountQualifier::Exact,
                vec![
                    (
                        Flops {
                            pipe: Pipe::CudaCore,
                            precision: Precision::F32,
                        },
                        8,
                        false,
                    ),
                    (
                        Bytes {
                            space: Space::Global,
                            direction: Direction::Load,
                        },
                        16,
                        false,
                    ),
                    (
                        Bytes {
                            space: Space::Shared,
                            direction: Direction::Store,
                        },
                        4,
                        false,
                    ),
                    (Conversions, 1, false),
                ],
            ),
            block(
                1,
                CountQualifier::AtMost,
                vec![
                    (
                        Flops {
                            pipe: Pipe::CudaCore,
                            precision: Precision::F32,
                        },
                        2,
                        false,
                    ),
                    (
                        Flops {
                            pipe: Pipe::CudaCore,
                            precision: Precision::F16,
                        },
                        4,
                        false,
                    ),
                ],
            ),
            block(
                2,
                CountQualifier::Exact,
                vec![
                    (
                        Bytes {
                            space: Space::Global,
                            direction: Direction::Store,
                        },
                        2,
                        true,
                    ),
                    (
                        UnquantifiedBytes {
                            space: Space::Global,
                            direction: Direction::Load,
                        },
                        1,
                        false,
                    ),
                ],
            ),
        ]
    }

    fn ids(n: u32) -> Vec<BlockId> {
        (0..n).map(BlockId).collect()
    }

    #[test]
    fn soft_filter_none_matches_all_some_restricts() {
        let blocks = fixture();
        let s = Stats::new(&blocks);
        assert_eq!(s.flops(&ids(3), None).value, 14);
        assert_eq!(s.flops(&ids(3), Some(Precision::F32)).value, 10);
        assert_eq!(s.flops(&ids(3), Some(Precision::F16)).value, 4);
        assert_eq!(s.flops(&ids(3), Some(Precision::F64)).value, 0);
        assert_eq!(s.bytes(&ids(3), Some(Space::Global), None).value, 18);
        assert_eq!(
            s.bytes(&ids(3), Some(Space::Global), Some(Direction::Load))
                .value,
            16
        );
        assert_eq!(s.bytes(&ids(3), None, Some(Direction::Store)).value, 6);
    }

    #[test]
    fn bytes_exclude_unquantified_which_has_its_own_query() {
        let blocks = fixture();
        let s = Stats::new(&blocks);
        // The unquantified load contributes 0 bytes but 1 visible op.
        assert_eq!(
            s.bytes(&ids(3), Some(Space::Global), Some(Direction::Load))
                .ops,
            1
        );
        assert_eq!(s.unquantified_memory_ops(&ids(3)).ops, 1);
    }

    #[test]
    fn qualifier_propagates_from_blocks_and_predication() {
        let blocks = fixture();
        let s = Stats::new(&blocks);
        // Block 0 only: everything exact.
        assert_eq!(
            s.flops(&[BlockId(0)], None).qualifier,
            CountQualifier::Exact
        );
        // Mixing in the at_most block taints the total.
        assert_eq!(s.flops(&ids(2), None).qualifier, CountQualifier::AtMost);
        // A predicated instruction taints even an exact block.
        assert_eq!(
            s.bytes(&[BlockId(2)], None, Some(Direction::Store))
                .qualifier,
            CountQualifier::AtMost
        );
        // Empty selection is an exact zero.
        assert_eq!(
            s.flops(&[], None),
            Tally {
                value: 0,
                ops: 0,
                qualifier: CountQualifier::Exact
            }
        );
    }

    #[test]
    fn block_subsets_select() {
        let blocks = fixture();
        let s = Stats::new(&blocks);
        assert_eq!(s.flops(&[BlockId(1)], None).value, 6);
        assert_eq!(s.conversions(&[BlockId(0)]).ops, 1);
        assert_eq!(s.conversions(&[BlockId(1)]).ops, 0);
    }
}
