//! What one warp's execution of a memory instruction touches: the
//! 32-byte sectors the addresses of its executing lanes fall in, from the affine address, the element width, the
//! block shape and the threads that execute the instruction.
//! The lane-dependent part is evaluated for every lane; the uniform
//! part (CTA index, loop counters, parameters) only shifts the whole
//! set, so every alignment of it under which the lanes' accesses are
//! aligned is tried and the count is reported as a range.

use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::instruction_counts::classify::Space;
use crate::analysis::scalar::affine::{Affine, Axis, Var};
use crate::analysis::scalar::lane_eval::{depends_on_lane, eval};
use crate::analysis::scalar::symexpr::SymExpr;
use crate::analysis::thread_participation::ThreadSet;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};

pub const SECTOR: i64 = 32;

/// Offsets within 2^62 bytes leave room to add an access size, a
/// sector and an alignment in i64.
const OFFSET_LIMIT: i128 = 1 << 62;

/// Whether the terms of `a` that `varies` picks sum within
/// `OFFSET_LIMIT` bytes of 0 at every value `bound` gives them, each
/// term's range taken on its own; each bound holds 0, so every partial
/// sum does too.
pub fn offsets_fit(
    a: &Affine,
    varies: impl Fn(&Var) -> bool,
    bound: &dyn Fn(&Var) -> Option<(i64, i64)>,
) -> bool {
    let mut sum = Some((0i128, 0i128));
    for (v, c) in a.terms.iter().filter(|(v, _)| varies(v)) {
        let Some((lo, hi)) = Affine::term(v.clone(), c.clone()).range(bound) else {
            return false;
        };
        sum = sum.and_then(|(x, y)| Some((x.checked_add(lo)?, y.checked_add(hi)?)));
    }
    sum.is_some_and(|(lo, hi)| -OFFSET_LIMIT <= lo && hi <= OFFSET_LIMIT)
}

/// A thread index's values in a block of `shape`.
pub fn in_block(shape: [u32; 3]) -> impl Fn(&Var) -> Option<(i64, i64)> {
    move |v| {
        let n = match v {
            Var::Tid(Axis::X) => shape[0],
            Var::Tid(Axis::Y) => shape[1],
            Var::Tid(Axis::Z) => shape[2],
            _ => return None,
        };
        Some((0, i64::from(n) - 1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CountRange {
    pub min: u32,
    pub max: u32,
}

/// Sectors per warp request, and the fewest the distinct bytes the
/// executing lanes touch could occupy if packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    pub sectors: CountRange,
    pub ideal: CountRange,
}

/// Sectors per warp request of an access of `bytes` at an address
/// aligned to `align` (the access size; `cp.async`'s cp-size when a
/// smaller src-size is read), or why they cannot be counted:
/// a lane coefficient that is not a constant (bind the parameter), or
/// a block shape that is not known (`.reqntid` or `--launch`).
pub fn warp_footprint(
    a: &Affine,
    bytes: u32,
    align: u32,
    shape: [u32; 3],
    executes: impl Fn([i64; 3]) -> bool,
) -> Result<Footprint, String> {
    if bytes == 0 {
        let zero = CountRange { min: 0, max: 0 };
        return Ok(Footprint {
            sectors: zero,
            ideal: zero,
        });
    }
    let mut lane_terms: Vec<(&Var, i64)> = Vec::new();
    for (v, c) in &a.terms {
        if !depends_on_lane(v) {
            continue;
        }
        match c.as_const() {
            Some(k) => lane_terms.push((v, k)),
            None => return Err(format!("the coefficient of {v} is {c}; bind it")),
        }
    }
    let [nx, ny, nz] = shape.map(i64::from);
    let threads = nx * ny * nz;
    if threads == 0 {
        return Err("block shape unknown: no .reqntid; pass --launch".to_owned());
    }
    if !offsets_fit(a, depends_on_lane, &in_block(shape)) {
        return Err("lane offsets reach 2^62 bytes".to_owned());
    }
    let bytes = i64::from(bytes);
    let align = i64::from(align.max(1));
    let warps: Vec<Vec<i64>> = (0..(threads + 31) / 32)
        .map(|warp| {
            (warp * 32..((warp + 1) * 32).min(threads))
                .map(|t| [t % nx, (t / nx) % ny, t / (nx * ny)])
                .filter(|&tid| executes(tid))
                .map(|tid| lane_terms.iter().map(|(v, k)| k * eval(v, tid)).sum())
                .collect()
        })
        .filter(|offsets: &Vec<i64>| !offsets.is_empty())
        .collect();
    // PTX ISA §6.4.1 (Addresses as Operands): an address must be aligned
    // to the access size, else the behavior is undefined. So the uniform
    // part is congruent to minus the lanes' common residue.
    let residue = warps.first().map_or(0, |w| w[0].rem_euclid(align));
    if warps
        .iter()
        .flatten()
        .any(|o| o.rem_euclid(align) != residue)
    {
        return Err(format!(
            "lane addresses differ modulo the {align} B access size"
        ));
    }
    let (mut min, mut max) = (u32::MAX, 0u32);
    let (mut ideal_min, mut ideal_max) = (u32::MAX, 0u32);
    for offsets in &warps {
        let mut sorted = offsets.clone();
        sorted.sort_unstable();
        let mut distinct = 0;
        let mut end = i64::MIN;
        for &o in &sorted {
            distinct += (o + bytes - end.max(o)).max(0);
            end = end.max(o + bytes);
        }
        let packed = ((distinct + SECTOR - 1) / SECTOR) as u32;
        ideal_min = ideal_min.min(packed);
        ideal_max = ideal_max.max(packed);
        let mut shift = (align - residue) % align;
        while shift < SECTOR {
            let touched: BTreeSet<i64> = offsets
                .iter()
                .flat_map(|&o| {
                    let lo = (o + shift).div_euclid(SECTOR);
                    let hi = (o + shift + bytes - 1).div_euclid(SECTOR);
                    lo..=hi
                })
                .collect();
            let n = touched.len() as u32;
            min = min.min(n);
            max = max.max(n);
            shift += align;
        }
    }
    if min == u32::MAX {
        let none = CountRange { min: 0, max: 0 };
        return Ok(Footprint {
            sectors: none,
            ideal: none,
        });
    }
    Ok(Footprint {
        sectors: CountRange { min, max },
        ideal: CountRange {
            min: ideal_min,
            max: ideal_max,
        },
    })
}

/// One warp's lane pattern of an access inside a loop: the sectors it
/// touches at every alignment of its base, and how the counter moves it.
pub struct SectorPattern {
    by_residue: [u32; SECTOR as usize],
    /// The lanes' common residue modulo the alignment; none when they
    /// differ, which no base aligns.
    residue: Option<i64>,
    align: i64,
    stride: i64,
}

impl SectorPattern {
    pub fn new(offsets: &[i64], bytes: i64, align: i64, stride: i64) -> SectorPattern {
        let mut by_residue = [0; SECTOR as usize];
        for (shift, slot) in by_residue.iter_mut().enumerate() {
            let touched: BTreeSet<i64> = offsets
                .iter()
                .filter(|_| bytes > 0)
                .flat_map(|&o| {
                    (o + shift as i64).div_euclid(SECTOR)
                        ..=(o + shift as i64 + bytes - 1).div_euclid(SECTOR)
                })
                .collect();
            *slot = touched.len() as u32;
        }
        SectorPattern {
            by_residue,
            residue: offsets
                .first()
                .map(|o| o.rem_euclid(align))
                .filter(|&r| offsets.iter().all(|o| o.rem_euclid(align) == r)),
            align,
            stride,
        }
    }

    /// Whether a base at `rho` modulo a sector keeps every iteration's
    /// access aligned (PTX ISA §6.4.1); iterations count from 1.
    pub fn aligned(&self, rho: i64, trips: i64) -> bool {
        self.residue.is_some_and(|r| {
            (1..=trips).all(|k| (rho + r + self.stride * k).rem_euclid(self.align) == 0)
        })
    }

    /// Sectors touched over the loop with the base at `rho`.
    pub fn total(&self, rho: i64, trips: i64) -> u64 {
        (1..=trips)
            .map(|k| {
                u64::from(self.by_residue[(rho + self.stride * k).rem_euclid(SECTOR) as usize])
            })
            .sum()
    }
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct LoopBytes {
    pub requested: u64,
    /// The distinct bytes: at most the sum over the bases, at least the
    /// largest base on each pointer, summed (bases on one pointer may
    /// overlap); `None` when an access's address is not read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unique: Option<ByteRange>,
    /// The bytes of the 32-byte sectors those requests touch, over the
    /// alignments of each base pointer, when every access on it is
    /// aligned under one of them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sector_bytes: Option<ByteRange>,
    /// Some contributing access is predicated: all are upper bounds.
    pub at_most: bool,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub min: u64,
    pub max: u64,
}

/// A memory operand's bound affine address, for the per-loop footprint.
pub struct AccessForm {
    /// `None` when the address is not read.
    pub form: Option<Affine>,
    pub bytes: u32,
    /// The access size the address is aligned to.
    pub align: u32,
    pub space: Space,
    /// The threads that execute the instruction.
    pub set: ThreadSet,
    /// The instruction's guard, if any, is a thread-index condition.
    pub guard_exact: bool,
    /// The block may not run on every iteration of its loop, and no
    /// thread-index selection explains that.
    pub conditional: bool,
    /// The address holds a uniform variable defined in its loop.
    pub uniform_varies: bool,
}

/// Global bytes one CTA requests over one execution of loop `id`'s
/// own blocks, and the distinct bytes they touch: every lane of every
/// executing thread at every iteration, as intervals merged per
/// base (the form without its lane, counter and constant parts).
/// Bases on different pointers are assumed disjoint; bases on one
/// pointer, or on none known, may overlap, so the distinct bytes are
/// at most their sum and at least the largest on each pointer. Needs
/// a numeric trip count, the block shape, and every global address's
/// lane and counter coefficients as constants.
pub fn loop_bytes(
    id: LoopId,
    trips: i64,
    shape: [u32; 3],
    forms: &[AccessForm],
    pointer: impl Fn(&Affine) -> Option<String>,
) -> Option<LoopBytes> {
    let [nx, ny, nz] = shape.map(i64::from);
    let threads = nx * ny * nz;
    if threads == 0 {
        return None;
    }
    let global: Vec<&AccessForm> = forms
        .iter()
        .filter(|f| matches!(f.space, Space::Global | Space::Generic))
        .collect();
    if global.is_empty() {
        return None;
    }
    let mut requested = 0u64;
    let mut complete = true;
    let mut at_most = false;
    let mut per_base: HashMap<Affine, Vec<(i64, i64)>> = HashMap::new();
    // Each warp's pattern, with the trips it is summed over.
    let mut patterns: HashMap<Affine, Vec<(SectorPattern, i64)>> = HashMap::new();
    for f in global {
        let set = &f.set;
        let lanes = (0..threads)
            .filter(|&t| set.contains([t % nx, (t / nx) % ny, t / (nx * ny)], shape))
            .count() as u64;
        let bytes = u64::from(f.bytes)
            .checked_mul(trips as u64)?
            .checked_mul(lanes)?;
        requested = requested.checked_add(bytes)?;
        at_most |= !f.guard_exact || f.conditional;
        let varies = |v: &Var| {
            crate::analysis::scalar::lane_eval::depends_on_lane(v) || at_trip(v, id, 0).is_some()
        };
        let bound = |v: &Var| match v {
            Var::Iter(l) if *l == id => Some((0, trips)),
            _ => in_block(shape)(v),
        };
        let read = |a: &&Affine| offsets_fit(a, varies, &bound) && !f.uniform_varies;
        let Some(form) = (f.form.as_ref()).filter(read) else {
            complete &= f.bytes == 0 || lanes == 0;
            continue;
        };
        let stride = match form.terms.get(&Var::Iter(id)) {
            Some(c) => c.as_const()?,
            None => 0,
        };
        // A floor or mod of the counter moves the address by its
        // value at each trip.
        let mut stepped = Vec::new();
        for (v, c) in &form.terms {
            if *v != Var::Iter(id) && at_trip(v, id, 0).is_some() {
                stepped.push((v, c.as_const()?));
            }
        }
        // One sector pattern per warp and trip: within the budget the
        // intervals keep below.
        if !stepped.is_empty() && trips.saturating_mul((threads + 31) / 32) > 1 << 22 {
            return None;
        }
        let shift = |k: i64| -> i64 {
            stride * k
                + (stepped.iter())
                    .map(|(v, c)| c * at_trip(v, id, k).unwrap_or(0))
                    .sum::<i64>()
        };
        if form.terms.iter().any(|(v, c)| {
            crate::analysis::scalar::lane_eval::depends_on_lane(v) && c.as_const().is_none()
        }) {
            return None;
        }
        let bytes = i64::from(f.bytes);
        // Which lane an election picks is unspecified: a lane-dependent
        // address touched by the elected lane is a bound.
        if f.set.pred.elect_active().is_some()
            && form
                .terms
                .keys()
                .any(crate::analysis::scalar::lane_eval::depends_on_lane)
        {
            at_most = true;
        }
        let mut key = form.clone();
        key.terms.retain(|v, _| {
            !crate::analysis::scalar::lane_eval::depends_on_lane(v) && at_trip(v, id, 0).is_none()
        });
        key.base = SymExpr::sub(key.base.clone(), SymExpr::Const(key.base.const_part()));
        let intervals = per_base.entry(key.clone()).or_default();
        for warp in 0..(threads + 31) / 32 {
            let offsets: Vec<i64> = (warp * 32..((warp + 1) * 32).min(threads))
                .map(|t| [t % nx, (t / nx) % ny, t / (nx * ny)])
                .filter(|&tid| set.contains(tid, shape))
                .map(|tid| crate::analysis::scalar::lane_eval::eval_lane(form, tid))
                .collect();
            let align = i64::from(f.align);
            let pats = patterns.entry(key.clone()).or_default();
            if offsets.is_empty() {
            } else if stepped.is_empty() {
                pats.push((SectorPattern::new(&offsets, bytes, align, stride), trips));
            } else {
                for k in 1..=trips {
                    let at: Vec<i64> = offsets.iter().map(|o| o + shift(k)).collect();
                    pats.push((SectorPattern::new(&at, bytes, align, 0), 1));
                }
            }
        }
        for t in 0..threads {
            let tid = [t % nx, (t / nx) % ny, t / (nx * ny)];
            if !set.contains(tid, shape) {
                continue;
            }
            let base = crate::analysis::scalar::lane_eval::eval_lane(form, tid);
            if intervals.len() as i64 + trips > 1 << 22 {
                return None;
            }
            for k in 1..=trips {
                let start = base + shift(k);
                intervals.push((start, start + bytes));
            }
        }
    }
    let (mut most, mut loose) = (0i64, 0i64);
    let mut largest: HashMap<String, i64> = HashMap::new();
    for (key, intervals) in per_base {
        let len = union_len(intervals);
        most += len;
        match pointer(&key) {
            Some(p) => {
                let m = largest.entry(p).or_default();
                *m = (*m).max(len);
            }
            None => loose = loose.max(len),
        }
    }
    let unique = ByteRange {
        min: largest.values().sum::<i64>().max(loose) as u64,
        max: most as u64,
    };
    let mut sector_bytes = Some(ByteRange { min: 0, max: 0 });
    for pats in patterns.values() {
        let totals: Vec<u64> = (0..SECTOR)
            .filter(|&rho| pats.iter().all(|(p, n)| p.aligned(rho, *n)))
            .map(|rho| pats.iter().map(|(p, n)| p.total(rho, *n)).sum())
            .collect();
        sector_bytes = match (sector_bytes, totals.iter().min(), totals.iter().max()) {
            (Some(s), Some(&lo), Some(&hi)) => Some(ByteRange {
                min: s.min + lo * SECTOR as u64,
                max: s.max + hi * SECTOR as u64,
            }),
            _ => None,
        };
    }
    Some(LoopBytes {
        requested,
        unique: complete.then_some(unique),
        sector_bytes: sector_bytes.filter(|_| complete),
        at_most,
    })
}

/// The bytes the half-open intervals cover.
fn union_len(mut intervals: Vec<(i64, i64)>) -> i64 {
    intervals.sort_unstable();
    let mut len = 0;
    let mut cur: Option<(i64, i64)> = None;
    for (s, e) in intervals {
        match cur {
            Some((cs, ce)) if s <= ce => cur = Some((cs, ce.max(e))),
            Some((cs, ce)) => {
                len += ce - cs;
                cur = Some((s, e));
            }
            None => cur = Some((s, e)),
        }
    }
    len + cur.map_or(0, |(cs, ce)| ce - cs)
}

/// `v` at trip `k` of loop `id`, when `v` is that loop's counter or a
/// floor or mod of it.
pub fn at_trip(v: &Var, id: LoopId, k: i64) -> Option<i64> {
    match v {
        Var::Iter(l) if *l == id => Some(k),
        Var::Div(inner, d) => at_trip(inner, id, k).map(|x| x.div_euclid(*d)),
        Var::Mod(inner, m) => at_trip(inner, id, k).map(|x| x.rem_euclid(*m)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::thread_participation::Pred;

    fn tid() -> Affine {
        Affine::var(Var::Tid(Axis::X))
    }

    fn fp(a: &Affine, bytes: u32, shape: [u32; 3]) -> (u32, u32) {
        fp_of(a, bytes, shape, |_| true)
    }

    fn fp_of(
        a: &Affine,
        bytes: u32,
        shape: [u32; 3],
        executes: impl Fn([i64; 3]) -> bool,
    ) -> (u32, u32) {
        fp_aligned(a, bytes, bytes, shape, executes)
    }

    fn fp_aligned(
        a: &Affine,
        bytes: u32,
        align: u32,
        shape: [u32; 3],
        executes: impl Fn([i64; 3]) -> bool,
    ) -> (u32, u32) {
        let f = warp_footprint(a, bytes, align, shape, executes).unwrap();
        (f.sectors.min, f.sectors.max)
    }

    #[test]
    fn coalesced_strided_and_broadcast_warps() {
        let c = |n: i64| SymExpr::Const(n);
        // 32 lanes × 4 B contiguous: four sectors when aligned, one more
        // when the base sits inside a sector.
        assert_eq!(fp(&tid().scale(c(4)), 4, [128, 1, 1]), (4, 5));
        // 2 B elements: two sectors.
        assert_eq!(fp(&tid().scale(c(2)), 2, [64, 1, 1]), (2, 3));
        // A 128 B stride per lane: every lane its own sector.
        assert_eq!(fp(&tid().scale(c(128)), 4, [32, 1, 1]), (32, 32));
        // The same address for every lane.
        assert_eq!(
            fp(&Affine::invariant(SymExpr::sym("p")), 4, [32, 1, 1]),
            (1, 1)
        );
        // k5's A tile: 4 rows of 8 halves per warp, rows 512 B apart.
        let row = Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 8)).scale(c(512));
        let col = Affine::var(Var::Mod(Box::new(Var::Tid(Axis::X)), 8)).scale(c(2));
        assert_eq!(fp(&(row + col), 2, [64, 1, 1]), (4, 8));
        // A 2D block: a warp is two rows of 16 lanes, rows 64 B apart.
        let a = tid().scale(c(4)) + Affine::var(Var::Tid(Axis::Y)).scale(c(64));
        assert_eq!(fp(&a, 4, [16, 16, 1]), (4, 5));
    }

    #[test]
    fn only_the_executing_lanes_touch_sectors() {
        let c = |n: i64| SymExpr::Const(n);
        let coalesced = tid().scale(c(4));
        // One lane per warp: one sector, whatever the alignment.
        assert_eq!(
            fp_of(&coalesced, 4, [128, 1, 1], |t| t[0] % 32 == 0),
            (1, 1)
        );
        // Every other lane still spans the same four sectors.
        assert_eq!(fp_of(&coalesced, 4, [128, 1, 1], |t| t[0] % 2 == 0), (4, 5));
        // The upper half of a 64-thread block: only its warp counts.
        assert_eq!(fp_of(&coalesced, 4, [64, 1, 1], |t| t[0] >= 32), (4, 5));
        // No executing lane at this shape.
        assert_eq!(fp_of(&coalesced, 4, [32, 1, 1], |_| false), (0, 0));
    }

    #[test]
    fn a_pattern_sums_its_sectors_over_the_aligned_bases() {
        // 32 lanes × 4 B, moved 128 B a trip: 4 sectors a trip when the
        // base starts a sector, 5 otherwise; only 4-aligned bases apply.
        let offsets: Vec<i64> = (0..32).map(|t| 4 * t).collect();
        let p = SectorPattern::new(&offsets, 4, 4, 128);
        assert!(p.aligned(0, 8) && p.aligned(4, 8) && !p.aligned(2, 8));
        assert_eq!((p.total(0, 8), p.total(4, 8)), (32, 40));
        // A 2 B step per trip on 4 B words misaligns odd trips.
        assert!(!SectorPattern::new(&offsets, 4, 4, 2).aligned(0, 2));
        // Lanes at different residues: no base aligns them.
        let odd: Vec<i64> = (0..32).map(|t| 2 * t).collect();
        assert!(!SectorPattern::new(&odd, 4, 4, 0).aligned(0, 1));
        // One word for the whole warp: one sector a trip.
        assert_eq!(SectorPattern::new(&[0; 32], 4, 4, 0).total(28, 8), 8);
        // A copy of no bytes touches nothing.
        assert_eq!(SectorPattern::new(&offsets, 0, 16, 0).total(16, 4), 0);
        // Two loads on one base, `4·tid + 4·k − 4` and `4·tid + 4`, over two
        // trips from k = 1: 17 sectors at the base residue 28, not the 18
        // a k = 0 start would give.
        let a = SectorPattern::new(&(0..32).map(|t| 4 * t - 4).collect::<Vec<_>>(), 4, 4, 4);
        let b = SectorPattern::new(&(0..32).map(|t| 4 * t + 4).collect::<Vec<_>>(), 4, 4, 0);
        let best = (0..SECTOR)
            .filter(|&rho| a.aligned(rho, 2) && b.aligned(rho, 2))
            .map(|rho| a.total(rho, 2) + b.total(rho, 2))
            .min();
        assert_eq!(best, Some(17));
    }

    #[test]
    fn the_ideal_packs_the_executing_lanes_bytes() {
        let c = |n: i64| SymExpr::Const(n);
        let ideal = |a: &Affine, bytes: u32, shape: [u32; 3], executes: fn([i64; 3]) -> bool| {
            let f = warp_footprint(a, bytes, bytes, shape, executes).unwrap();
            (f.ideal.min, f.ideal.max)
        };
        // 32 lanes × 2 B pack into two sectors; at a 128 B stride they take 32.
        assert_eq!(ideal(&tid().scale(c(128)), 2, [32, 1, 1], |_| true), (2, 2));
        // One address for the whole warp: one sector is all it needs.
        assert_eq!(
            ideal(&Affine::invariant(SymExpr::sym("p")), 2, [32, 1, 1], |_| {
                true
            }),
            (1, 1)
        );
        // Lanes t and t + 16 read the same word: 64 distinct bytes, two sectors.
        let folded = Affine::var(Var::Mod(Box::new(Var::Tid(Axis::X)), 16)).scale(c(4));
        assert_eq!(ideal(&folded, 4, [32, 1, 1], |_| true), (2, 2));
        // 8 lanes × 2 B fit one sector; a 1-lane 4 B word too.
        assert_eq!(
            ideal(&tid().scale(c(2)), 2, [32, 1, 1], |t| t[0] < 8),
            (1, 1)
        );
        assert_eq!(
            ideal(&tid().scale(c(4)), 4, [32, 1, 1], |t| t[0] == 0),
            (1, 1)
        );
        // 33 lanes × 4 B: a full warp needs 4, the lone lane 1.
        assert_eq!(ideal(&tid().scale(c(4)), 4, [33, 1, 1], |_| true), (1, 4));
    }

    #[test]
    fn the_uniform_part_keeps_the_lanes_aligned() {
        let c = |n: i64| SymExpr::Const(n);
        // A 4 B word at 2·tid on lane 1 only sits at 2 mod 4, so the
        // uniform part is 2 mod 4 and the word never straddles a sector.
        assert_eq!(
            fp_of(&tid().scale(c(2)), 4, [32, 1, 1], |t| t[0] == 1),
            (1, 1)
        );
        // 4 B words 6 B apart on odd lanes: all at 2 mod 4.
        assert_eq!(
            fp_of(&tid().scale(c(6)), 4, [32, 1, 1], |t| t[0] % 2 == 1),
            (6, 7)
        );
        // A cp.async of 3 source bytes per 16 B slot is aligned to 16, not
        // 3: the slots start a sector or sit 16 B into one.
        assert_eq!(
            fp_aligned(&tid().scale(c(16)), 3, 16, [32, 1, 1], |_| true),
            (16, 17)
        );
        // On every lane no uniform part aligns them all.
        assert!(
            warp_footprint(&tid().scale(c(2)), 4, 4, [32, 1, 1], |_| true)
                .unwrap_err()
                .contains("modulo")
        );
    }

    #[test]
    fn offsets_fit_bounds_each_term() {
        let c = |n: i64| SymExpr::Const(n);
        let k = 1 << 60;
        // 2^60·x − 2^61·⌊x/2⌋ is at most 2^60, but 2^60·x alone reaches 2^63.
        let a = tid().scale(c(k))
            + Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 2)).scale(c(-2 * k));
        assert_eq!(a.range(&in_block([8, 1, 1])), Some((0, i128::from(k))));
        assert!(!offsets_fit(&a, depends_on_lane, &in_block([8, 1, 1])));
        assert!(offsets_fit(&a, depends_on_lane, &in_block([2, 1, 1])));
    }

    #[test]
    fn zero_bytes_need_neither_shape_nor_lane_coefficients() {
        let address = tid().scale(SymExpr::sym("stride"));
        let zero = CountRange { min: 0, max: 0 };
        assert_eq!(
            warp_footprint(&address, 0, 16, [0, 0, 0], |_| panic!("no lanes needed")),
            Ok(Footprint {
                sectors: zero,
                ideal: zero
            })
        );
    }

    #[test]
    fn unknown_shape_or_symbolic_lane_coefficient_is_refused() {
        let k = Affine::var(Var::Tid(Axis::Y)).scale(SymExpr::sym("param_2"));
        assert!(
            warp_footprint(&k, 2, 2, [32, 32, 1], |_| true)
                .unwrap_err()
                .contains("bind")
        );
        assert!(
            warp_footprint(&tid(), 4, 4, [0, 0, 0], |_| true)
                .unwrap_err()
                .contains("--launch")
        );
    }

    /// 128 threads read 4 B each at `p + 4·tid + 512·k` over 4 trips: 2048
    /// requested bytes, all distinct, in 64 sectors when `p` is
    /// sector-aligned and 80 when not. A store to the same addresses
    /// touches no other byte; one at `p + q + …`, on the same pointer
    /// at an unknown offset, may overlap the load: the distinct bytes
    /// are at least the larger base's and at most their sum.
    #[test]
    fn a_loop_requests_and_touches_its_lanes_bytes_at_every_trip() {
        let c = |n: i64| SymExpr::Const(n);
        let id = LoopId(0);
        let form = Affine::invariant(SymExpr::sym("param_0"))
            + tid().scale(c(4))
            + Affine::var(Var::Iter(id)).scale(c(512));
        let access = |form: &Affine| AccessForm {
            form: Some(form.clone()),
            bytes: 4,
            align: 4,
            space: Space::Global,
            set: ThreadSet { pred: Pred::True },
            guard_exact: true,
            conditional: false,
            uniform_varies: false,
        };
        let on_param = |_: &Affine| Some("param_0".to_owned());
        let one = loop_bytes(id, 4, [128, 1, 1], &[access(&form)], on_param).unwrap();
        assert_eq!(one.requested, 2048);
        assert_eq!(
            one.unique,
            Some(ByteRange {
                min: 2048,
                max: 2048
            })
        );
        assert_eq!(
            one.sector_bytes,
            Some(ByteRange {
                min: 2048,
                max: 2560
            })
        );
        let same = [access(&form), access(&form)];
        let same = loop_bytes(id, 4, [128, 1, 1], &same, on_param).unwrap();
        assert_eq!(same.requested, 4096);
        assert_eq!(
            same.unique,
            Some(ByteRange {
                min: 2048,
                max: 2048
            })
        );
        let shifted = form.clone() + Affine::invariant(SymExpr::sym("param_2"));
        let forms = [access(&form), access(&shifted)];
        let two = loop_bytes(id, 4, [128, 1, 1], &forms, on_param).unwrap();
        assert_eq!(
            two.unique,
            Some(ByteRange {
                min: 2048,
                max: 4096
            })
        );
        assert_eq!(loop_bytes(id, 4, [0, 1, 1], &forms, on_param), None);
    }
}
