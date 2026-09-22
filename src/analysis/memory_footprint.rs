//! What one warp's execution of a memory instruction touches: the
//! 32-byte sectors the addresses of its executing lanes fall in, from the affine address, the element width, the
//! block shape and the threads that execute the instruction.
//! The lane-dependent part is evaluated for every lane; the uniform
//! part (CTA index, loop counters, parameters) only shifts the whole
//! set, so every alignment of it under which the lanes' accesses are
//! aligned is tried and the count is reported as a range.

#[cfg(test)]
use crate::analysis::scalar::affine::Axis;
use crate::analysis::scalar::affine::{Affine, Var};
use crate::analysis::scalar::lane_eval::{depends_on_lane, eval};
use serde::Serialize;
use std::collections::BTreeSet;

pub const SECTOR: i64 = 32;

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
    let bytes = i64::from(bytes.max(1));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scalar::symexpr::SymExpr;

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
}
