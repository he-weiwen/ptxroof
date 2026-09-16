//! What one warp's execution of a memory instruction touches: the
//! 32-byte sectors and 128-byte lines its 32 lanes' addresses fall in,
//! from the affine address, the element width and the block shape.
//! The lane-dependent part is evaluated for every lane; the uniform
//! part (CTA index, loop counters, parameters) only shifts the whole
//! set, so every alignment of it that respects the element size is
//! tried and the count is reported as a range.

#[cfg(test)]
use crate::analysis::scalar::affine::Axis;
use crate::analysis::scalar::affine::{Affine, Var};
use crate::analysis::scalar::lane_eval::{depends_on_lane, eval};
use serde::Serialize;
use std::collections::BTreeSet;

pub const SECTOR: i64 = 32;
pub const LINE: i64 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CountRange {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Footprint {
    pub sectors: CountRange,
    pub lines: CountRange,
}

/// Sectors and lines per warp request, or why they cannot be counted:
/// a lane coefficient that is not a constant (bind the parameter), or
/// a block shape that is not known (`.reqntid` or `--launch`).
pub fn warp_footprint(a: &Affine, bytes: u32, shape: [u32; 3]) -> Result<Footprint, String> {
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
    let count = |unit: i64| -> CountRange {
        let (mut min, mut max) = (u32::MAX, 0u32);
        for warp in 0..(threads + 31) / 32 {
            let offsets: Vec<i64> = (warp * 32..((warp + 1) * 32).min(threads))
                .map(|t| {
                    let tid = [t % nx, (t / nx) % ny, t / (nx * ny)];
                    lane_terms.iter().map(|(v, k)| k * eval(v, tid)).sum()
                })
                .collect();
            let mut shift = 0;
            while shift < unit {
                let touched: BTreeSet<i64> = offsets
                    .iter()
                    .flat_map(|&o| {
                        let lo = (o + shift).div_euclid(unit);
                        let hi = (o + shift + bytes - 1).div_euclid(unit);
                        lo..=hi
                    })
                    .collect();
                let n = touched.len() as u32;
                min = min.min(n);
                max = max.max(n);
                shift += bytes;
            }
        }
        CountRange { min, max }
    };
    Ok(Footprint {
        sectors: count(SECTOR),
        lines: count(LINE),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scalar::symexpr::SymExpr;

    fn tid() -> Affine {
        Affine::var(Var::Tid(Axis::X))
    }

    fn fp(a: &Affine, bytes: u32, shape: [u32; 3]) -> (u32, u32, u32, u32) {
        let f = warp_footprint(a, bytes, shape).unwrap();
        (f.sectors.min, f.sectors.max, f.lines.min, f.lines.max)
    }

    #[test]
    fn coalesced_strided_and_broadcast_warps() {
        let c = |n: i64| SymExpr::Const(n);
        // 32 lanes × 4 B contiguous: four sectors, one line when aligned,
        // one more of each when the base sits inside a sector.
        assert_eq!(fp(&tid().scale(c(4)), 4, [128, 1, 1]), (4, 5, 1, 2));
        // 2 B elements: two sectors.
        assert_eq!(fp(&tid().scale(c(2)), 2, [64, 1, 1]), (2, 3, 1, 2));
        // A 128 B stride per lane: every lane its own line.
        assert_eq!(fp(&tid().scale(c(128)), 4, [32, 1, 1]), (32, 32, 32, 32));
        // The same address for every lane.
        assert_eq!(
            fp(&Affine::invariant(SymExpr::sym("p")), 4, [32, 1, 1]),
            (1, 1, 1, 1)
        );
        // k5's A tile: 4 rows of 8 halves per warp, rows 512 B apart.
        let row = Affine::var(Var::Div(Box::new(Var::Tid(Axis::X)), 8)).scale(c(512));
        let col = Affine::var(Var::Mod(Box::new(Var::Tid(Axis::X)), 8)).scale(c(2));
        assert_eq!(fp(&(row + col), 2, [64, 1, 1]), (4, 8, 4, 8));
        // A 2D block: a warp is two rows of 16 lanes, rows 64 B apart.
        let a = tid().scale(c(4)) + Affine::var(Var::Tid(Axis::Y)).scale(c(64));
        assert_eq!(fp(&a, 4, [16, 16, 1]), (4, 5, 1, 2));
    }

    #[test]
    fn unknown_shape_or_symbolic_lane_coefficient_is_refused() {
        let k = Affine::var(Var::Tid(Axis::Y)).scale(SymExpr::sym("param_2"));
        assert!(
            warp_footprint(&k, 2, [32, 32, 1])
                .unwrap_err()
                .contains("bind")
        );
        assert!(
            warp_footprint(&tid(), 4, [0, 0, 0])
                .unwrap_err()
                .contains("--launch")
        );
    }
}
