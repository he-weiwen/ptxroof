//! A CTA's threads by thread ID and its warps (PTX ISA §2.2, Thread
//! Hierarchy).

use std::ops::Range;

/// The threads of a warp: PTX ISA §2.2.1, "Typically, a warp has 32
/// threads", which PTX names `WARP_SZ`.
pub(crate) const WARP_SZ: i64 = 32;

/// A `shfl.sync` control c that reads the whole warp: a clamp value
/// c[4:0] of 31 and a segment mask c[12:8] of 0 (PTX ISA §9.7.9.6:
/// "cval[4:0] = c[4:0]; // clamp value", "segmask[4:0] = c[12:8];").
pub(crate) fn whole_warp_shuffle(c: i64) -> bool {
    c & 0x1f == 0x1f && (c >> 8) & 0x1f == 0
}

/// The threads of a block of `shape`.
pub(crate) fn threads(shape: [u32; 3]) -> i64 {
    shape.iter().map(|&n| i64::from(n)).product()
}

/// The CTAs of a grid of `grid`; `None` past u64.
pub(crate) fn ctas(grid: [u32; 3]) -> Option<u64> {
    grid.iter().try_fold(1u64, |n, &d| n.checked_mul(d.into()))
}

/// The thread ID of `tid` in a block of `shape`, `%tid.x` varying
/// fastest: PTX ISA §9.7.16.1, "warp-rank of a warp is defined as:
/// (%tid.x + %tid.y * %ntid.x + %tid.z * %ntid.x * %ntid.y) / 32".
pub(crate) fn to_1d(tid: [i64; 3], shape: [u32; 3]) -> i64 {
    let [nx, ny, _] = shape.map(i64::from);
    tid[0] + nx * (tid[1] + ny * tid[2])
}

/// The thread of thread ID `t` in a block of `shape`.
pub(crate) fn to_3d(t: i64, shape: [u32; 3]) -> [i64; 3] {
    let [nx, ny, _] = shape.map(i64::from);
    [t % nx, (t / nx) % ny, t / (nx * ny)]
}

/// The warps of a block of `shape` as ranges of thread IDs: PTX ISA
/// §3.1, "each warp contains threads of consecutive, increasing thread
/// IDs with the first warp containing thread 0".
pub(crate) fn warps_of(shape: [u32; 3]) -> impl Iterator<Item = Range<i64>> {
    let threads = threads(shape);
    (0..threads)
        .step_by(WARP_SZ as usize)
        .map(move |base| base..(base + WARP_SZ).min(threads))
}
