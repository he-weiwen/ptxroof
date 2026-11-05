//! A block's threads by linear index and its warps.

use std::ops::Range;

/// The linear index of thread `tid` in a block of `shape`, `%tid.x`
/// varying fastest.
pub(in crate::analysis) fn linear(tid: [i64; 3], shape: [u32; 3]) -> i64 {
    let [nx, ny, _] = shape.map(i64::from);
    tid[0] + nx * (tid[1] + ny * tid[2])
}

/// The thread of linear index `t` in a block of `shape`.
pub(in crate::analysis) fn thread(t: i64, shape: [u32; 3]) -> [i64; 3] {
    let [nx, ny, _] = shape.map(i64::from);
    [t % nx, (t / nx) % ny, t / (nx * ny)]
}

/// The warps of a block of `shape` as ranges of linear thread indices:
/// PTX ISA §3.1, "each warp contains threads of consecutive, increasing
/// thread IDs with the first warp containing thread 0".
pub(in crate::analysis) fn warps_of(shape: [u32; 3]) -> impl Iterator<Item = Range<i64>> {
    let threads: i64 = shape.iter().map(|&n| i64::from(n)).product();
    (0..threads)
        .step_by(32)
        .map(move |base| base..(base + 32).min(threads))
}
