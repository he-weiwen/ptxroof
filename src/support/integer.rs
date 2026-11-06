//! Integer arithmetic the analyses share.

/// The greatest common divisor of `a` and `b`; `gcd(0, 0)` is 0.
pub(crate) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The m with `lo` and `hi` both in [origin + m·d, origin + (m + 1)·d),
/// d > 0: the block of d both lie in, counted from `origin`.
pub(crate) fn same_window(lo: i128, hi: i128, origin: i128, d: i128) -> Option<i128> {
    let m = lo.checked_sub(origin)?.div_euclid(d);
    (hi.checked_sub(origin)?.div_euclid(d) == m).then_some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_divides_both() {
        assert_eq!(gcd(32, 12), 4);
        assert_eq!(gcd(12, 32), 4);
        assert_eq!(gcd(0, 7), 7);
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn same_window_counts_blocks_from_the_origin() {
        assert_eq!(same_window(10, 11, 0, 8), Some(1));
        assert_eq!(same_window(7, 8, 0, 8), None);
        assert_eq!(same_window(-2, -1, 0, 8), Some(-1));
        assert_eq!(same_window(-2, 1, -128, 256), Some(0));
        assert_eq!(same_window(i128::MIN, 0, 1, 8), None);
    }
}
