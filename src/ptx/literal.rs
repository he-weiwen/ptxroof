//! Integer literal interpretation shared by parsing and analyses.

/// Parse a PTX integer literal: decimal, hex (`0x..`), optionally
/// negative.
pub(crate) fn parse_int(text: &str) -> Option<i64> {
    let (neg, rest) = match text.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, text),
    };
    let val = if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()?
    } else {
        rest.parse::<i64>().ok()?
    };
    Some(if neg { -val } else { val })
}
