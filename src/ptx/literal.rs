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

/// The signed value of the literal's low `width` bits: PTX ISA §4.5.1,
/// "each integer constant is converted to the appropriate size based on
/// the data or instruction type at its use". It reads what `parse_int`
/// reads, and a hexadecimal literal above 2^63 − 1 as its 64 bits:
/// `0xFFFFFFF0` at width 32 is −16.
pub(crate) fn parse_bits(text: &str, width: u32) -> Option<i64> {
    let value = parse_int(text).or_else(|| {
        let hex = text
            .strip_prefix("0x")
            .or_else(|| text.strip_prefix("0X"))?;
        u64::from_str_radix(hex, 16).ok().map(|v| v as i64)
    })?;
    let unused = 64 - width;
    Some(value << unused >> unused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_literal_reads_as_the_signed_value_of_its_low_bits() {
        for (text, width, value) in [
            ("16", 32, Some(16)),
            ("-16", 32, Some(-16)),
            ("0xFFFFFFFF", 32, Some(-1)),
            ("0xFFFFFFF0", 32, Some(-16)),
            ("0xFFFFFFF0", 64, Some(0xFFFF_FFF0)),
            ("0x7FFFFFFF", 32, Some(i64::from(i32::MAX))),
            ("0x8000", 16, Some(-32768)),
            ("0xfffffffffffffff0", 64, Some(-16)),
            ("0xfffffffffffffff0", 32, Some(-16)),
            ("0x8000000000000000", 64, Some(i64::MIN)),
            ("0x1ffffffffffffffff", 64, None),
            ("16U", 32, None),
        ] {
            assert_eq!(parse_bits(text, width), value, "{text} at {width}");
        }
    }
}
