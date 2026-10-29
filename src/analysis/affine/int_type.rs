//! Integer type widths and representable ranges used by scalar and predicate analysis.

/// An integer type of 8 to 64 bits: unsigned `.u`, signed `.s`, or a
/// `.b` bit pattern (PTX ISA §5.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntType {
    pub(crate) kind: IntKind,
    pub(crate) width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntKind {
    Unsigned,
    Signed,
    Bits,
}

impl IntType {
    /// The type a modifier names, `u32`.
    pub(crate) fn from_name(m: &str) -> Option<IntType> {
        let (kind, width) = m.split_at_checked(1)?;
        let width = width
            .parse()
            .ok()
            .filter(|w| matches!(w, 8 | 16 | 32 | 64))?;
        let kind = match kind {
            "u" => IntKind::Unsigned,
            "s" => IntKind::Signed,
            "b" => IntKind::Bits,
            _ => return None,
        };
        Some(IntType { kind, width })
    }

    /// The first integer type among an instruction's modifiers.
    pub(crate) fn of(mods: &[&str]) -> Option<IntType> {
        mods.iter().find_map(|m| IntType::from_name(m))
    }

    /// The values the type reads, [lo, hi): a bit pattern as either sign.
    pub(crate) fn values(self) -> (i128, i128) {
        let (half, full) = (1i128 << (self.width - 1), 1i128 << self.width);
        match self.kind {
            IntKind::Unsigned => (0, full),
            IntKind::Signed => (-half, half),
            IntKind::Bits => (-half, full),
        }
    }

    /// The values it reads that are not negative.
    pub(crate) fn nonnegative(self) -> (i128, i128) {
        (0, self.values().1)
    }
}
