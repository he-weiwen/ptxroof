//! What the affine analysis knows of an SSA value at its definition.

use crate::analysis::affine::affine_expr::AffineExpr;

/// An SSA value's affine expression, or why it has none and how many
/// low bits every definition of it leaves 0 (LLVM's
/// `getMinTrailingZeros`).
#[derive(Clone)]
pub(crate) enum AffineProperties {
    Expr(AffineExpr),
    Unknown {
        reason: String,
        min_trailing_zeros: u32,
    },
}

impl AffineProperties {
    /// The expression, or why there is none.
    pub(crate) fn expr(self) -> Result<AffineExpr, String> {
        match self {
            AffineProperties::Expr(a) => Ok(a),
            AffineProperties::Unknown { reason, .. } => Err(reason),
        }
    }
}
