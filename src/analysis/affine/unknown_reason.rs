//! Why the affine analysis has no affine expression for a value.

/// Why an SSA value has no affine expression.
#[derive(Clone, PartialEq)]
pub(crate) enum UnknownReason {
    /// A sentence that reads the same wherever the value is read.
    Other(String),
}

impl UnknownReason {
    /// The reason as a sentence.
    pub(crate) fn render(&self) -> String {
        match self {
            UnknownReason::Other(s) => s.clone(),
        }
    }
}

impl From<String> for UnknownReason {
    fn from(s: String) -> Self {
        UnknownReason::Other(s)
    }
}

impl From<&str> for UnknownReason {
    fn from(s: &str) -> Self {
        UnknownReason::Other(s.to_owned())
    }
}
