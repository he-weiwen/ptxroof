//! Report construction, output schema, and text rendering.

pub mod build;
pub mod names;
pub mod schema;
pub mod text;

pub use build::{AnalyzeError, AnalyzeOptions, BindingSpec, analyze, parse_bind};
pub use schema::Report;

// Compatibility exports for existing callers.
pub use crate::analysis::instruction_counts::{collect, stats};
pub use collect::{BlockMeasurements, ClassCounts, CountQualifier, collect};
pub use schema as tree;
pub use stats::{Stats, Tally};
