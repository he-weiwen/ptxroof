//! Report construction, the annotated PTX, and the ncu comparisons.

pub mod annotate;
pub mod bindings;
pub mod build;
pub mod measured;
pub mod names;
pub mod schema;

pub use bindings::{BindingSpec, parse_bind};
pub use build::{AnalyzeError, AnalyzeOptions, analyze};
pub use schema::Report;
