//! Report construction, output schema, and text rendering.

pub mod build;
pub mod names;
pub mod schema;
pub mod text;

pub use build::{AnalyzeError, AnalyzeOptions, BindingSpec, analyze, parse_bind};
pub use schema::Report;
