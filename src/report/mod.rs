//! Report construction, output schema, and text rendering.

pub mod annotate;
pub mod bindings;
pub mod build;
pub mod measured;
pub mod names;
pub mod schema;
pub mod text;

pub use bindings::{BindingSpec, parse_bind};
pub use build::{AnalyzeError, AnalyzeOptions, analyze};
pub use schema::Report;
