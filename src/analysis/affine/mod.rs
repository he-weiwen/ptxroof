//! Symbolic and affine scalar reasoning, definition tracing, and trip counts.

pub mod affine_expr;
pub(crate) mod evolution;
pub(crate) mod int_type;
pub mod lane_eval;
pub(crate) mod loop_bounds;
pub mod recurrences;
pub mod sym_expr;
pub mod trip_counts;
