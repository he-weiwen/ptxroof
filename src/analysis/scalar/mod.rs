//! Symbolic and affine scalar reasoning, definition tracing, and trip counts.

pub mod affine;
pub(crate) mod evolution;
pub(crate) mod integer;
pub mod lane_eval;
pub(crate) mod loop_bounds;
pub mod recurrences;
pub mod symexpr;
pub mod trip_counts;
