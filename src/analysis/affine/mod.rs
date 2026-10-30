//! Symbolic and affine scalar reasoning, definition tracing, and trip counts.

pub(crate) mod affine_analysis;
pub mod affine_expr;
pub(crate) mod affine_properties;
pub(crate) mod exit_limit;
pub mod induction_variables;
pub(crate) mod int_type;
pub mod lane_eval;
pub mod sym_expr;
pub mod trip_counts;
pub(crate) mod unknown_reason;
