//! Symbolic and affine scalar reasoning, definition tracing, and trip counts.

pub mod affine_expr;
pub(crate) mod exit_limit;
pub mod induction_variables;
pub(crate) mod int_type;
pub mod lane_eval;
pub(crate) mod scalar_evolution;
pub mod sym_expr;
pub mod trip_counts;
