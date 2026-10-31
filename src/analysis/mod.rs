//! Analyses over the parsed PTX representation.

pub mod affine;
pub mod control_flow;
pub mod dead_code;
pub(crate) mod execution_counts;
pub mod instruction_counts;
pub(crate) mod kernel_analyses;
pub mod loop_names;
pub mod memory_footprint;
pub mod register_effects;
pub(crate) mod special_registers;
pub mod thread_participation;
pub mod uniformity_info;
