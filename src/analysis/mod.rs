//! Analyses over the parsed PTX representation.

pub mod control_flow;
pub mod dead_code;
pub mod instruction_counts;
pub(crate) mod kernel_data;
pub mod loop_names;
pub mod memory_footprint;
pub mod register_effects;
pub mod scalar;
pub(crate) mod special_registers;
pub mod thread_participation;
pub mod uniformity;
