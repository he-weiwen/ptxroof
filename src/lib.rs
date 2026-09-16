//! ptxroof — static roofline analysis for PTX kernels (v2 of
//! nvptx_analyzer).
//!
//! The library exposes parsing, analysis, and report construction.
//! `main.rs` handles arguments, file input, output selection, and errors.

pub mod analysis;
pub mod ptx;
pub mod report;
pub mod support;

// Compatibility paths for existing library consumers.
pub use analysis::instruction_counts::classify;
pub use analysis::scalar::{affine, trace as tracer, trip_counts as trips};
pub use analysis::thread_participation as threads;
pub mod footprint {
    pub use crate::analysis::memory_footprint::*;
    pub use crate::analysis::scalar::lane_eval::{depends_on_lane, eval_lane};
}

/// Tool version, baked in from Cargo.toml at compile time.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_nonempty() {
        assert!(!super::version().is_empty());
    }
}
