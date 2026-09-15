//! ptxroof — static roofline analysis for PTX kernels (v2 of
//! nvptx_analyzer).
//!
//! The library/binary
//! split is architectural: everything analyzable lives here behind a
//! `Result`-returning API; `main.rs` only parses arguments and renders
//! errors.

pub mod affine;
pub mod cfg;
pub mod classify;
pub mod core;
pub mod footprint;
pub mod parse;
pub mod report;
pub mod tracer;
pub mod trips;

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
