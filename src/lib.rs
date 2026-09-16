//! ptxroof — static roofline analysis for PTX kernels (v2 of
//! nvptx_analyzer).
//!
//! The library exposes parsing, analysis, and report construction.
//! `main.rs` handles arguments, file input, output selection, and errors.

pub mod analysis;
pub mod ptx;
pub mod report;
pub mod support;

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
