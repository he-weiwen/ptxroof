//! ptxroof — static roofline analysis for PTX kernels (v2 of
//! nvptx_analyzer).
//!
//! The library exposes parsing, analysis, and report construction.
//! `main.rs` handles arguments, file input, output selection, and errors.

pub mod analysis;
pub mod ptx;
pub mod report;
pub mod ssa;
pub mod support;
