//! Dominance and loop discovery over the PTX control-flow graph.

pub mod dominators;
pub mod loops;

pub use dominators::{DominanceInfo, dominators};
pub use loops::{Loop, LoopForest, LoopId, loop_forest};
