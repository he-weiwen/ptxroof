//! Dominance, loop discovery and branch joins over the PTX control-flow
//! graph.

pub mod dominators;
pub mod loops;
pub mod sync_dependence;

pub use dominators::{DominanceInfo, dominators};
pub use loops::{Loop, LoopForest, LoopId, loop_forest};
