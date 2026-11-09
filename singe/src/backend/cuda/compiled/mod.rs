//! Compilation of captured graphs into native CUDA graphs, in three stages.
//!
//! `plan` chooses a provider for each operation and plans memory and cache keys, `build` builds the native graph from the plan, and `executable` holds the compiled graph that replays.

/// Compiled CUDA graphs, retained resources, and replay statistics.
pub mod executable;

pub mod build;

/// CUDA graph lowering plans, provider compilation, and memory planning.
pub mod plan;

#[cfg(all(test, feature = "testing"))]
mod tests;
