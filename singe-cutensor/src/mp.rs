//! cuTENSORMp wrappers.

/// Multi-process handles, CUDA contexts, and NCCL communicator lifetimes.
pub mod context;
/// Workspace allocations retained with their multi-process context.
pub mod memory;
/// Multi-process operation descriptors, plan preferences, execution, and workspace requirements.
pub mod plan;
/// Tensor mode metadata, rank partitions, and local shard layout queries.
pub mod tensor;
/// Typed native multi-process algorithm and plan-attribute selectors.
pub mod types;
