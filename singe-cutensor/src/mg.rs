//! cuTENSORMg wrappers.

/// Multi-GPU handle creation, participating devices, and context ownership.
pub mod context;
/// Distributed contraction descriptors, kernel selection, and execution plans.
pub mod contraction;
/// Multi-GPU tensor-copy descriptors and execution plans.
pub mod copy;
/// Distributed device allocations and owned multi-GPU workspace storage.
pub mod memory;
/// Tensor mode metadata, device partitions, and distributed tensor descriptors.
pub mod tensor;
/// Typed native multi-GPU algorithm selectors.
pub mod types;
/// Per-device and host workspace requirements for multi-GPU plans.
pub mod workspace;
