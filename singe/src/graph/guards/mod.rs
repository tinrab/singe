mod check;
mod create;
mod failure;
mod input;
pub(crate) mod storage;
mod validation;

use std::collections::HashSet;

pub use self::failure::GraphGuardFailure;

#[cfg(test)]
use crate::graph::ValueId;
use crate::{
    data_type::DataType, device::Device, graph::InputDimensionGuard, layout::Layout, tensor::Tensor,
};

/// Runtime invariants required by a compiled graph execution order.
#[derive(Debug, Clone)]
pub struct GraphGuards {
    pub(crate) inputs: Vec<InputGuard>,
    #[cfg(test)]
    pub(crate) constants: Vec<ConstantGuard>,
    pub(crate) input_names: HashSet<String>,
}

/// Metadata required for a reachable graph input.
#[derive(Debug, Clone)]
pub struct InputGuard {
    pub(crate) name: String,
    pub(crate) data_type: DataType,
    pub(crate) device: Device,
    pub(crate) layout: Layout,
    pub(crate) initializer: Option<Tensor>,
    #[cfg(test)]
    pub(crate) storage_span_bytes: usize,
    pub(crate) is_static_initializer: bool,
    pub(crate) static_storage: Option<StaticInputStorageGuard>,
    pub(crate) dimension_guards: Vec<InputDimensionGuard>,
}

/// Process-local storage identity guard for a static graph input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticInputStorageGuard {
    /// Storage owner identity.
    pub(crate) storage_identity: usize,
    /// Allocation data pointer identity, when available.
    pub(crate) data_pointer_identity: Option<usize>,
    /// Tensor storage element offset.
    pub(crate) offset: usize,
    /// Backing storage allocation byte size.
    pub(crate) storage_byte_size: usize,
    /// Storage write version when the guard was taken.
    pub(crate) version: u64,
}

/// Metadata required for a reachable graph constant.
#[derive(Debug, Clone)]
#[cfg(test)]
pub struct ConstantGuard {
    pub(crate) value: ValueId,
    pub(crate) data_type: DataType,
    pub(crate) layout: Layout,
    pub(crate) kind: ConstantGuardKind,
}

#[derive(Debug, Clone)]
#[cfg(test)]
pub enum ConstantGuardKind {
    HostContent { digest: u64 },
    GraphLocal { device: Device },
}

#[cfg(test)]
mod tests;
