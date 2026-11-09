/// Structured CUDA graph compile or replay failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphFailureCategory {
    /// Graph contains a node whose output is not on CUDA.
    NonCudaNode,
    /// Graph or replay work targets a different CUDA device than the selected graph device.
    NonSelectedDevice,
    /// Operation data type is unsupported by the CUDA graph path.
    UnsupportedDataType,
    /// Tensor layout is unsupported by the CUDA graph path.
    UnsupportedLayout,
    /// Operation is unsupported by the CUDA graph path.
    UnsupportedOperation,
    /// A planned lowering provider failed and no rejected candidate materialized the node.
    ProviderFailure,
    /// Graph contains symbolic input dimensions that the CUDA graph path cannot record yet.
    DynamicDimensionUnsupported,
    /// Graph has no native CUDA graph nodes when native graph work was required.
    EmptyNativeGraph,
    /// CUDA allocation, graph capture, or native graph construction failed.
    AllocatorCaptureFailure,
    /// Runtime guard validation rejected replay inputs.
    RuntimeGuardFailure,
    /// Error does not fit a more specific CUDA graph category.
    Other,
}

impl GraphFailureCategory {}
