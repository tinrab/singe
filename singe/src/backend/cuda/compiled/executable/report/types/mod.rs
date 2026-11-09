mod alias;
mod operation;
mod partition;
mod provider;
mod report;

pub use self::{
    alias::GraphCompileAliasGroupReport,
    operation::GraphCompileOperationReport,
    partition::{GraphCompileIrPartitionFields, GraphCompilePartitionReport},
    provider::{
        GraphCompileProviderCandidate, GraphCompileProviderDecision, GraphCompileProviderFailure,
        GraphCompileProviderRejection,
    },
    report::GraphCompileReport,
};
use crate::backend::registry::LayoutSupport;

/// How an operation output is represented by CUDA graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCompileValueMaterialization {
    /// Runtime input placeholder owned by the compiled graph.
    RuntimeInput,
    /// Static input backed by a graph initializer tensor.
    StaticInput,
    /// Constant tensor embedded in the compiled graph.
    Constant,
    /// Metadata-only alias/view output.
    Alias,
    /// `CudaGraphValue` materialized by a native CUDA graph provider operation.
    NativeGraphOutput,
    /// Operation had no realized CUDA graph output.
    None,
}

/// Input layout required by a CUDA graph operation or provider partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphCompileInputLayoutRequirement {
    /// Inputs may keep any valid strided layout.
    AnyStrided,
    /// Inputs must be contiguous at offset zero before provider lowering.
    Contiguous,
}

pub fn input_layout_requirement_for_support(
    layout_support: LayoutSupport,
) -> GraphCompileInputLayoutRequirement {
    match layout_support {
        LayoutSupport::AnyStrided => GraphCompileInputLayoutRequirement::AnyStrided,
        LayoutSupport::MaterializesToContiguous | LayoutSupport::RequiresContiguous => {
            GraphCompileInputLayoutRequirement::Contiguous
        }
    }
}
