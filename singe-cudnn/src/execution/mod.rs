pub mod advanced;
pub mod convolution;
mod engine_config;
mod execution_plan;
mod kernel_cache;
pub mod normalization;
mod operation_graph;
pub mod pointwise;
pub mod reduction;
pub mod resample;
pub mod rng;
pub mod tensor_ops;
mod variant_pack;

#[cfg(all(test, feature = "testing"))]
mod tests;

// TODO: refactor
pub use self::{
    engine_config::{
        EngineConfig, EngineHeuristics, EngineKnobInfo, IntermediateInfo, KnobChoiceInfo,
        LayoutInfo,
    },
    execution_plan::ExecutionPlan,
    kernel_cache::KernelCache,
    operation_graph::OperationGraph,
    variant_pack::{VariantPack, VariantPackOverride},
};

use std::sync::Arc;

use singe_cuda::context::Context as CudaContext;

use crate::error::{Error, Result};

pub(crate) fn validate_context_pair(
    expected: Option<&Arc<CudaContext>>,
    actual: Option<&Arc<CudaContext>>,
    name: &str,
) -> Result<()> {
    match (expected, actual) {
        (Some(expected), Some(actual)) if expected.as_ref() != actual.as_ref() => {
            Err(Error::ContextMismatch { name: name.into() })
        }
        _ => Ok(()),
    }
}
