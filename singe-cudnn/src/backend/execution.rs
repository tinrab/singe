pub mod attention;
/// Native operations for expanding and contracting band-matrix representations.
pub mod band_matrix;
/// Block-scale quantization and dequantization operands, attributes, and native operations.
pub mod block_scale;
/// Native convolution descriptors and forward, data-gradient, and filter-gradient operations.
pub mod convolution;
mod engine_config;
mod execution_plan;
mod intermediate;
mod kernel_cache;
/// Native matrix-multiplication descriptors, optional dimension operands, and cache keys.
pub mod matmul;
/// Native mixture-of-experts grouped matrix multiplication and routing operations.
pub mod moe;
pub mod normalization;
mod operation_graph;
/// Native paged-cache loading with container, page-table, and sequence operands.
pub mod paged_cache;
mod plan_cache;
/// Native pointwise descriptors and operations with mode-dependent operands.
pub mod pointwise;
/// Native reduction descriptors and tensor operations.
pub mod reduction;
/// Native resampling geometry, padding, and forward and backward operations.
pub mod resample;
/// Native random-number distributions, seed operands, and generation operations.
pub mod rng;
/// Native rotary position embedding operands and forward and backward operations.
pub mod rope;
pub mod tensor_ops;
mod variant_pack;
mod warmup;

#[cfg(all(test, feature = "testing"))]
mod tests;

pub(crate) use execution_plan::RetainedExecution;
pub(crate) use variant_pack::WorkspaceInitialization;
pub(crate) use warmup::capture_plan_warmup;

pub use crate::backend::execution::{
    engine_config::{
        EngineChoice, EngineConfig, EngineHeuristics, EngineHeuristicsResult, EngineKnobInfo,
        EnginePlanningConfig, HeuristicQueryStatus, KnobChoiceInfo, LayoutInfo,
    },
    execution_plan::{ExecutionPlan, ExecutionPlanTarget},
    intermediate::{IntermediateId, IntermediateInfo},
    kernel_cache::{KernelCache, KernelCacheJson},
    operation_graph::{OperationGraph, OperationGraphConfig},
    plan_cache::{ExecutionPlanCache, PlanCacheAdmission},
    variant_pack::{VariantPack, VariantPackOverride},
    warmup::WarmupOutcome,
};
