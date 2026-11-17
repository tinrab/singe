//! Finalized operation graphs, execution plans, tensor bindings, and engine selection.

mod engine_config;
mod execution_plan;
mod intermediate;
mod kernel_cache;
mod operation_graph;
mod plan_cache;
mod variant_pack;
mod warmup;

pub(crate) use self::{
    execution_plan::RetainedExecution, variant_pack::WorkspaceInitialization,
    warmup::capture_plan_warmup,
};
pub use crate::backend::execution::{
    engine_config::{
        EngineChoice, EngineConfig, EngineHeuristics, EngineHeuristicsResult, EngineKnobInfo,
        EnginePlanningOptions, HeuristicQueryStatus, KnobChoiceInfo, LayoutInfo,
    },
    execution_plan::{ExecutionPlan, ExecutionPlanTarget},
    intermediate::{IntermediateId, IntermediateInfo},
    kernel_cache::{KernelCache, KernelCacheJson},
    operation_graph::{OperationGraph, OperationGraphOptions},
    plan_cache::{ExecutionPlanCache, PlanCacheAdmission},
    variant_pack::{VariantPack, VariantPackOverride},
    warmup::WarmupOutcome,
};

#[cfg(all(test, feature = "testing"))]
mod tests;
