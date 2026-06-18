use crate::{
    data_type::DataType,
    error::Error,
    tensor::{BackendTensorUid, TensorId},
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct BindingReplacement {
    pub(crate) source_id: TensorId,
    pub(crate) target_id: TensorId,
    pub(crate) byte_offset: i64,
}

/// A non-virtual tensor that must be bound by the caller before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredTensor {
    /// Frontend graph tensor ID accepted by [`crate::frontend::bindings::Bindings::set`].
    pub id: TensorId,
    /// Backend tensor UID used in the cuDNN variant pack after lowering.
    pub backend_uid: BackendTensorUid,
    /// Optional frontend tensor name, when one was assigned in the graph.
    pub name: Option<String>,
    /// Tensor data type expected by cuDNN.
    pub data_type: DataType,
    /// Logical tensor dimensions.
    pub dimensions: Vec<i64>,
    /// Logical tensor strides.
    pub strides: Vec<i64>,
}

/// A lowered alias/view tensor whose pointer is derived from another binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasTensorBinding {
    /// Frontend source tensor ID that must be bound by the caller.
    pub source_id: TensorId,
    /// Backend source tensor UID used when materializing the variant pack.
    pub source_backend_uid: BackendTensorUid,
    /// Frontend alias tensor ID.
    pub target_id: TensorId,
    /// Backend alias tensor UID used when materializing the variant pack.
    pub target_backend_uid: BackendTensorUid,
    /// Optional alias tensor name.
    pub target_name: Option<String>,
    /// Byte offset added to the source pointer for the alias.
    pub byte_offset: i64,
}

/// Controls how many times each candidate plan is measured during autotuning.
///
/// Autotuning times candidate engine configurations and chooses the fastest one
/// for a particular problem and device.
#[derive(Debug, Clone, Copy)]
pub struct AutotuneConfig {
    /// Number of measured executions for each candidate plan.
    pub iterations: usize,
    /// Number of unmeasured executions before timing a candidate plan.
    pub warmup: usize,
}

impl AutotuneConfig {
    pub fn new() -> Self {
        Self {
            iterations: 10,
            warmup: 1,
        }
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing result for one candidate execution plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanTiming {
    /// Candidate index before autotune ranking.
    pub original_plan_index: usize,
    /// Candidate index after ranking by measured time.
    pub ranked_plan_index: usize,
    /// Measured elapsed time in milliseconds.
    pub elapsed_ms: f32,
}

/// Result of autotuning a set of built plan candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct AutotuneResult {
    /// Original candidate index selected as the winner.
    pub winner_original_plan_index: usize,
    /// Ranked candidate index selected as the winner.
    pub winner_ranked_plan_index: usize,
    /// Per-plan timing data sorted by measured time.
    pub timings: Vec<PlanTiming>,
}

/// Policy used when turning heuristic candidates into executable plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum BuildPlanPolicy {
    /// Build candidates sequentially and return the first successfully built
    /// plan.
    FirstSupported,
    /// Build candidates in parallel windows and return the first successful
    /// plan in heuristic order from the earliest successful window.
    FirstSupportedParallel { window_size: usize },
    /// Try to build every candidate engine configuration.
    AllSupported,
}

/// Support-check result for one candidate plan.
#[derive(Debug)]
pub struct PlanSupport {
    /// Candidate plan index.
    pub plan_index: usize,
    /// Whether cuDNN reported the candidate as supported.
    pub supported: bool,
    /// Error returned while checking support, if any.
    pub error: Option<Error>,
}
