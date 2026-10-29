use std::{collections::BTreeMap, num::NonZeroUsize};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    backend::{heuristic::BackendHeuristicMode, tensor::TensorId},
    error::Result,
    frontend::{
        graph::{Graph, RuntimeShapeConstraints},
        plan::{EngineConfigFilter, EngineConfigMetadata, PlanExclusion, PlanId},
    },
};

/// Execution-time shape policy, independent of the optional structural graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeShapePolicy {
    pub overrides_enabled: bool,
    pub bounds: BTreeMap<TensorId, RuntimeShapeConstraints>,
}

impl From<&Graph> for RuntimeShapePolicy {
    fn from(graph: &Graph) -> Self {
        Self {
            overrides_enabled: graph.dynamic_shape_enabled() || graph.override_shape_enabled(),
            bounds: graph.runtime_shape_constraints.clone(),
        }
    }
}

/// Policy used when turning heuristic candidates into executable plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BuildPlanPolicy {
    /// Build candidates sequentially and return the first successfully built
    /// plan.
    FirstSupported,
    /// Build candidates in parallel windows and return the first successful
    /// plan in collection order from the earliest successful window.
    FirstSupportedParallel {
        /// Maximum number of candidates built concurrently within each ordered search window.
        window_size: NonZeroUsize,
    },
    /// Builds candidates in collection order, subject to the configured plan-count limit.
    AllSupported,
    /// Builds candidates with at most `worker_count` workers, subject to the plan-count limit.
    AllSupportedParallel {
        /// Maximum number of concurrent build workers.
        worker_count: NonZeroUsize,
    },
}

/// Configuration eligibility without native plan compilation, matching C++ support queries.
/// Borrows already queried requirements and preserves the candidate's stable identity.
/// Compilation outcomes are reported separately by [`crate::frontend::plan::PlanBuildReport`].
#[derive(Debug, Clone)]
pub struct PlanSupport<'a> {
    /// Borrowed stable candidate identity.
    pub plan: &'a PlanId,
    /// Cached engine requirements and notes used for eligibility checks.
    pub metadata: &'a EngineConfigMetadata,
    /// First filter exclusion, or `None` when cached metadata is eligible.
    pub exclusion: Option<PlanExclusion>,
}

impl PlanSupport<'_> {
    /// Returns whether cached requirements pass the filter; this does not compile or execute a plan.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.exclusion.is_none()
    }
}

/// Native heuristic family requested during candidate discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HeuristicMode {
    /// Native instant decision-tree heuristic with low CPU overhead.
    Instant,
    /// Native mode A decision-tree heuristic.
    A,
    /// Native mode B, which may use neural-network ranking and more CPU time before falling back to instant heuristics.
    B,
    /// Native fallback engines prioritizing functional coverage rather than optimal performance.
    Fallback,
}

impl From<HeuristicMode> for BackendHeuristicMode {
    fn from(mode: HeuristicMode) -> Self {
        match mode {
            HeuristicMode::Instant => Self::Instant,
            HeuristicMode::A => Self::A,
            HeuristicMode::B => Self::B,
            HeuristicMode::Fallback => Self::Fallback,
        }
    }
}

/// Candidate-discovery modes, build policy, resource targets, and cached-metadata eligibility filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileConfig {
    heuristic_modes: Vec<HeuristicMode>,
    build_policy: BuildPlanPolicy,
    #[serde(default)]
    max_plans: Option<NonZeroUsize>,
    sm_count_target: Option<u32>,
    shared_memory_limit: Option<u32>,
    #[serde(default)]
    filter: EngineConfigFilter,
}

impl CompileConfig {
    /// Creates instant-heuristic discovery with first-supported building, no plan-count cap or resource overrides, and an empty eligibility filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heuristic_modes: vec![HeuristicMode::Instant],
            build_policy: BuildPlanPolicy::FirstSupported,
            max_plans: None,
            sm_count_target: None,
            shared_memory_limit: None,
            filter: EngineConfigFilter::new(),
        }
    }

    /// Default configuration with the requested heuristic modes.
    pub fn from_heuristic_modes(heuristic_modes: impl Into<Vec<HeuristicMode>>) -> Self {
        Self::new().with_heuristic_modes(heuristic_modes)
    }

    /// Replaces heuristic query modes, preserving caller order.
    #[must_use]
    pub fn with_heuristic_modes(mut self, heuristic_modes: impl Into<Vec<HeuristicMode>>) -> Self {
        self.heuristic_modes = heuristic_modes.into();
        self
    }

    /// Replaces the candidate-build scheduling and selection policy.
    #[must_use]
    pub fn with_build_policy(mut self, build_policy: BuildPlanPolicy) -> Self {
        self.build_policy = build_policy;
        self
    }

    /// Caps the successful plans admitted by batch compilation, in candidate collection order.
    /// Failed and excluded candidates do not consume this limit.
    /// First-supported policies still select one plan.
    /// Parallel workers can finish additional plans in their last window; those plans
    /// remain in the build collection, excluded from automatic selection.
    /// Explicit incremental builds can select them without recompilation.
    /// `None` removes the cap without changing the build policy.
    #[must_use]
    pub fn with_max_plans(mut self, max_plans: impl Into<Option<NonZeroUsize>>) -> Self {
        self.max_plans = max_plans.into();
        self
    }

    /// Targets this many SMs during discovery and candidate construction.
    /// Zero uses every SM; `None` leaves the native target unset.
    /// The target overrides the graph default and survives cached and parallel builds.
    /// `None` restores inheritance from the graph or retained candidate.
    #[must_use]
    pub fn with_sm_count_target(mut self, sm_count_target: impl Into<Option<u32>>) -> Self {
        self.sm_count_target = sm_count_target.into();
        self
    }

    /// Sets a dynamic shared-memory budget in bytes.
    /// Zero uses the device limit.
    /// Runtime-fusion engines use this budget when resolving pipeline stages;
    /// other engines ignore it.
    /// Supplying a new budget at build time reconfigures the candidate before its
    /// resource requirements and filters are applied, including incremental builds.
    /// `None` removes the override, retaining an existing candidate's budget during reconstruction.
    #[must_use]
    pub fn with_shared_memory_limit(mut self, bytes: impl Into<Option<u32>>) -> Self {
        self.shared_memory_limit = bytes.into();
        self
    }

    /// Returns the dynamic shared-memory budget passed to engine planning.
    #[must_use]
    pub fn shared_memory_limit(&self) -> Option<u32> {
        self.shared_memory_limit
    }

    /// Replaces the engine eligibility filter applied to every candidate built with this configuration.
    #[must_use]
    pub fn with_filter(mut self, filter: EngineConfigFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Borrows the cached-metadata eligibility filter applied during builds.
    #[must_use]
    pub fn filter(&self) -> &EngineConfigFilter {
        &self.filter
    }

    /// Borrows heuristic query modes in requested order.
    #[must_use]
    pub fn heuristic_modes(&self) -> &[HeuristicMode] {
        &self.heuristic_modes
    }

    /// Returns the candidate-build scheduling and selection policy.
    #[must_use]
    pub fn build_policy(&self) -> BuildPlanPolicy {
        self.build_policy
    }

    /// Optional batch selection limit, independent of autotuning's measurement limit.
    #[must_use]
    pub fn max_plans(&self) -> Option<NonZeroUsize> {
        self.max_plans
    }

    /// Returns the SM-count override, or `None` to inherit the graph or retained candidate target.
    #[must_use]
    pub fn sm_count_target(&self) -> Option<u32> {
        self.sm_count_target
    }

    pub(crate) fn cache_key_json(&self) -> Result<Value> {
        // Filter lists are sets for eligibility; retain caller order outside the cache projection for diagnostics.
        let canonical = Self {
            filter: self.filter.canonical(),
            ..self.clone()
        };
        Ok(serde_json::to_value(canonical)?)
    }
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self::new()
    }
}
