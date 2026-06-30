use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    error::Result,
    frontend::plan::BuildPlanPolicy,
    heuristic::BackendHeuristicMode,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HeuristicMode {
    Instant,
    A,
    B,
    Fallback,
}

impl HeuristicMode {
    pub fn backend_mode(self) -> BackendHeuristicMode {
        match self {
            HeuristicMode::Instant => BackendHeuristicMode::Instant,
            HeuristicMode::A => BackendHeuristicMode::A,
            HeuristicMode::B => BackendHeuristicMode::B,
            HeuristicMode::Fallback => BackendHeuristicMode::Fallback,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileConfig {
    heuristic_modes: Vec<HeuristicMode>,
    build_policy: BuildPlanPolicy,
    sm_count_target: Option<i64>,
    max_workspace_size: Option<usize>,
    max_shared_memory_size: Option<usize>,
    excluded_engine_name_substrings: Vec<String>,
    include_numerical_notes: Vec<BackendNumericalNote>,
    exclude_numerical_notes: Vec<BackendNumericalNote>,
    include_behavior_notes: Vec<BackendBehaviorNote>,
    exclude_behavior_notes: Vec<BackendBehaviorNote>,
}

impl CompileConfig {
    pub fn new() -> Self {
        Self {
            heuristic_modes: vec![HeuristicMode::Instant],
            build_policy: BuildPlanPolicy::FirstSupported,
            sm_count_target: None,
            max_workspace_size: None,
            max_shared_memory_size: None,
            excluded_engine_name_substrings: Vec::new(),
            include_numerical_notes: Vec::new(),
            exclude_numerical_notes: Vec::new(),
            include_behavior_notes: Vec::new(),
            exclude_behavior_notes: Vec::new(),
        }
    }

    pub fn with_heuristic_modes(mut self, heuristic_modes: impl Into<Vec<HeuristicMode>>) -> Self {
        self.heuristic_modes = heuristic_modes.into();
        self
    }

    pub fn with_build_policy(mut self, build_policy: BuildPlanPolicy) -> Self {
        self.build_policy = build_policy;
        self
    }

    pub fn with_sm_count_target(mut self, sm_count_target: i64) -> Self {
        self.sm_count_target = Some(sm_count_target);
        self
    }

    pub fn with_max_workspace_size(mut self, max_workspace_size: usize) -> Self {
        self.max_workspace_size = Some(max_workspace_size);
        self
    }

    pub fn with_max_shared_memory_size(mut self, max_shared_memory_size: usize) -> Self {
        self.max_shared_memory_size = Some(max_shared_memory_size);
        self
    }

    pub fn with_excluded_engine_name_substrings(
        mut self,
        excluded_engine_name_substrings: impl Into<Vec<String>>,
    ) -> Self {
        self.excluded_engine_name_substrings = excluded_engine_name_substrings.into();
        self
    }

    pub fn with_included_numerical_notes(
        mut self,
        notes: impl Into<Vec<BackendNumericalNote>>,
    ) -> Self {
        self.include_numerical_notes = notes.into();
        self
    }

    pub fn with_excluded_numerical_notes(
        mut self,
        notes: impl Into<Vec<BackendNumericalNote>>,
    ) -> Self {
        self.exclude_numerical_notes = notes.into();
        self
    }

    pub fn with_included_behavior_notes(
        mut self,
        notes: impl Into<Vec<BackendBehaviorNote>>,
    ) -> Self {
        self.include_behavior_notes = notes.into();
        self
    }

    pub fn with_excluded_behavior_notes(
        mut self,
        notes: impl Into<Vec<BackendBehaviorNote>>,
    ) -> Self {
        self.exclude_behavior_notes = notes.into();
        self
    }

    pub fn heuristic_modes(&self) -> &[HeuristicMode] {
        &self.heuristic_modes
    }

    pub fn max_workspace_size(&self) -> Option<usize> {
        self.max_workspace_size
    }

    pub fn max_shared_memory_size(&self) -> Option<usize> {
        self.max_shared_memory_size
    }

    pub fn excluded_engine_name_substrings(&self) -> &[String] {
        &self.excluded_engine_name_substrings
    }

    pub fn build_policy(&self) -> BuildPlanPolicy {
        self.build_policy
    }

    pub fn sm_count_target(&self) -> Option<i64> {
        self.sm_count_target
    }

    pub fn include_numerical_notes(&self) -> &[BackendNumericalNote] {
        &self.include_numerical_notes
    }

    pub fn exclude_numerical_notes(&self) -> &[BackendNumericalNote] {
        &self.exclude_numerical_notes
    }

    pub fn include_behavior_notes(&self) -> &[BackendBehaviorNote] {
        &self.include_behavior_notes
    }

    pub fn exclude_behavior_notes(&self) -> &[BackendBehaviorNote] {
        &self.exclude_behavior_notes
    }

    pub(crate) fn has_engine_filters(&self) -> bool {
        self.max_workspace_size.is_some()
            || self.max_shared_memory_size.is_some()
            || !self.excluded_engine_name_substrings.is_empty()
            || !self.include_numerical_notes.is_empty()
            || !self.exclude_numerical_notes.is_empty()
            || !self.include_behavior_notes.is_empty()
            || !self.exclude_behavior_notes.is_empty()
    }

    pub(crate) fn cache_key_json(&self) -> Result<serde_json::Value> {
        let js = serde_json::to_value(self)?;
        Ok(js)
    }
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self::new()
    }
}
