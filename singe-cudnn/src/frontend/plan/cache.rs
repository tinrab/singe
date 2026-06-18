use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{error::Result, execution::EngineConfig, knob::BackendKnobType};

/// Engine and knob choices cached for a previously compiled frontend graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedPlanChoice {
    pub engine_index: i64,
    pub knob_choices: Vec<(BackendKnobType, i64)>,
}

impl CachedPlanChoice {
    pub(crate) fn from_engine_config(engine_config: &EngineConfig) -> Result<Self> {
        Ok(Self {
            engine_index: engine_config.engine_index()?.as_i64(),
            knob_choices: engine_config
                .knob_choices()?
                .into_iter()
                .map(|choice| (choice.knob_type, choice.value))
                .collect(),
        })
    }
}

/// In-process cache of selected cuDNN engine choices.
///
/// This intentionally stores engine indices and knob choices rather than raw execution-plan descriptors.
/// A cache hit still rebuilds descriptors for the current context, but skips heuristic collection and filtering.
#[derive(Debug, Default)]
pub struct PlanCache {
    choices: HashMap<String, Vec<CachedPlanChoice>>,
}

impl PlanCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&[CachedPlanChoice]> {
        self.choices.get(key).map(Vec::as_slice)
    }

    pub fn insert(&mut self, key: impl Into<String>, choices: Vec<CachedPlanChoice>) {
        if !choices.is_empty() {
            self.choices.insert(key.into(), choices);
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<Vec<CachedPlanChoice>> {
        self.choices.remove(key)
    }

    pub fn clear(&mut self) {
        self.choices.clear();
    }

    pub fn len(&self) -> usize {
        self.choices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }
}
