use std::collections::HashMap;

use crate::backend::execution::EngineChoice;

/// In-process cache of selected cuDNN engine choices.
///
/// This intentionally stores engine indices and knob choices rather than raw execution-plan descriptors.
/// A cache hit rebuilds descriptors for the current context and reapplies compile filters,
/// while skipping heuristic collection.
/// Cached choices must satisfy the prepared graph's required backward engine and knob values.
/// Additional compatible tuning knobs remain available, and a miss falls back to normal discovery.
/// Dynamic-shape graphs share keys across tensor dimensions and strides. Each hit builds against
/// the current graph's tensor specifications; it does not reuse another graph's bindings or workspace.
/// Diagnostic names, runtime scalar defaults and mutable kernel-cache payloads do not distinguish entries.
/// Rebuilding descriptors preserves the current graph's constants, scalar defaults and tensor identities.
/// Keys normalize typed tensor IDs by operation use and omit unused declarations; cached choices never contain binding IDs.
/// Note filters and blocked-name patterns use set identity, while heuristic-mode order remains significant.
/// Rebuilt candidates use the caller's current filter order when reporting exclusions.
/// Choice order establishes rebuild priority, with the preferred plan first.
/// [`crate::frontend::plan::CompiledGraph::cached_plan_choices`] places its selected plan first without changing the stored collection.
#[derive(Debug, Default)]
pub struct PlanCache {
    choices: HashMap<String, Vec<EngineChoice>>,
}

impl PlanCache {
    /// Creates an empty cache of ordered native engine choices.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrows the ordered engine choices for a graph key, or returns `None` on a miss.
    pub fn get(&self, key: &str) -> Option<&[EngineChoice]> {
        self.choices.get(key).map(Vec::as_slice)
    }

    /// Stores nonempty ordered choices for a key, replacing its previous choices; an empty list leaves the cache unchanged.
    pub fn insert(&mut self, key: impl Into<String>, choices: Vec<EngineChoice>) {
        if !choices.is_empty() {
            self.choices.insert(key.into(), choices);
        }
    }

    /// Removes and returns the ordered choices for a key, or returns `None` on a miss.
    pub fn remove(&mut self, key: &str) -> Option<Vec<EngineChoice>> {
        self.choices.remove(key)
    }

    /// Removes all graph keys and their retained engine choices.
    pub fn clear(&mut self) {
        self.choices.clear();
    }

    /// Returns the number of graph keys with cached choices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.choices.len()
    }

    /// Returns whether no graph keys have cached choices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }
}
