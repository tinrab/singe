use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        engine::EngineIndex,
        execution::{EngineConfig, KnobChoiceInfo},
    },
    error::{Error, Result},
    frontend::{
        logging,
        plan::{CompileConfig, PlanExclusion},
    },
};

/// In-memory engine eligibility filter applied to cached engine metadata.
/// Resource limits exclude known requirements above them; unknown shared-memory requirements remain eligible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfigFilter {
    max_workspace_size: Option<usize>,
    max_shared_memory_size: Option<usize>,
    excluded_engine_name_substrings: Vec<String>,
    included_numerical_notes: Vec<BackendNumericalNote>,
    excluded_numerical_notes: Vec<BackendNumericalNote>,
    included_behavior_notes: Vec<BackendBehaviorNote>,
    excluded_behavior_notes: Vec<BackendBehaviorNote>,
}

impl EngineConfigFilter {
    /// Creates a filter without resource limits, name exclusions, or note requirements.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the workspace exclusion limit, or removes it with `None`.
    #[must_use]
    pub fn with_max_workspace_size(mut self, max_workspace_size: impl Into<Option<usize>>) -> Self {
        self.max_workspace_size = max_workspace_size.into();
        self
    }

    /// Excludes known shared-memory requirements above this byte limit.
    /// Unknown requirements remain eligible, matching C++ plan filtering.
    /// `None` removes this exclusion limit while preserving note and name filters.
    #[must_use]
    pub fn with_max_shared_memory_size(
        mut self,
        max_shared_memory_size: impl Into<Option<usize>>,
    ) -> Self {
        self.max_shared_memory_size = max_shared_memory_size.into();
        self
    }

    /// Replaces case-sensitive engine-name substring exclusions; an empty list removes name exclusions.
    #[must_use]
    pub fn with_excluded_engine_name_substrings(
        mut self,
        substrings: impl Into<Vec<String>>,
    ) -> Self {
        self.excluded_engine_name_substrings = substrings.into();
        self
    }

    /// Replaces numerical notes that every accepted engine must contain.
    #[must_use]
    pub fn with_included_numerical_notes(
        mut self,
        notes: impl Into<Vec<BackendNumericalNote>>,
    ) -> Self {
        self.included_numerical_notes = notes.into();
        self
    }

    /// Replaces numerical notes whose presence excludes an engine.
    #[must_use]
    pub fn with_excluded_numerical_notes(
        mut self,
        notes: impl Into<Vec<BackendNumericalNote>>,
    ) -> Self {
        self.excluded_numerical_notes = notes.into();
        self
    }

    /// Replaces behavior notes that every accepted engine must contain.
    #[must_use]
    pub fn with_included_behavior_notes(
        mut self,
        notes: impl Into<Vec<BackendBehaviorNote>>,
    ) -> Self {
        self.included_behavior_notes = notes.into();
        self
    }

    /// Replaces behavior notes whose presence excludes an engine.
    #[must_use]
    pub fn with_excluded_behavior_notes(
        mut self,
        notes: impl Into<Vec<BackendBehaviorNote>>,
    ) -> Self {
        self.excluded_behavior_notes = notes.into();
        self
    }

    /// Returns the workspace exclusion limit in bytes, or `None` when unset.
    #[must_use]
    pub fn max_workspace_size(&self) -> Option<usize> {
        self.max_workspace_size
    }

    /// Returns the shared-memory exclusion limit in bytes, or `None` when unset.
    #[must_use]
    pub fn max_shared_memory_size(&self) -> Option<usize> {
        self.max_shared_memory_size
    }

    /// Borrows case-sensitive engine-name substrings that reject a candidate.
    #[must_use]
    pub fn excluded_engine_name_substrings(&self) -> &[String] {
        &self.excluded_engine_name_substrings
    }

    /// Borrows numerical notes required on every accepted engine.
    #[must_use]
    pub fn included_numerical_notes(&self) -> &[BackendNumericalNote] {
        &self.included_numerical_notes
    }

    /// Borrows numerical notes whose presence rejects an engine.
    #[must_use]
    pub fn excluded_numerical_notes(&self) -> &[BackendNumericalNote] {
        &self.excluded_numerical_notes
    }

    /// Borrows behavior notes required on every accepted engine.
    #[must_use]
    pub fn included_behavior_notes(&self) -> &[BackendBehaviorNote] {
        &self.included_behavior_notes
    }

    /// Borrows behavior notes whose presence rejects an engine.
    #[must_use]
    pub fn excluded_behavior_notes(&self) -> &[BackendBehaviorNote] {
        &self.excluded_behavior_notes
    }

    /// Matches cached metadata without allocation or native descriptor queries.
    /// Every included note is required, and any excluded note rejects the engine.
    /// Unknown shared-memory requirements do not trigger the shared-memory limit.
    #[must_use]
    pub fn accepts(&self, metadata: &EngineConfigMetadata) -> bool {
        self.exclusion(metadata).is_none()
    }

    /// Describes why cached requirements fail this filter, without native calls.
    /// Returns the first matching resource, name or note exclusion in that order.
    #[must_use]
    pub fn exclusion(&self, metadata: &EngineConfigMetadata) -> Option<PlanExclusion> {
        if let Some(limit) = self.max_workspace_size
            && metadata.workspace_size > limit
        {
            return Some(PlanExclusion::Workspace {
                required: metadata.workspace_size,
                limit,
            });
        }
        if let (Some(required), Some(limit)) =
            (metadata.shared_memory_size, self.max_shared_memory_size)
            && required > limit
        {
            return Some(PlanExclusion::SharedMemory { required, limit });
        }
        if let Some(pattern) = self
            .excluded_engine_name_substrings
            .iter()
            .find(|pattern| metadata.name.contains(pattern.as_str()))
        {
            return Some(PlanExclusion::EngineName {
                pattern: pattern.clone(),
            });
        }
        if let Some(&note) = self
            .included_numerical_notes
            .iter()
            .find(|note| !metadata.numerical_notes.contains(note))
        {
            return Some(PlanExclusion::MissingNumericalNote { note });
        }
        if let Some(&note) = self
            .excluded_numerical_notes
            .iter()
            .find(|note| metadata.numerical_notes.contains(note))
        {
            return Some(PlanExclusion::ExcludedNumericalNote { note });
        }
        if let Some(&note) = self
            .included_behavior_notes
            .iter()
            .find(|note| !metadata.behavior_notes.contains(note))
        {
            return Some(PlanExclusion::MissingBehaviorNote { note });
        }
        self.excluded_behavior_notes
            .iter()
            .find(|note| metadata.behavior_notes.contains(note))
            .map(|&note| PlanExclusion::ExcludedBehaviorNote { note })
    }

    /// Order-independent projection used by compilation-cache keys.
    pub(crate) fn canonical(&self) -> Self {
        let mut canonical = self.clone();
        canonical.excluded_engine_name_substrings.sort_unstable();
        canonical.excluded_engine_name_substrings.dedup();
        canonical.included_numerical_notes.sort_unstable();
        canonical.included_numerical_notes.dedup();
        canonical.excluded_numerical_notes.sort_unstable();
        canonical.excluded_numerical_notes.dedup();
        canonical.included_behavior_notes.sort_unstable();
        canonical.included_behavior_notes.dedup();
        canonical.excluded_behavior_notes.sort_unstable();
        canonical.excluded_behavior_notes.dedup();
        canonical
    }
}

/// A finalized configuration and its queried metadata before assigning a collection identity.
#[derive(Debug)]
pub struct PreparedEngineConfig {
    pub engine_config: EngineConfig,
    pub metadata: EngineConfigMetadata,
}

impl PreparedEngineConfig {
    pub(crate) fn create(
        engine_config: EngineConfig,
        frontend_workspace_bytes: usize,
    ) -> Result<Self> {
        let metadata = EngineConfigMetadata::create(&engine_config)?
            .with_frontend_workspace(frontend_workspace_bytes)?;
        Ok(Self {
            engine_config,
            metadata,
        })
    }
}

/// Cached native engine metadata retained across compilation and artifact restoration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfigMetadata {
    /// Canonical native engine and tuning-knob tag.
    pub name: String,
    /// Global native engine index.
    pub engine_index: EngineIndex,
    /// Cached workspace requirement in bytes; candidate preparation adds frontend workspace.
    pub workspace_size: usize,
    /// Native shared-memory requirement in bytes, or `None` when unavailable.
    pub shared_memory_size: Option<usize>,
    /// Native numerical properties used for eligibility filtering.
    pub numerical_notes: Vec<BackendNumericalNote>,
    /// Native behavior properties used for eligibility filtering.
    pub behavior_notes: Vec<BackendBehaviorNote>,
    /// Canonical tuning-knob choices sorted by native knob identity.
    pub knob_choices: Vec<KnobChoiceInfo>,
}

impl EngineConfigMetadata {
    pub(crate) fn exclusion(&self, config: &CompileConfig) -> Option<PlanExclusion> {
        config.filter().exclusion(self)
    }

    /// Collects requirements and notes using the configuration's retained native identity.
    /// Sorts candidate knob choices without changing the configuration's native snapshot order.
    /// Failed optional shared-memory queries leave that requirement unknown, matching C++ filtering.
    /// Optional failures reach diagnostic logging; the direct native getter remains fallible.
    ///
    /// # Errors
    /// Returns mandatory native query errors or an error for workspace sizes that cannot be represented.
    pub fn create(engine_config: &EngineConfig) -> Result<Self> {
        let engine = engine_config.engine()?;
        let choice = engine_config.choice()?;
        let mut knob_choices = choice.knob_choices.clone();
        knob_choices.sort_unstable_by_key(|choice| choice.knob_type);
        let shared_memory_size = match engine_config.shared_memory_size() {
            Ok(size) => size,
            Err(error) => {
                logging::diagnostic!(
                    "shared-memory requirement for engine {} is unavailable: {error}",
                    choice.engine_index,
                );
                None
            }
        };

        Ok(Self {
            name: engine_config.tag()?.to_owned(),
            engine_index: choice.engine_index,
            workspace_size: engine_config.workspace_size()?,
            shared_memory_size,
            numerical_notes: engine.numerical_notes()?.to_vec(),
            behavior_notes: engine.behavior_notes()?.to_vec(),
            knob_choices,
        })
    }

    pub(crate) fn with_frontend_workspace(mut self, bytes: usize) -> Result<Self> {
        self.workspace_size = self
            .workspace_size
            .checked_add(bytes)
            .ok_or_else(|| Error::out_of_range("candidate workspace bytes"))?;
        Ok(self)
    }

    pub(crate) fn matches_compile_config(&self, compile_config: &CompileConfig) -> bool {
        compile_config.filter().accepts(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(name: &str, workspace_size: usize) -> EngineConfigMetadata {
        EngineConfigMetadata {
            name: name.to_owned(),
            engine_index: EngineIndex::new(0),
            workspace_size,
            shared_memory_size: None,
            numerical_notes: Vec::new(),
            behavior_notes: vec![BackendBehaviorNote::RuntimeCompilation],
            knob_choices: Vec::new(),
        }
    }

    #[test]
    fn test_exclusions_follow_resource_name_note_order() {
        let filter = EngineConfigFilter::new()
            .with_max_workspace_size(16)
            .with_excluded_engine_name_substrings(["eng4_".to_owned()]);
        assert_eq!(
            filter.exclusion(&metadata("eng4_k1", 32)),
            Some(PlanExclusion::Workspace {
                required: 32,
                limit: 16,
            })
        );
        assert_eq!(
            filter.exclusion(&metadata("eng4_k1", 8)),
            Some(PlanExclusion::EngineName {
                pattern: "eng4_".to_owned(),
            })
        );
        assert!(filter.accepts(&metadata("eng1_k1", 8)));
    }

    #[test]
    fn test_unknown_shared_memory_is_eligible() {
        let filter = EngineConfigFilter::new().with_max_shared_memory_size(0);
        assert!(filter.accepts(&metadata("eng0", 0)));
    }

    #[test]
    fn test_required_behavior_notes() {
        let filter = EngineConfigFilter::new()
            .with_included_behavior_notes([BackendBehaviorNote::SupportsCudaGraphNativeApi]);
        assert_eq!(
            filter.exclusion(&metadata("eng0", 0)),
            Some(PlanExclusion::MissingBehaviorNote {
                note: BackendBehaviorNote::SupportsCudaGraphNativeApi,
            })
        );
    }

    #[test]
    fn test_canonical_filter_sorts_and_deduplicates() {
        let filter = EngineConfigFilter::new().with_excluded_engine_name_substrings([
            "b".to_owned(),
            "a".to_owned(),
            "b".to_owned(),
        ]);
        assert_eq!(
            filter.canonical().excluded_engine_name_substrings(),
            ["a", "b"]
        );
    }
}
