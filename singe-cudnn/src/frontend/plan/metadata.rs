use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    error::{Error, Result},
    execution::{EngineConfig, KnobChoiceInfo, OperationGraph},
    frontend::operation::CompileConfig,
};

pub(crate) fn is_workspace_within_limit(
    workspace_size: usize,
    max_workspace_size: Option<usize>,
) -> bool {
    max_workspace_size.is_none_or(|max_workspace_size| workspace_size <= max_workspace_size)
}

pub(crate) fn workspace_limit_error(
    workspace_size: usize,
    max_workspace_size: Option<usize>,
) -> Option<Error> {
    (!is_workspace_within_limit(workspace_size, max_workspace_size)).then(|| {
        Error::FrontendCompile(format!(
            "skipping plan since workspace violation. requires {workspace_size}"
        ))
    })
}

#[derive(Debug, Clone, Copy)]
pub struct EngineConfigFilter<'a> {
    max_workspace_size: Option<usize>,
    max_shared_memory_size: Option<usize>,
    excluded_engine_name_substrings: &'a [String],
    include_numerical_notes: &'a [BackendNumericalNote],
    exclude_numerical_notes: &'a [BackendNumericalNote],
    include_behavior_notes: &'a [BackendBehaviorNote],
    exclude_behavior_notes: &'a [BackendBehaviorNote],
}

impl<'a> EngineConfigFilter<'a> {
    pub fn new() -> Self {
        Self {
            max_workspace_size: None,
            max_shared_memory_size: None,
            excluded_engine_name_substrings: &[],
            include_numerical_notes: &[],
            exclude_numerical_notes: &[],
            include_behavior_notes: &[],
            exclude_behavior_notes: &[],
        }
    }

    pub(crate) fn from_compile_config(compile_config: &'a CompileConfig) -> Self {
        Self {
            max_workspace_size: compile_config.max_workspace_size(),
            max_shared_memory_size: compile_config.max_shared_memory_size(),
            excluded_engine_name_substrings: compile_config.excluded_engine_name_substrings(),
            include_numerical_notes: compile_config.include_numerical_notes(),
            exclude_numerical_notes: compile_config.exclude_numerical_notes(),
            include_behavior_notes: compile_config.include_behavior_notes(),
            exclude_behavior_notes: compile_config.exclude_behavior_notes(),
        }
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
        excluded_engine_name_substrings: &'a [String],
    ) -> Self {
        self.excluded_engine_name_substrings = excluded_engine_name_substrings;
        self
    }

    pub fn with_included_numerical_notes(
        mut self,
        include_numerical_notes: &'a [BackendNumericalNote],
    ) -> Self {
        self.include_numerical_notes = include_numerical_notes;
        self
    }

    pub fn with_excluded_numerical_notes(
        mut self,
        exclude_numerical_notes: &'a [BackendNumericalNote],
    ) -> Self {
        self.exclude_numerical_notes = exclude_numerical_notes;
        self
    }

    pub fn with_included_behavior_notes(
        mut self,
        include_behavior_notes: &'a [BackendBehaviorNote],
    ) -> Self {
        self.include_behavior_notes = include_behavior_notes;
        self
    }

    pub fn with_excluded_behavior_notes(
        mut self,
        exclude_behavior_notes: &'a [BackendBehaviorNote],
    ) -> Self {
        self.exclude_behavior_notes = exclude_behavior_notes;
        self
    }

    pub fn accepts(self, metadata: &EngineConfigMetadata) -> bool {
        metadata.is_within_workspace_limit(self.max_workspace_size)
            && metadata.is_within_shared_memory_limit(self.max_shared_memory_size)
            && metadata.is_not_named_by(self.excluded_engine_name_substrings)
            && metadata.has_all_numerical_notes(self.include_numerical_notes)
            && metadata.has_no_numerical_notes(self.exclude_numerical_notes)
            && metadata.has_all_behavior_notes(self.include_behavior_notes)
            && metadata.has_no_behavior_notes(self.exclude_behavior_notes)
    }

    pub(crate) fn rejection_error(self, metadata: &EngineConfigMetadata) -> Error {
        workspace_limit_error(metadata.workspace_size, self.max_workspace_size)
            .unwrap_or_else(|| Error::FrontendCompile(format!("plan {} rejected", metadata.name)))
    }
}

impl Default for EngineConfigFilter<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached descriptor metadata for one heuristic engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfigMetadata {
    pub name: String,
    pub engine_index: i64,
    pub workspace_size: usize,
    pub shared_memory_size: usize,
    pub numerical_notes: Vec<BackendNumericalNote>,
    pub behavior_notes: Vec<BackendBehaviorNote>,
    pub knob_choices: Vec<KnobChoiceInfo>,
}

impl EngineConfigMetadata {
    pub fn create(operation_graph: &OperationGraph, engine_config: &EngineConfig) -> Result<Self> {
        let engine = engine_config.engine()?;
        let engine_index = engine.index().as_i64();
        let knob_choices = engine_config.knob_choices()?;
        let mut name = format!("engine{engine_index}");
        for choice in &knob_choices {
            name.push_str(&format!("_k{}={}", choice.knob_type as i32, choice.value));
        }

        Ok(Self {
            name,
            engine_index,
            workspace_size: engine_config.workspace_size()?,
            shared_memory_size: engine_config.shared_memory_size()?,
            numerical_notes: engine.numerical_notes(operation_graph.descriptor())?,
            behavior_notes: engine.behavior_notes(operation_graph.descriptor())?,
            knob_choices,
        })
    }

    pub(crate) fn matches_compile_config(&self, compile_config: &CompileConfig) -> bool {
        EngineConfigFilter::from_compile_config(compile_config).accepts(self)
    }

    pub(crate) fn is_within_workspace_limit(&self, max_workspace_size: Option<usize>) -> bool {
        workspace_limit_error(self.workspace_size, max_workspace_size).is_none()
    }

    pub(crate) fn is_within_shared_memory_limit(
        &self,
        max_shared_memory_size: Option<usize>,
    ) -> bool {
        max_shared_memory_size
            .is_none_or(|max_shared_memory_size| self.shared_memory_size <= max_shared_memory_size)
    }

    pub(crate) fn is_not_named_by(&self, blocked_name_substrings: &[String]) -> bool {
        !blocked_name_substrings
            .iter()
            .any(|blocked_name| self.name.contains(blocked_name))
    }

    pub(crate) fn has_all_numerical_notes(&self, required_notes: &[BackendNumericalNote]) -> bool {
        required_notes
            .iter()
            .all(|note| self.numerical_notes.contains(note))
    }

    pub(crate) fn has_no_numerical_notes(&self, excluded_notes: &[BackendNumericalNote]) -> bool {
        !excluded_notes
            .iter()
            .any(|note| self.numerical_notes.contains(note))
    }

    pub(crate) fn has_all_behavior_notes(&self, required_notes: &[BackendBehaviorNote]) -> bool {
        required_notes
            .iter()
            .all(|note| self.behavior_notes.contains(note))
    }

    pub(crate) fn has_no_behavior_notes(&self, excluded_notes: &[BackendBehaviorNote]) -> bool {
        !excluded_notes
            .iter()
            .any(|note| self.behavior_notes.contains(note))
    }
}
