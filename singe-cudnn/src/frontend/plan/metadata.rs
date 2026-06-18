use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    error::Result,
    execution::{EngineConfig, KnobChoiceInfo, OperationGraph},
};

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
}
