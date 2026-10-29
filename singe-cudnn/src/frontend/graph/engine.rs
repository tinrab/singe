use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        engine::{Engine, EngineIndex},
        execution::{EngineKnobInfo, EnginePlanningConfig},
    },
    context::Context,
    error::Result,
    frontend::graph::Graph,
};

impl Graph {
    /// Finalizes an engine for metadata queries before choosing its performance knobs.
    /// Pass Some(context) for a live target or None for the graph's attached deviceless properties.
    /// Resource targets override graph defaults when supplied.
    /// The returned engine retains its descriptor and actual live CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if graph preparation or target/context resolution fails, or engine creation or finalization fails, including invalid resource targets or incompatible graph and target metadata.
    pub fn engine(
        &self,
        context: Option<&Context>,
        engine_index: EngineIndex,
        planning: EnginePlanningConfig,
    ) -> Result<Engine> {
        self.prepare(context)?.engine(engine_index, planning)
    }

    /// Number of native engines available for the live or deviceless operation graph.
    ///
    /// # Errors
    ///
    /// Returns an error if graph preparation or target/context resolution fails, or querying or converting the native engine count fails.
    pub fn engine_count(&self, context: Option<&Context>) -> Result<usize> {
        self.prepare(context)?.engine_count()
    }

    /// Global engine indices for the live or deviceless operation graph.
    ///
    /// # Errors
    ///
    /// Returns an error if graph preparation or target/context resolution fails, or querying the engine count fails or the count cannot be represented as `i64`.
    pub fn engine_indices(&self, context: Option<&Context>) -> Result<Vec<EngineIndex>> {
        self.prepare(context)?.engine_indices()
    }

    /// Adjustable knobs supported by an engine for this graph and compilation target.
    ///
    /// # Errors
    ///
    /// Returns an error if graph preparation or target/context resolution fails, or engine construction, native knob-metadata querying, or knob-type decoding fails.
    pub fn knobs_for_engine(
        &self,
        context: Option<&Context>,
        engine_index: EngineIndex,
    ) -> Result<Vec<EngineKnobInfo>> {
        self.engine(context, engine_index, EnginePlanningConfig::default())?
            .knob_infos()
            .map(<[_]>::to_vec)
    }

    /// Numerical notes for an engine using this graph's actual target properties.
    ///
    /// # Errors
    ///
    /// Returns an error if graph preparation or target/context resolution fails, or engine construction, native numerical-note querying, or note decoding fails.
    pub fn numerical_notes_for_engine(
        &self,
        context: Option<&Context>,
        engine_index: EngineIndex,
    ) -> Result<Vec<BackendNumericalNote>> {
        self.engine(context, engine_index, EnginePlanningConfig::default())?
            .numerical_notes()
            .map(<[_]>::to_vec)
    }

    /// Behavior notes for an engine using this graph's actual target properties.
    ///
    /// # Errors
    ///
    /// Returns an error if graph preparation or target/context resolution fails, or engine construction, native behavior-note querying, or note decoding fails.
    pub fn behavior_notes_for_engine(
        &self,
        context: Option<&Context>,
        engine_index: EngineIndex,
    ) -> Result<Vec<BackendBehaviorNote>> {
        self.engine(context, engine_index, EnginePlanningConfig::default())?
            .behavior_notes()
            .map(<[_]>::to_vec)
    }
}
