/// A CUDA graph compilation stage included in pipeline and cache identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileStage {
    /// Captures tensor operations into a graph.
    Capture,
    /// Normalizes the captured graph before backend compilation.
    Normalize,
    /// Analyzes tensor shapes and storage aliases.
    ShapeAndAliasAnalysis,
    /// Assigns graph operations to backend providers.
    BackendPartition,
    /// Combines compatible operations into fused backend work.
    Fusion,
    /// Plans allocations and reuse of tensor storage.
    MemoryPlan,
    /// Builds the executable CUDA graph.
    Executable,
}

/// A compile stage and its version salt for cache invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompileStageVersion {
    stage: CompileStage,
    version: u64,
}

/// An ordered list of CUDA compile stages and their cache version salts.
/// The default lists capture, normalization, shape and alias analysis, backend partitioning, fusion, memory planning, and executable construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilePipeline {
    stages: Vec<CompileStage>,
    stage_versions: Vec<CompileStageVersion>,
}

impl CompilePipeline {
    /// Stores the supplied stage order with version salt one for every entry.
    /// This constructor does not validate stage ordering or remove duplicate stages.
    pub fn new(stages: Vec<CompileStage>) -> Self {
        let stage_versions = stages.iter().copied().map(default_stage_version).collect();
        Self {
            stages,
            stage_versions,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_stage_version(mut self, stage: CompileStage, version: u64) -> Self {
        if let Some(stage_version) = self
            .stage_versions
            .iter_mut()
            .find(|stage_version| stage_version.stage == stage)
        {
            stage_version.version = version;
        }
        self
    }

    /// Returns stage version salts in pipeline order.
    #[must_use]
    pub fn stage_versions(&self) -> &[CompileStageVersion] {
        &self.stage_versions
    }
}

impl CompileStageVersion {
    fn new(stage: CompileStage, version: u64) -> Self {
        Self { stage, version }
    }

    /// Returns the compile stage this version belongs to.
    #[must_use]
    pub fn stage(self) -> CompileStage {
        self.stage
    }

    /// Returns the version salt for this compile stage.
    #[must_use]
    pub fn version(self) -> u64 {
        self.version
    }
}

impl Default for CompilePipeline {
    fn default() -> Self {
        Self::new(vec![
            CompileStage::Capture,
            CompileStage::Normalize,
            CompileStage::ShapeAndAliasAnalysis,
            CompileStage::BackendPartition,
            CompileStage::Fusion,
            CompileStage::MemoryPlan,
            CompileStage::Executable,
        ])
    }
}

fn default_stage_version(stage: CompileStage) -> CompileStageVersion {
    CompileStageVersion::new(stage, 1)
}
