use std::{
    fmt,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    sync::Arc,
};

use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        execution::{EngineConfig, ExecutionPlan},
    },
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        plan::{
            CompileConfig, EngineChoice, EngineConfigFilter, PlanSupport,
            metadata::EngineConfigMetadata,
        },
    },
};

/// Opaque identity of one candidate in its owning plan collection.
/// Identities survive filtering, reordering, build-stage transitions and timing rankings.
/// Restored collections receive fresh identities.
#[derive(Clone)]
pub struct PlanId {
    collection: Arc<()>,
    ordinal: usize,
}

impl PlanId {
    pub(crate) fn collection(count: usize) -> impl Iterator<Item = Self> {
        let collection = Arc::new(());
        (0..count).map(move |ordinal| Self {
            collection: Arc::clone(&collection),
            ordinal,
        })
    }
}

impl PartialEq for PlanId {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && Arc::ptr_eq(&self.collection, &other.collection)
    }
}

impl Eq for PlanId {}

impl Hash for PlanId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.collection).hash(state);
        self.ordinal.hash(state);
    }
}

impl fmt::Debug for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "plan{}", self.ordinal)
    }
}

/// Stage at which a candidate-specific planning or tuning failure occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanFailureStage {
    /// Reading planning settings or resource metadata, including finalized workspace requirements.
    Requirements,
    /// Creating the target context, reconstructing an engine configuration or compiling a native plan.
    Build,
    /// Preparing or executing warmup runs before measurement.
    Warmup,
    /// Measuring candidate execution time.
    Measurement,
}

impl fmt::Display for PlanFailureStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Requirements => "requirements query",
            Self::Build => "build",
            Self::Warmup => "warmup",
            Self::Measurement => "measurement",
        })
    }
}

/// A candidate-specific failure with its original typed cause.
#[derive(Debug, Clone, thiserror::Error)]
#[error("`{plan}` failed during {stage}: {source}")]
pub struct PlanFailure {
    /// Stable identity of the failed candidate.
    pub plan: PlanId,
    /// Planning or tuning stage that failed.
    pub stage: PlanFailureStage,
    /// Original typed cause, shared without erasing its error variant.
    pub source: Arc<Error>,
}

/// Selection-policy reason for excluding a candidate, independently of its build outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanExclusion {
    /// The known workspace requirement exceeds the configured byte limit.
    Workspace {
        /// Known workspace requirement in bytes.
        required: usize,
        /// Configured exclusion limit in bytes.
        limit: usize,
    },
    /// The known shared-memory requirement exceeds the configured byte limit.
    SharedMemory {
        /// Known shared-memory requirement in bytes.
        required: usize,
        /// Configured exclusion limit in bytes.
        limit: usize,
    },
    /// The native engine name contains an excluded substring.
    EngineName {
        /// Excluded case-sensitive substring matched within the engine name.
        pattern: String,
    },
    /// A required numerical note is absent.
    MissingNumericalNote {
        /// Required numerical note.
        note: BackendNumericalNote,
    },
    /// An excluded numerical note is present.
    ExcludedNumericalNote {
        /// Excluded numerical note.
        note: BackendNumericalNote,
    },
    /// A required behavior note is absent.
    MissingBehaviorNote {
        /// Required behavior note.
        note: BackendBehaviorNote,
    },
    /// An excluded behavior note is present.
    ExcludedBehaviorNote {
        /// Excluded behavior note.
        note: BackendBehaviorNote,
    },
    /// A caller-supplied predicate rejected the candidate.
    Filter,
    /// The candidate exceeds the configured count of plans admitted to automatic selection.
    BuildLimit {
        /// Maximum admitted plan count.
        limit: NonZeroUsize,
    },
}

impl fmt::Display for PlanExclusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace { required, limit } => write!(
                f,
                "workspace requires {required} bytes, exceeding the {limit}-byte limit"
            ),
            Self::SharedMemory { required, limit } => write!(
                f,
                "shared memory requires {required} bytes, exceeding the {limit}-byte limit"
            ),
            Self::EngineName { pattern } => {
                write!(f, "engine name matches excluded pattern {pattern:?}")
            }
            Self::MissingNumericalNote { note } => {
                write!(f, "engine lacks required numerical note {note:?}")
            }
            Self::ExcludedNumericalNote { note } => {
                write!(f, "engine has excluded numerical note {note:?}")
            }
            Self::MissingBehaviorNote { note } => {
                write!(f, "engine lacks required behavior note {note:?}")
            }
            Self::ExcludedBehaviorNote { note } => {
                write!(f, "engine has excluded behavior note {note:?}")
            }
            Self::Filter => f.write_str("excluded by a custom engine predicate"),
            Self::BuildLimit { limit } => write!(f, "the build limit of {limit} plans was reached"),
        }
    }
}

/// Candidate build outcome, independently of selection-policy exclusions.
#[derive(Debug, Clone)]
pub enum PlanBuildStatus {
    /// Native construction has not been attempted.
    NotAttempted,
    /// A native execution plan was built successfully.
    Built,
    /// The attempted build failed with the retained typed cause.
    Failed(PlanFailure),
}

/// Build outcome and independent selection policy for a candidate.
/// Filtering a built or failed entry preserves its original build outcome.
/// Display includes both the build outcome and any exclusion, preserving failure diagnostics after filtering.
#[derive(Debug, Clone)]
pub struct PlanBuildReport {
    /// Stable identity of the reported candidate.
    pub plan: PlanId,
    /// Candidate name, including its graph-name prefix when present.
    pub name: String,
    /// Native engine index and canonical tuning-knob choices.
    pub choice: EngineChoice,
    /// Build outcome retained even when the candidate is excluded.
    pub status: PlanBuildStatus,
    /// Independent selection-policy exclusion, or `None` when eligible.
    pub exclusion: Option<PlanExclusion>,
}

impl PlanBuildReport {
    /// Returns whether the candidate was built successfully and has no selection-policy exclusion.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.exclusion.is_none() && matches!(self.status, PlanBuildStatus::Built)
    }
}

impl fmt::Display for PlanBuildReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.plan, self.name)?;
        match &self.status {
            PlanBuildStatus::NotAttempted => f.write_str(": build not attempted"),
            PlanBuildStatus::Built => f.write_str(": built"),
            PlanBuildStatus::Failed(failure) => write!(f, ": {failure}"),
        }?;
        if let Some(exclusion) = &self.exclusion {
            write!(f, "; excluded: {exclusion}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct PlanCandidate {
    pub id: PlanId,
    pub engine_config: Arc<EngineConfig>,
    /// Knob choices are canonicalized when querying native metadata or restoring an artifact.
    pub metadata: EngineConfigMetadata,
}

impl PlanCandidate {
    pub fn choice(&self) -> EngineChoice {
        EngineChoice::new(
            self.metadata.engine_index,
            self.metadata.knob_choices.clone(),
        )
    }

    pub fn name(&self, graph_name: Option<&str>) -> String {
        graph_name.map_or_else(
            || self.metadata.name.clone(),
            |graph| format!("{graph}/{}", self.metadata.name),
        )
    }
}

#[derive(Debug)]
pub enum PlanBuildOutcome {
    NotAttempted,
    Built(ExecutionPlan),
    Failed(PlanFailure),
}

#[derive(Debug)]
pub struct BuiltPlanEntry {
    pub outcome: PlanBuildOutcome,
    pub candidate: PlanCandidate,
    pub exclusion: Option<PlanExclusion>,
}

impl BuiltPlanEntry {
    pub(crate) fn support(&self, filter: &EngineConfigFilter) -> PlanSupport<'_> {
        PlanSupport {
            plan: &self.candidate.id,
            metadata: &self.candidate.metadata,
            exclusion: self
                .exclusion
                .clone()
                .or_else(|| filter.exclusion(&self.candidate.metadata)),
        }
    }

    pub(crate) fn unbuilt(candidate: PlanCandidate, config: &CompileConfig) -> Self {
        // Native resource overrides can change requirements, so filter after applying them.
        let overrides_resources =
            config.shared_memory_limit().is_some() || config.sm_count_target().is_some();
        let exclusion = if overrides_resources {
            None
        } else {
            candidate.metadata.exclusion(config)
        };
        Self {
            candidate,
            outcome: PlanBuildOutcome::NotAttempted,
            exclusion,
        }
    }

    pub fn report(&self, graph_name: Option<&str>) -> PlanBuildReport {
        PlanBuildReport {
            plan: self.candidate.id.clone(),
            name: self.candidate.name(graph_name),
            choice: self.candidate.choice(),
            status: match &self.outcome {
                PlanBuildOutcome::NotAttempted => PlanBuildStatus::NotAttempted,
                PlanBuildOutcome::Built(_) => PlanBuildStatus::Built,
                PlanBuildOutcome::Failed(failure) => PlanBuildStatus::Failed(failure.clone()),
            },
            exclusion: self.exclusion.clone(),
        }
    }

    pub fn execution_plan(&self, graph_name: Option<&str>) -> Result<&ExecutionPlan> {
        self.available_plan().ok_or_else(|| {
            FrontendError::PlanUnavailable {
                report: Box::new(self.report(graph_name)),
            }
            .into()
        })
    }

    pub fn available_plan(&self) -> Option<&ExecutionPlan> {
        match (&self.outcome, &self.exclusion) {
            (PlanBuildOutcome::Built(plan), None) => Some(plan),
            _ => None,
        }
    }

    pub fn is_available(&self) -> bool {
        self.available_plan().is_some()
    }
}

#[derive(Debug)]
pub struct CompiledPlanEntry {
    // Destroy the plan before releasing its engine configuration.
    pub execution_plan: ExecutionPlan,
    pub candidate: PlanCandidate,
}
