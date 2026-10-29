use serde::{Deserialize, Serialize};

pub use crate::backend::execution::normalization::NormalizationSavedStatistics;
use crate::backend::{normalization::BackendNormalizationForwardPhase, tensor::TensorId};

/// Inference has no saved-statistics operands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationStatistics<T = TensorId> {
    /// Inference phase without saved training statistics.
    #[default]
    Inference,
    /// Training phase with saved mean and reciprocal normalization factor.
    Training(NormalizationSavedStatistics<T>),
}

impl<T> NormalizationStatistics<T> {
    /// Returns the native forward phase selected by this statistics request.
    pub fn phase(self) -> BackendNormalizationForwardPhase {
        match self {
            Self::Inference => BackendNormalizationForwardPhase::Inference,
            Self::Training(_) => BackendNormalizationForwardPhase::Training,
        }
    }

    /// Returns saved-statistics destinations in training, or `None` for inference.
    pub fn saved(self) -> Option<NormalizationSavedStatistics<T>> {
        match self {
            Self::Inference => None,
            Self::Training(stats) => Some(stats),
        }
    }
}
