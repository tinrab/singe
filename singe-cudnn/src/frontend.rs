//! cuDNN frontend graph API.

/// Tensor pointer registrations, retained scalar defaults, and runtime layout overrides.
pub mod bindings;
/// Attention graph compositions, score modifiers, and quantized forward and backward operands.
pub mod composite;
/// Typed errors for graph validation, native preparation, planning, bindings, and execution.
pub mod error;
/// Declarative tensor graphs with validation, target-specific preparation, and compilation.
pub mod graph;
pub mod logging;
pub(crate) mod lower;
pub mod operation;
/// Engine candidates, compiled selections, execution arguments, caching, and autotuning.
pub mod plan;
pub mod reorder;
/// Aligned frontend scratch layouts and initialization metadata preceding native workspace.
pub mod workspace;

use crate::{error::Result, frontend::error::Error as FrontendError};

/// Oldest cuDNN runtime supported by frontend preparation, implementation selection and saved native artifacts.
pub(crate) const BASELINE_CUDNN_VERSION: u64 = 92_700;

pub(crate) fn require_baseline(actual_version: u64) -> Result<()> {
    if actual_version < BASELINE_CUDNN_VERSION {
        return Err(FrontendError::FeatureRequiresVersion {
            feature: "cudnn frontend",
            min_version: "9.27",
            actual_version,
        }
        .into());
    }
    Ok(())
}
