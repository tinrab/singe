//! cuDNN frontend graph API.

pub mod bindings;
pub mod composite;
pub mod errata;
pub mod error;
pub mod graph;
pub mod logging;
pub(crate) mod lower;
pub mod operation;
pub mod plan;
pub mod workspace;

use singe_core::LibraryVersion;

use crate::{error::Result, frontend::error::Error as FrontendError};

/// Oldest cuDNN runtime supported by frontend preparation, implementation selection and saved native artifacts.
pub(crate) const BASELINE_CUDNN_VERSION: LibraryVersion = LibraryVersion::new(9, 27, 0);

pub(crate) fn require_baseline(actual_version: LibraryVersion) -> Result<()> {
    if actual_version < BASELINE_CUDNN_VERSION {
        return Err(FrontendError::FeatureRequiresVersion {
            feature: "cudnn frontend",
            min_version: BASELINE_CUDNN_VERSION,
            actual_version,
        }
        .into());
    }
    Ok(())
}
