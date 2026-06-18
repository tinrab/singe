use serde::{Deserialize, Serialize};

/// Attributes for a GenStats operation.
///
/// GenStats computes per-channel sum and sum-of-squares outputs for batch
/// normalization finalize.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GenStatsConfig;

impl GenStatsConfig {
    pub fn new() -> Self {
        Self
    }
}
