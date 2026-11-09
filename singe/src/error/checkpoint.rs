use std::io::Error as IoError;

use safetensors::SafeTensorError;
use serde_json::Error as JsonError;
use thiserror::Error;

/// Errors from reading and writing named-state checkpoints.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CheckpointError {
    /// A checkpoint could not be read or written.
    #[error("checkpoint io")]
    Io(#[from] IoError),

    /// A safetensors checkpoint is malformed.
    #[error("checkpoint format")]
    Format(#[from] SafeTensorError),

    /// A shard index is malformed.
    #[error("checkpoint shard index")]
    Index(#[source] JsonError),

    /// Checkpoint names do not match the model schema.
    #[error("checkpoint keys: missing {missing:?}, unexpected {unexpected:?}")]
    Keys {
        /// Sorted schema names whose tied parameter or buffer has no checkpoint entry.
        missing: Vec<String>,
        /// Sorted checkpoint names absent from the model state schema.
        unexpected: Vec<String>,
    },
}
