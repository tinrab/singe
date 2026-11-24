use std::{io::Error as IoError, path::PathBuf};

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

    /// A flashpack checkpoint is malformed or cannot be written.
    #[error("flashpack checkpoint")]
    Flashpack(#[from] FlashpackError),

    // A load plan reads tensors the checkpoint does not store.
    #[error("checkpoint lacks tensors {names:?}")]
    Absent {
        // Sorted stored names the plan reads.
        names: Vec<String>,
    },

    /// Checkpoint names do not match the model schema.
    #[error("checkpoint keys: missing {missing:?}, unexpected {unexpected:?}")]
    Keys {
        /// Sorted schema names whose tied parameter or buffer has no checkpoint entry.
        missing: Vec<String>,
        /// Sorted checkpoint names absent from the model state schema.
        unexpected: Vec<String>,
    },
}

/// Errors from reading and writing flashpack files.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum FlashpackError {
    /// The metadata length runs past the start of the file.
    #[error("flashpack metadata of {length} bytes is longer than its file")]
    MetadataLength {
        /// The length the footer gives.
        length: u64,
    },

    /// The metadata is not the JSON flashpack writes.
    #[error("flashpack metadata")]
    Metadata(#[source] JsonError),

    /// The metadata lacks a field its format needs.
    #[error("flashpack metadata lacks `{0}`")]
    MissingField(&'static str),

    /// The metadata names a format other than `flashpack_v3` and `flashpack_v4`.
    #[error("flashpack format `{0}` is unsupported")]
    Format(String),

    /// The metadata names a data type Singe has no type for.
    #[error("flashpack data type `{0}` is unsupported")]
    DataType(String),

    /// A macroblock's bytes run past the payload or disagree with its element count.
    #[error(
        "flashpack macroblock {index} lies outside the payload or disagrees with its element count"
    )]
    Macroblock {
        /// The macroblock's position in the metadata.
        index: usize,
    },

    /// A tensor lies outside its macroblock.
    #[error("flashpack tensor `{0}` lies outside its macroblock")]
    Range(String),

    /// A tensor's length differs from its shape's element count.
    #[error("flashpack tensor `{name}` has {length} elements, which its shape does not")]
    Length {
        /// The tensor's name.
        name: String,
        /// The length the index gives.
        length: u64,
    },

    /// A flashpack file holds at least one tensor.
    #[error("flashpack file without tensors")]
    Empty,

    /// The destination has no file name to write beside.
    #[error("flashpack destination {0} names no file")]
    Path(PathBuf),
}
