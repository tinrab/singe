use std::{io, result::Result as StdResult};

use bomboni_request::error::RequestError;
use prost::DecodeError;
use thiserror::Error;

#[derive(Debug, Error)]
/// Error returned by ONNX decoding and parser conversion.
#[non_exhaustive]
pub enum Error {
    /// Failed to read ONNX data from an input stream or file.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// Failed to decode ONNX protobuf bytes.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// Failed to convert decoded protobuf into Singe's typed ONNX model.
    #[error(transparent)]
    Format(#[from] FormatError),
}

/// A decoded ONNX message that violates the typed model's format requirements.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct FormatError(RequestError);

impl FormatError {
    pub(crate) fn new(error: RequestError) -> Self {
        Self(error)
    }
}

/// Result type used by `singe-onnx`.
pub type Result<T> = StdResult<T, Error>;
