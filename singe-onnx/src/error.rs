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
    Format(#[from] RequestError),
}

/// Result type used by `singe-onnx`.
pub type Result<T> = StdResult<T, Error>;
