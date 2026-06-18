//! ONNX model parsing utilities.

/// Error types returned by ONNX decoding and parser conversion.
pub mod error;
/// Typed ONNX model representation parsed from generated protobuf values.
pub mod model;

/// Generated ONNX protobuf types.
///
/// These are generated from `proto/onnx.proto` at build time and kept public
/// for callers that need raw ONNX protobuf access.
pub mod proto {
    /// Raw ONNX protobuf module generated from `proto/onnx.proto`.
    #[allow(warnings)]
    #[allow(clippy::all)]
    #[allow(missing_docs)]
    #[allow(unused_qualifications)]
    pub mod onnx {
        bomboni_proto::include_proto!("onnx");
        bomboni_proto::include_proto!("onnx.plus");
    }
}

use std::{fs::File, io::Read, path::Path};

use bomboni_request::parse::RequestParseInto;
use prost::Message;

pub use crate::{
    error::{Error, Result},
    model::*,
};

/// Reads and parses an ONNX model from a reader.
pub fn read<R>(mut reader: R) -> Result<Model>
where
    R: Read,
{
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    decode(&bytes)
}

/// Reads and parses an ONNX model from a file.
pub fn read_path(path: impl AsRef<Path>) -> Result<Model> {
    read(File::open(path)?)
}

/// Decodes and parses an ONNX model from protobuf bytes.
pub fn decode(bytes: &[u8]) -> Result<Model> {
    let proto = proto::onnx::ModelProto::decode(bytes)?;
    Ok(proto.parse_into()?)
}

/// Decodes ONNX protobuf bytes without converting them to Singe's typed parser model.
pub fn decode_proto(bytes: &[u8]) -> Result<proto::onnx::ModelProto> {
    Ok(proto::onnx::ModelProto::decode(bytes)?)
}
