//! ONNX model utilities.

/// Error types returned by ONNX decoding and parser conversion.
pub mod error;
/// Typed ONNX model representation parsed from generated protobuf values.
pub mod model;

/// Generated ONNX protobuf types.
///
/// These types are generated from `proto/onnx.proto` at build time and kept public for callers that need raw ONNX protobuf access.
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

    /// Structure-only ONNX protobuf module generated from `proto/onnx_structure.proto`.
    #[allow(warnings)]
    #[allow(clippy::all)]
    #[allow(missing_docs)]
    #[allow(unused_qualifications)]
    pub mod onnx_structure {
        bomboni_proto::include_proto!("onnx_structure");
        bomboni_proto::include_proto!("onnx_structure.plus");
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

/// Reads and parses an ONNX model using the structure-only protobuf view.
pub fn read_structure<R>(mut reader: R) -> Result<Model>
where
    R: Read,
{
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    decode_structure(&bytes)
}

/// Reads and parses an ONNX model file using the structure-only protobuf view.
pub fn read_structure_path(path: impl AsRef<Path>) -> Result<Model> {
    read_structure(File::open(path)?)
}

/// Decodes and parses ONNX protobuf bytes using the structure-only protobuf view.
///
/// Inline tensor payload fields are omitted from this schema, so `prost` skips
/// them as unknown protobuf fields instead of allocating weight buffers.
pub fn decode_structure(bytes: &[u8]) -> Result<Model> {
    let proto = proto::onnx_structure::ModelProto::decode(bytes)?;
    Ok(proto.parse_into()?)
}

/// Reads and decodes an ONNX model using the structure-only protobuf view.
pub fn read_structure_proto<R>(mut reader: R) -> Result<proto::onnx_structure::ModelProto>
where
    R: Read,
{
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    decode_structure_proto(&bytes)
}

/// Reads and decodes an ONNX model file using the structure-only protobuf view.
pub fn read_structure_proto_path(
    path: impl AsRef<Path>,
) -> Result<proto::onnx_structure::ModelProto> {
    read_structure_proto(File::open(path)?)
}

/// Decodes ONNX protobuf bytes using the structure-only protobuf view.
///
/// Inline tensor payload fields are omitted from this schema, so `prost` skips
/// them as unknown protobuf fields instead of allocating weight buffers.
pub fn decode_structure_proto(bytes: &[u8]) -> Result<proto::onnx_structure::ModelProto> {
    Ok(proto::onnx_structure::ModelProto::decode(bytes)?)
}
