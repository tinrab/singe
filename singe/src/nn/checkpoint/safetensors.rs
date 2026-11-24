// The safetensors format: a little-endian u64 header length, a JSON header naming each tensor's type, shape, and byte range, then the tensors' bytes.

use std::fs::File;

use safetensors::{
    Dtype, SafeTensorError, SafeTensors,
    tensor::{self, Metadata, TensorView},
};

use super::shard::{StoredTensor, read_exact_at, to_usize_u64};
use crate::{
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

// The largest header a safetensors file may have, as the format allows.
const LARGEST_HEADER: u64 = 100_000_000;

// Reads the header of a safetensors file.
pub fn index(file: &File) -> Result<Vec<(String, StoredTensor)>> {
    let mut length = [0; 8];
    read_exact_at(file, 0, &mut length)?;
    let length = u64::from_le_bytes(length);
    if length > LARGEST_HEADER {
        return Err(SafeTensorError::HeaderTooLarge.into());
    }

    let mut header = vec![0; to_usize_u64(length)?];
    read_exact_at(file, 8, &mut header)?;
    let metadata: Metadata =
        serde_json::from_slice(&header).map_err(SafeTensorError::InvalidHeaderDeserialization)?;
    let data = length + 8;
    if file.metadata()?.len() != data + metadata.data_len() as u64 {
        return Err(SafeTensorError::MetadataIncompleteBuffer.into());
    }
    stored(&metadata, data)
}

// Reads the header of safetensors `bytes`.
pub fn index_bytes(bytes: &[u8]) -> Result<Vec<(String, StoredTensor)>> {
    let (header, metadata) = SafeTensors::read_metadata(bytes)?;
    stored(&metadata, header as u64 + 8)
}

// The tensors `metadata` lists, whose data starts at byte `data` of the file.
fn stored(metadata: &Metadata, data: u64) -> Result<Vec<(String, StoredTensor)>> {
    metadata
        .offset_keys()
        .into_iter()
        .map(|name| {
            let info = metadata
                .info(&name)
                .ok_or_else(|| Error::missing_input(format!("checkpoint tensor `{name}`")))?;
            let shape = Shape::create(
                info.shape
                    .iter()
                    .map(|&dimension| to_i64(dimension, "checkpoint dimension"))
                    .collect::<Result<Vec<_>>>()?,
            )?;
            let (start, end) = info.data_offsets;
            let stored = StoredTensor {
                data_type: data_type(info.dtype)?,
                shape,
                bytes: data + start as u64..data + end as u64,
            };
            Ok((name, stored))
        })
        .collect()
}

// Serializes `tensors` under their names.
pub fn serialize<'a>(tensors: impl IntoIterator<Item = (&'a str, &'a Tensor)>) -> Result<Vec<u8>> {
    let owned = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let dimensions = tensor
                .shape()
                .dimensions()
                .iter()
                .map(|&dimension| to_usize(dimension, "checkpoint dimension"))
                .collect::<Result<Vec<_>>>()?;
            Ok((
                name,
                dtype(tensor.data_type())?,
                dimensions,
                tensor.to_bytes()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let views = owned
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            Ok((*name, TensorView::new(*dtype, shape.clone(), bytes)?))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(tensor::serialize(views, None)?)
}

fn data_type(dtype: Dtype) -> Result<DataType> {
    Ok(match dtype {
        Dtype::BOOL => DataType::Bool,
        Dtype::U8 => DataType::U8,
        Dtype::I8 => DataType::I8,
        Dtype::U16 => DataType::U16,
        Dtype::I16 => DataType::I16,
        Dtype::U32 => DataType::U32,
        Dtype::I32 => DataType::I32,
        Dtype::U64 => DataType::U64,
        Dtype::I64 => DataType::I64,
        Dtype::F16 => DataType::F16,
        Dtype::BF16 => DataType::Bf16,
        Dtype::F32 => DataType::F32,
        Dtype::F64 => DataType::F64,
        Dtype::F8_E4M3 => DataType::F8E4M3,
        Dtype::F8_E5M2 => DataType::F8E5M2,
        other => {
            return Err(Error::unsupported_operation_with_details(
                "checkpoint dtype",
                format!("{other:?}"),
            ));
        }
    })
}

fn dtype(data_type: DataType) -> Result<Dtype> {
    Ok(match data_type {
        DataType::Bool => Dtype::BOOL,
        DataType::U8 => Dtype::U8,
        DataType::I8 => Dtype::I8,
        DataType::U16 => Dtype::U16,
        DataType::I16 => Dtype::I16,
        DataType::U32 => Dtype::U32,
        DataType::I32 => Dtype::I32,
        DataType::U64 => Dtype::U64,
        DataType::I64 => Dtype::I64,
        DataType::F16 => Dtype::F16,
        DataType::Bf16 => Dtype::BF16,
        DataType::F32 => Dtype::F32,
        DataType::F64 => Dtype::F64,
        DataType::F8E4M3 => Dtype::F8_E4M3,
        DataType::F8E5M2 => Dtype::F8_E5M2,
        DataType::F4E2M1 => {
            return Err(Error::unsupported_operation(
                "checkpoint low precision dtype",
            ));
        }
    })
}
