// The flashpack format of fal's `flashpack` package: the payload from byte 0, one macroblock per data type, then compact JSON metadata, its length as a little-endian u64, and the magic `FLASHPK\0`.
// Each macroblock starts on a multiple of lcm(`align_bytes`, element size), and each tensor in one on a multiple of `align_bytes` bytes; index offsets and lengths count elements of the tensor's macroblock.
// `flashpack_v3` holds one data type and names it `target_dtype`; `flashpack_v4` lists its macroblocks.

use std::{
    cmp::Reverse,
    ffi::OsString,
    fs::{self, File},
    io::{BufWriter, IntoInnerError, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::shard::{StoredTensor, read_exact_at, to_usize_u64};
use crate::{
    data_type::DataType,
    error::{FlashpackError, Result},
    shape::Shape,
    tensor::Tensor,
    utility::{checked_mul_u64, to_i64, to_u64},
};

const MAGIC: &[u8; 8] = b"FLASHPK\0";

const FOOTER: u64 = 16;

const ALIGN_BYTES: u64 = 128;
const ONE_TYPE: &str = "flashpack_v3";
const TYPES: &str = "flashpack_v4";

#[derive(Serialize, Deserialize)]
struct Metadata {
    format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_dtype: Option<String>,
    align_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    total_payload_bytes: Option<u64>,
    total_elems: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    macroblocks: Vec<Macroblock>,
    index: Vec<Entry>,
}

#[derive(Serialize, Deserialize)]
struct Macroblock {
    dtype: String,
    offset_bytes: u64,
    length_bytes: u64,
    length_elems: u64,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    name: String,
    shape: Vec<u64>,
    offset: u64,
    length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    macroblock: Option<usize>,
}

pub fn holds(file: &File) -> Result<bool> {
    let length = file.metadata()?.len();
    if length < FOOTER {
        return Ok(false);
    }

    let mut magic = [0; 8];
    read_exact_at(file, length - 8, &mut magic)?;
    Ok(&magic == MAGIC)
}

pub fn index(file: &File) -> Result<Vec<(String, StoredTensor)>> {
    let footer = file
        .metadata()?
        .len()
        .checked_sub(FOOTER)
        .ok_or(FlashpackError::MetadataLength { length: 0 })?;
    let mut length = [0; 8];
    read_exact_at(file, footer, &mut length)?;
    let length = u64::from_le_bytes(length);
    let payload = footer
        .checked_sub(length)
        .ok_or(FlashpackError::MetadataLength { length })?;
    let mut text = vec![0; to_usize_u64(length)?];
    read_exact_at(file, payload, &mut text)?;
    let metadata: Metadata = serde_json::from_slice(&text).map_err(FlashpackError::Metadata)?;

    let one_type = metadata.format == ONE_TYPE;
    let blocks = match metadata.format.as_str() {
        ONE_TYPE => {
            let dtype = metadata
                .target_dtype
                .ok_or(FlashpackError::MissingField("target_dtype"))?;
            let data_type = data_type(&dtype)?;
            vec![(
                data_type,
                0,
                checked_mul_u64(
                    metadata.total_elems,
                    data_type.size_bytes() as u64,
                    "flashpack payload bytes",
                )?,
                metadata.total_elems,
            )]
        }
        TYPES => metadata
            .macroblocks
            .iter()
            .map(|block| {
                Ok((
                    data_type(&block.dtype)?,
                    block.offset_bytes,
                    block.length_bytes,
                    block.length_elems,
                ))
            })
            .collect::<Result<Vec<_>>>()?,
        other => return Err(FlashpackError::Format(other.to_owned()).into()),
    };
    for (index, &(data_type, offset, bytes, elements)) in blocks.iter().enumerate() {
        let end = offset.checked_add(bytes);
        if end.is_none_or(|end| end > payload)
            || elements.checked_mul(data_type.size_bytes() as u64) != Some(bytes)
        {
            return Err(FlashpackError::Macroblock { index }.into());
        }
    }

    metadata
        .index
        .into_iter()
        .map(|entry| {
            let block = match entry.macroblock {
                Some(block) => block,
                None if one_type => 0,
                None => return Err(FlashpackError::MissingField("macroblock").into()),
            };
            let &(data_type, offset, _, elements) = blocks
                .get(block)
                .ok_or_else(|| FlashpackError::Range(entry.name.clone()))?;
            let dimensions = entry
                .shape
                .iter()
                .map(|&dimension| to_i64(dimension, "flashpack dimension"))
                .collect::<Result<Vec<_>>>()?;
            let shape = Shape::create(dimensions)?;
            if shape.element_count() as u64 != entry.length {
                return Err(FlashpackError::Length {
                    name: entry.name,
                    length: entry.length,
                }
                .into());
            }

            if entry
                .offset
                .checked_add(entry.length)
                .is_none_or(|end| end > elements)
            {
                return Err(FlashpackError::Range(entry.name).into());
            }

            let size = data_type.size_bytes() as u64;
            let start = offset + entry.offset * size;
            let stored = StoredTensor {
                data_type,
                shape,
                bytes: start..start + entry.length * size,
            };
            Ok((entry.name, stored))
        })
        .collect()
}

pub fn write<'a>(
    path: &Path,
    tensors: impl IntoIterator<Item = (&'a str, &'a Tensor)>,
) -> Result<()> {
    let mut tensors = tensors.into_iter().collect::<Vec<_>>();
    if tensors.is_empty() {
        return Err(FlashpackError::Empty.into());
    }

    tensors.sort_by_key(|(_, tensor)| Reverse(tensor.element_count()));
    let mut types = Vec::<(DataType, Vec<(&str, &Tensor)>)>::new();
    for (name, tensor) in tensors {
        let data_type = tensor.data_type();
        match types.iter_mut().find(|(other, _)| *other == data_type) {
            Some((_, members)) => members.push((name, tensor)),
            None => types.push((data_type, vec![(name, tensor)])),
        }
    }

    let mut macroblocks = Vec::with_capacity(types.len());
    let mut index = Vec::new();
    let mut cursor = 0_u64;
    for (block, (data_type, members)) in types.iter().enumerate() {
        let size = data_type.size_bytes() as u64;
        cursor = cursor.next_multiple_of(lcm(ALIGN_BYTES, size));
        let align_elements = ALIGN_BYTES / gcd(ALIGN_BYTES, size);
        let mut elements = 0_u64;
        for (name, tensor) in members {
            elements = elements.next_multiple_of(align_elements);
            let length = tensor.element_count() as u64;
            index.push(Entry {
                name: (*name).to_owned(),
                shape: tensor
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|&dimension| to_u64(dimension, "flashpack dimension"))
                    .collect::<Result<Vec<_>>>()?,
                offset: elements,
                length,
                macroblock: Some(block),
            });
            elements += length;
        }

        macroblocks.push(Macroblock {
            dtype: name(*data_type)?.to_owned(),
            offset_bytes: cursor,
            length_bytes: elements * size,
            length_elems: elements,
        });
        cursor += elements * size;
    }

    let total_elems = macroblocks.iter().map(|block| block.length_elems).sum();
    let metadata = if let [block] = macroblocks.as_slice() {
        Metadata {
            format: ONE_TYPE.to_owned(),
            target_dtype: Some(block.dtype.clone()),
            align_bytes: ALIGN_BYTES,
            total_payload_bytes: None,
            total_elems,
            macroblocks: Vec::new(),
            index: index
                .into_iter()
                .map(|entry| Entry {
                    macroblock: None,
                    ..entry
                })
                .collect(),
        }
    } else {
        Metadata {
            format: TYPES.to_owned(),
            target_dtype: None,
            align_bytes: ALIGN_BYTES,
            total_payload_bytes: Some(cursor),
            total_elems,
            macroblocks,
            index,
        }
    };
    let text = serde_json::to_vec(&metadata).map_err(FlashpackError::Metadata)?;

    let partial = partial_path(path)?;
    let written = write_file(&partial, &types, &metadata, &text);
    if written.is_err() {
        // The partial file is only ever this write's, so a failed write leaves nothing behind.
        let _ = fs::remove_file(&partial);
    }

    written?;
    fs::rename(&partial, path)?;
    Ok(())
}

fn write_file(
    path: &Path,
    types: &[(DataType, Vec<(&str, &Tensor)>)],
    metadata: &Metadata,
    text: &[u8],
) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);
    let blocks = types.iter().flat_map(|(_, members)| members);
    let offsets = metadata.index.iter().map(|entry| {
        let block = entry.macroblock.unwrap_or(0);
        let start = metadata
            .macroblocks
            .get(block)
            .map_or(0, |block| block.offset_bytes);
        let size = types[block].0.size_bytes() as u64;
        start + entry.offset * size
    });
    let mut cursor = 0_u64;
    for ((_, tensor), offset) in blocks.zip(offsets) {
        pad(&mut writer, offset - cursor)?;
        let bytes = tensor.to_bytes()?;
        writer.write_all(&bytes)?;
        cursor = offset + bytes.len() as u64;
    }

    writer.write_all(text)?;
    writer.write_all(&(text.len() as u64).to_le_bytes())?;
    writer.write_all(MAGIC)?;
    let file = writer.into_inner().map_err(IntoInnerError::into_error)?;
    file.sync_all()?;
    Ok(())
}

fn pad(writer: &mut impl Write, mut count: u64) -> Result<()> {
    const ZEROS: [u8; 4096] = [0; 4096];
    while count > 0 {
        let length = count.min(ZEROS.len() as u64);
        writer.write_all(&ZEROS[..to_usize_u64(length)?])?;
        count -= length;
    }
    Ok(())
}

fn partial_path(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| FlashpackError::Path(path.to_owned()))?;
    let mut partial = OsString::from(".");
    partial.push(name);
    partial.push(".partial");
    Ok(path.with_file_name(partial))
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

const fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

fn data_type(name: &str) -> Result<DataType> {
    Ok(match name {
        "bool" => DataType::Bool,
        "uint8" => DataType::U8,
        "int8" => DataType::I8,
        "uint16" => DataType::U16,
        "int16" => DataType::I16,
        "uint32" => DataType::U32,
        "int32" => DataType::I32,
        "uint64" => DataType::U64,
        "int64" => DataType::I64,
        "float16" => DataType::F16,
        "bfloat16" => DataType::Bf16,
        "float32" => DataType::F32,
        "float64" => DataType::F64,
        "float8_e4m3fn" => DataType::F8E4M3,
        "float8_e5m2" => DataType::F8E5M2,
        other => return Err(FlashpackError::DataType(other.to_owned()).into()),
    })
}

fn name(data_type: DataType) -> Result<&'static str> {
    Ok(match data_type {
        DataType::Bool => "bool",
        DataType::U8 => "uint8",
        DataType::I8 => "int8",
        DataType::U16 => "uint16",
        DataType::I16 => "int16",
        DataType::U32 => "uint32",
        DataType::I32 => "int32",
        DataType::U64 => "uint64",
        DataType::I64 => "int64",
        DataType::F16 => "float16",
        DataType::Bf16 => "bfloat16",
        DataType::F32 => "float32",
        DataType::F64 => "float64",
        DataType::F8E4M3 => "float8_e4m3fn",
        DataType::F8E5M2 => "float8_e5m2",
        DataType::F4E2M1 => return Err(FlashpackError::DataType("float4_e2m1fn".to_owned()).into()),
    })
}
