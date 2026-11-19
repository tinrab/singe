#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::{fs::File, io::Read, path::Path, slice};

use safetensors::{
    Dtype, SafeTensorError, SafeTensors,
    tensor::{Metadata, TensorInfo},
};

use crate::{
    data_type::{DataType, bf16, f16},
    device::Device,
    error::{Error, Result},
    parallel::{self, Parallelism},
    shape::Shape,
    tensor::{Tensor, TensorDataType},
    utility::{checked_mul, to_i64},
};

// The largest header a safetensors file may have, as the format allows.
const LARGEST_HEADER: u64 = 100_000_000;

// Where a shard's bytes are: in memory already, or in a file read a range at a time.
enum Source<'a> {
    Bytes(&'a [u8]),
    File(File),
}

// A safetensors file whose tensors are read where they lie, one range at a time, rather than held whole.
pub struct Shard<'a> {
    source: Source<'a>,
    // Where tensor data starts in the source.
    data: u64,
    metadata: Metadata,
}

impl<'a> Shard<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let (header, metadata) = SafeTensors::read_metadata(bytes)?;
        Ok(Self {
            source: Source::Bytes(bytes),
            data: header as u64 + 8,
            metadata,
        })
    }

    // Reads the header of the file at `path`, leaving its tensors where they are.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut length = [0; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        if length > LARGEST_HEADER {
            return Err(SafeTensorError::HeaderTooLarge.into());
        }

        let mut header = vec![0; to_usize_u64(length)?];
        file.read_exact(&mut header)?;
        let metadata: Metadata = serde_json::from_slice(&header)
            .map_err(SafeTensorError::InvalidHeaderDeserialization)?;
        let data = length + 8;
        if file.metadata()?.len() != data + metadata.data_len() as u64 {
            return Err(SafeTensorError::MetadataIncompleteBuffer.into());
        }

        Ok(Self {
            source: Source::File(file),
            data,
            metadata,
        })
    }

    // The shard's tensor names, in the order their bytes lie.
    pub fn names(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }

    pub fn info(&self, name: &str) -> Result<TensorInfo> {
        self.metadata
            .info(name)
            .cloned()
            .ok_or_else(|| Error::missing_input(format!("checkpoint tensor `{name}`")))
    }

    pub fn read_as(
        &mut self,
        info: &TensorInfo,
        target: DataType,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<()> {
        let source = stored_type(info.dtype)?;
        if source == target {
            return self.read_at(info, offset, buffer);
        }

        let (target_size, source_size) = (target.size_bytes(), source.size_bytes());
        let first = offset / target_size;
        let count = buffer.len() / target_size;
        let mut stored =
            AlignedBytes::zeroed(checked_mul(count, source_size, "checkpoint chunk bytes")?);
        self.read_at(info, first * source_size, stored.as_mut())?;
        cast_floats(source, stored.as_ref(), target, buffer)
    }

    // Returns the whole of `info`'s tensor as `target` on the host.
    pub fn tensor_as(&mut self, info: &TensorInfo, target: DataType) -> Result<Tensor> {
        let shape = shape_of(info)?;
        let mut bytes = AlignedBytes::zeroed(checked_mul(
            shape.element_count(),
            target.size_bytes(),
            "checkpoint tensor bytes",
        )?);
        self.read_as(info, target, 0, bytes.as_mut())?;
        Tensor::from_bytes(Device::Cpu, target, &shape, bytes.as_ref())
    }

    // Fills `buffer` with the stored bytes of `info`'s tensor from byte `offset` of them.
    fn read_at(&mut self, info: &TensorInfo, offset: usize, buffer: &mut [u8]) -> Result<()> {
        let (start, end) = info.data_offsets;
        let from = start + offset;
        if from + buffer.len() > end {
            return Err(Error::out_of_range("checkpoint tensor read"));
        }

        match &mut self.source {
            Source::Bytes(bytes) => {
                let from = to_usize_u64(self.data)? + from;
                buffer.copy_from_slice(&bytes[from..from + buffer.len()]);
            }
            Source::File(file) => read_file_at(file, self.data + from as u64, buffer)?,
        }
        Ok(())
    }
}

// The bytes each thread reads of a larger range.
const READ_PART: usize = 1 << 20;

// Fills `buffer` from `file` at byte `at`, in parts of `READ_PART` on all cores: a single thread copies from the page cache at about 3 GB/s, which was most of a warm load of Qwen3-0.6B.
#[cfg(unix)]
fn read_file_at(file: &File, at: u64, buffer: &mut [u8]) -> Result<()> {
    parallel::for_each_chunk_mut(buffer, READ_PART, Parallelism::Pool, |index, part| {
        Ok(file.read_exact_at(part, at + (index * READ_PART) as u64)?)
    })
}

#[cfg(not(unix))]
fn read_file_at(file: &mut File, at: u64, buffer: &mut [u8]) -> Result<()> {
    file.seek(SeekFrom::Start(at))?;
    Ok(file.read_exact(buffer)?)
}

pub struct AlignedBytes {
    words: Vec<u64>,
    length: usize,
}

impl AlignedBytes {
    pub fn zeroed(length: usize) -> Self {
        Self {
            words: vec![0; length.div_ceil(8)],
            length,
        }
    }
}

impl AsRef<[u8]> for AlignedBytes {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the words span at least `length` initialized bytes, and any byte is a valid `u8`.
        unsafe { slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.length) }
    }
}

impl AsMut<[u8]> for AlignedBytes {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: the words span at least `length` initialized bytes, and any bytes written make valid words.
        unsafe { slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.length) }
    }
}

fn cast_floats(source: DataType, stored: &[u8], target: DataType, output: &mut [u8]) -> Result<()> {
    fn into<S: TensorDataType>(values: &[S], target: DataType, output: &mut [u8]) -> Result<()> {
        fn write<T: TensorDataType>(
            values: &[impl TensorDataType],
            output: &mut [u8],
        ) -> Result<()> {
            for (slot, value) in T::from_bytes_mut(output)?.iter_mut().zip(values) {
                *slot = T::from_scalar(value.into_scalar());
            }
            Ok(())
        }

        match target {
            DataType::F16 => write::<f16>(values, output),
            DataType::Bf16 => write::<bf16>(values, output),
            DataType::F32 => write::<f32>(values, output),
            DataType::F64 => write::<f64>(values, output),
            _ => Err(Error::unsupported_data_type("checkpoint cast target")),
        }
    }

    match source {
        DataType::F16 => into(f16::from_bytes(stored)?, target, output),
        DataType::Bf16 => into(bf16::from_bytes(stored)?, target, output),
        DataType::F32 => into(f32::from_bytes(stored)?, target, output),
        DataType::F64 => into(f64::from_bytes(stored)?, target, output),
        _ => Err(Error::unsupported_data_type("checkpoint cast source")),
    }
}

pub fn shape_of(info: &TensorInfo) -> Result<Shape> {
    Shape::create(
        info.shape
            .iter()
            .map(|&dimension| to_i64(dimension, "checkpoint dimension"))
            .collect::<Result<Vec<_>>>()?,
    )
}

fn to_usize_u64(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::out_of_range("checkpoint offset"))
}

pub fn stored_type(dtype: Dtype) -> Result<DataType> {
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
        other => {
            return Err(Error::unsupported_operation_with_details(
                "checkpoint dtype",
                format!("{other:?}"),
            ));
        }
    })
}
