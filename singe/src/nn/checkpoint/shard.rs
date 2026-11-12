use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use safetensors::{
    Dtype, SafeTensorError, SafeTensors,
    tensor::{Metadata, TensorInfo},
};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
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

    // Fills `buffer` with the bytes of `info`'s tensor as `target`, from byte `offset` of it as `target`, casting what the file holds if it holds another type.
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
        let mut stored = vec![0; checked_mul(count, source_size, "checkpoint chunk bytes")?];
        self.read_at(info, first * source_size, &mut stored)?;
        let shape = Shape::create([to_i64(count, "checkpoint chunk elements")?])?;
        let cast = Tensor::from_bytes(Device::Cpu, source, &shape, &stored)?.cast(target)?;
        buffer.copy_from_slice(&cast.to_bytes()?);
        Ok(())
    }

    // Returns the whole of `info`'s tensor as `target` on the host.
    pub fn tensor_as(&mut self, info: &TensorInfo, target: DataType) -> Result<Tensor> {
        let shape = shape_of(info)?;
        let mut bytes = vec![
            0;
            checked_mul(
                shape.element_count(),
                target.size_bytes(),
                "checkpoint tensor bytes"
            )?
        ];
        self.read_as(info, target, 0, &mut bytes)?;
        Tensor::from_bytes(Device::Cpu, target, &shape, &bytes)
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
            Source::File(file) => {
                file.seek(SeekFrom::Start(self.data + from as u64))?;
                file.read_exact(buffer)?;
            }
        }
        Ok(())
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
