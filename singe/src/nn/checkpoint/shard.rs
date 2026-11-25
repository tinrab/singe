#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::{collections::HashMap, fs::File, ops::Range, path::Path, slice};
#[cfg(windows)]
use std::{io::ErrorKind, os::windows::fs::FileExt};

use super::{flashpack, safetensors};
use crate::{
    data_type::{DataType, bf16, f16},
    device::Device,
    error::{Error, Result},
    parallel::{self, Parallelism},
    shape::Shape,
    tensor::{Tensor, TensorDataType},
    utility::checked_mul,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredTensor {
    pub data_type: DataType,
    pub shape: Shape,
    pub bytes: Range<u64>,
}

// Where a shard's bytes are: in memory already, or in a file read a range at a time.
enum Source<'a> {
    Bytes(&'a [u8]),
    File(File),
}

pub struct Shard<'a> {
    source: Source<'a>,
    tensors: Vec<(String, StoredTensor)>,
    positions: HashMap<String, usize>,
}

impl<'a> Shard<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        Self::create(Source::Bytes(bytes), safetensors::index_bytes(bytes)?)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let tensors = if flashpack::holds(&file)? {
            flashpack::index(&file)?
        } else {
            safetensors::index(&file)?
        };
        Self::create(Source::File(file), tensors)
    }

    fn create(source: Source<'a>, mut tensors: Vec<(String, StoredTensor)>) -> Result<Self> {
        tensors.sort_by_key(|(_, stored)| stored.bytes.start);
        let mut positions = HashMap::with_capacity(tensors.len());
        for (position, (name, stored)) in tensors.iter().enumerate() {
            // Reads and device views take a tensor's extent from its shape, so its bytes must be exactly that long.
            let length = checked_mul(
                stored.shape.element_count(),
                stored.data_type.size_bytes(),
                "checkpoint tensor bytes",
            )?;
            if stored.bytes.end.checked_sub(stored.bytes.start) != Some(length as u64) {
                return Err(Error::invalid_argument(format!(
                    "checkpoint tensor `{name}` stores bytes its shape does not hold"
                )));
            }

            if positions.insert(name.clone(), position).is_some() {
                return Err(Error::invalid_argument(format!(
                    "checkpoint tensor `{name}` stored twice"
                )));
            }
        }
        Ok(Self {
            source,
            tensors,
            positions,
        })
    }

    // The shard's tensor names, in the order their bytes lie.
    pub fn names(&self) -> Vec<String> {
        self.tensors.iter().map(|(name, _)| name.clone()).collect()
    }

    pub fn stored(&self, name: &str) -> Result<StoredTensor> {
        self.positions
            .get(name)
            .map(|&position| self.tensors[position].1.clone())
            .ok_or_else(|| Error::missing_input(format!("checkpoint tensor `{name}`")))
    }

    pub fn read_as(
        &self,
        stored: &StoredTensor,
        target: DataType,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<()> {
        self.read_with(stored, target, offset, buffer, Parallelism::Pool)
    }

    pub fn read_as_serial(
        &self,
        stored: &StoredTensor,
        target: DataType,
        offset: usize,
        buffer: &mut [u8],
    ) -> Result<()> {
        self.read_with(stored, target, offset, buffer, Parallelism::Serial)
    }

    fn read_with(
        &self,
        stored: &StoredTensor,
        target: DataType,
        offset: usize,
        buffer: &mut [u8],
        parallelism: Parallelism,
    ) -> Result<()> {
        if stored.data_type == target {
            return self.read_at(stored, offset, buffer, parallelism);
        }

        let (target_size, source_size) = (target.size_bytes(), stored.data_type.size_bytes());
        let first = offset / target_size;
        let count = buffer.len() / target_size;
        let mut bytes =
            AlignedBytes::zeroed(checked_mul(count, source_size, "checkpoint chunk bytes")?);
        self.read_at(stored, first * source_size, bytes.as_mut(), parallelism)?;
        cast_floats(stored.data_type, bytes.as_ref(), target, buffer)
    }

    pub fn tensor_as(&self, stored: &StoredTensor, target: DataType) -> Result<Tensor> {
        let mut bytes = AlignedBytes::zeroed(checked_mul(
            stored.shape.element_count(),
            target.size_bytes(),
            "checkpoint tensor bytes",
        )?);
        self.read_as(stored, target, 0, bytes.as_mut())?;
        Tensor::from_bytes(Device::Cpu, target, &stored.shape, bytes.as_ref())
    }

    fn read_at(
        &self,
        stored: &StoredTensor,
        offset: usize,
        buffer: &mut [u8],
        parallelism: Parallelism,
    ) -> Result<()> {
        let from = stored.bytes.start + offset as u64;
        if from + buffer.len() as u64 > stored.bytes.end {
            return Err(Error::out_of_range("checkpoint tensor read"));
        }

        match &self.source {
            Source::Bytes(bytes) => {
                let from = to_usize_u64(from)?;
                buffer.copy_from_slice(&bytes[from..from + buffer.len()]);
            }
            Source::File(file) => read_file_at(file, from, buffer, parallelism)?,
        }
        Ok(())
    }
}

// The bytes each thread reads of a larger range.
const READ_PART: usize = 1 << 20;

fn read_file_at(file: &File, at: u64, buffer: &mut [u8], parallelism: Parallelism) -> Result<()> {
    parallel::for_each_chunk_mut(buffer, READ_PART, parallelism, |index, part| {
        read_exact_at(file, at + (index * READ_PART) as u64, part)
    })
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

pub fn to_usize_u64(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::out_of_range("checkpoint offset"))
}

#[cfg(unix)]
pub fn read_exact_at(file: &File, at: u64, buffer: &mut [u8]) -> Result<()> {
    Ok(file.read_exact_at(buffer, at)?)
}

#[cfg(windows)]
pub fn read_exact_at(file: &File, mut at: u64, mut buffer: &mut [u8]) -> Result<()> {
    while !buffer.is_empty() {
        let read = file.seek_read(buffer, at)?;
        if read == 0 {
            return Err(std::io::Error::from(ErrorKind::UnexpectedEof).into());
        }

        buffer = &mut buffer[read..];
        at += read as u64;
    }
    Ok(())
}
