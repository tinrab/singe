/// CPU storage implementation.
pub mod cpu;

/// CUDA storage implementation.
#[cfg(feature = "cuda")]
pub mod cuda;

/// Meta device storage, which describes allocations without making them.
pub mod meta;

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use crate::storage::cuda::CudaStorage;
use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    storage::{cpu::CpuStorage, meta::MetaStorage},
};

/// Untyped device allocation backing one or more tensors.
///
/// Storage owns raw bytes and the device identity.
/// Tensors own their data type, logical shape, strides, and view offsets, so tensors of different data types may view one allocation.
#[derive(Debug)]
pub enum Storage {
    /// Host memory.
    Cpu(CpuStorage),
    /// CUDA device memory.
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    /// Meta device bytes, described but not allocated.
    Meta(MetaStorage),
}

impl Storage {
    /// Returns the device that owns this allocation.
    #[must_use]
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.device(),
            Self::Meta(_) => Device::Meta,
        }
    }

    /// Returns the initialized, addressable storage span in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Cpu(storage) => storage.byte_size(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.byte_size(),
            Self::Meta(storage) => storage.byte_size(),
        }
    }

    /// Returns retained allocation capacity, including spare host capacity.
    /// Layout validation uses `byte_size` and never reads uninitialized capacity.
    #[must_use]
    pub fn allocation_byte_size(&self) -> usize {
        match self {
            Self::Cpu(storage) => storage.allocation_byte_size(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.byte_size(),
            Self::Meta(storage) => storage.value().map_or(0, CpuStorage::allocation_byte_size),
        }
    }

    /// Returns whether the allocation holds zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.byte_size() == 0
    }

    /// Returns an identity for this storage owner object.
    ///
    /// This is process-local material intended for compiled-executable guards and cache keys, not serialization.
    #[must_use]
    pub fn storage_identity(&self) -> usize {
        match self {
            Self::Cpu(storage) => storage.storage_identity(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.storage_identity(),
            Self::Meta(storage) => storage.storage_identity(),
        }
    }

    /// Returns an identity for the allocation data pointer, when one is available.
    ///
    /// This is process-local material intended for pointer-stability guards.
    #[must_use]
    pub fn data_pointer_identity(&self) -> Option<usize> {
        match self {
            Self::Cpu(storage) => storage.data_pointer_identity(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.data_pointer_identity(),
            Self::Meta(_) => None,
        }
    }

    /// Returns the number of writes applied to this allocation since it was created.
    ///
    /// Every tensor that shares the allocation observes the same version.
    #[must_use]
    pub fn version(&self) -> u64 {
        match self {
            Self::Cpu(storage) => storage.version(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.version(),
            Self::Meta(storage) => storage.version(),
        }
    }

    /// Records a completed write to this allocation.
    pub fn bump_version(&self) {
        match self {
            Self::Cpu(storage) => storage.bump_version(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.bump_version(),
            Self::Meta(storage) => storage.bump_version(),
        }
    }

    /// Allocates zero-initialized storage for `element_count` elements of `data_type` on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is unsupported or the requested byte size overflows or cannot be allocated.
    pub fn create_zeroed(
        device: Device,
        data_type: DataType,
        element_count: usize,
    ) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuStorage::create(data_type, element_count)?)),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Ok(Self::Cuda(CudaStorage::create_zeroed(
                device,
                data_type,
                element_count,
            )?)),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::unsupported_device(
                "cuda storage requires the cuda feature",
            )),
            Device::Meta => Ok(Self::Meta(MetaStorage::new(
                element_count
                    .checked_mul(data_type.size_bytes())
                    .ok_or_else(|| Error::out_of_range("meta storage byte size"))?,
            ))),
        }
    }

    /// Copies host bytes holding whole `data_type` elements into new storage on `device`.
    /// Meta storage keeps only their length.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` does not contain a whole number of valid `data_type` elements, the device is unsupported, or allocation or copying fails.
    pub fn create_from_bytes(device: Device, data_type: DataType, data: &[u8]) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuStorage::create_from_bytes(
                device, data_type, data,
            )?)),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Ok(Self::Cuda(CudaStorage::create_from_bytes(
                device, data_type, data,
            )?)),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::unsupported_device(
                "cuda storage requires the cuda feature",
            )),
            Device::Meta => Ok(Self::Meta(MetaStorage::new(data.len()))),
        }
    }

    /// Copies storage contents to host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if this is meta storage without a retained value, `dst` has the wrong length, or the device copy fails.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        match self {
            Self::Cpu(storage) => storage.copy_to_host(dst),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.copy_to_host(dst),
            Self::Meta(storage) => storage
                .value()
                .ok_or(Error::MetaTensorData)?
                .copy_to_host(dst),
        }
    }

    /// Clones this allocation to `device`.
    /// Moving to the meta device keeps only the byte size; moving meta storage off it needs a retained value.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination device is unsupported, this is meta storage without a retained value and `device` is not meta, or the copy fails.
    pub fn clone_to_device(&self, device: Device) -> Result<Self> {
        match (self, device) {
            (Self::Meta(storage), Device::Meta) => Ok(Self::Meta(storage.duplicate()?)),
            (_, Device::Meta) => Ok(Self::Meta(MetaStorage::new(self.byte_size()))),
            (Self::Meta(storage), _) => {
                let host = Self::Cpu(
                    storage
                        .value()
                        .ok_or(Error::MetaTensorData)?
                        .clone_on_host()?,
                );
                if device.is_cpu() {
                    Ok(host)
                } else {
                    host.clone_to_device(device)
                }
            }
            (Self::Cpu(storage), Device::Cpu) => Ok(Self::Cpu(storage.clone_on_host()?)),
            #[cfg(feature = "cuda")]
            (Self::Cuda(storage), Device::Cuda(_)) => {
                Ok(Self::Cuda(storage.clone_to_device(device)?))
            }
            _ => {
                let mut bytes = vec![0_u8; self.byte_size()];
                self.copy_to_host(&mut bytes)?;
                Self::create_from_bytes(device, DataType::U8, &bytes)
            }
        }
    }

    /// Fills the allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend zero-fill operation fails.
    pub fn fill_zeroes(&mut self) -> Result<()> {
        match self {
            Self::Cpu(storage) => storage.fill_zeroes(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.fill_zeroes(),
            Self::Meta(_) => Ok(()),
        }
    }

    /// Returns the host allocation, when this storage is on the CPU.
    #[must_use]
    pub fn as_cpu(&self) -> Option<&CpuStorage> {
        match self {
            Self::Cpu(storage) => Some(storage),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => None,
            Self::Meta(_) => None,
        }
    }

    /// Returns the CUDA allocation, when this storage is on a CUDA device.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn as_cuda(&self) -> Option<&CudaStorage> {
        match self {
            Self::Cuda(storage) => Some(storage),
            Self::Cpu(_) | Self::Meta(_) => None,
        }
    }
}

impl From<CpuStorage> for Storage {
    fn from(storage: CpuStorage) -> Self {
        Self::Cpu(storage)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaStorage> for Storage {
    fn from(storage: CudaStorage) -> Self {
        Self::Cuda(storage)
    }
}

impl From<MetaStorage> for Storage {
    fn from(storage: MetaStorage) -> Self {
        Self::Meta(storage)
    }
}

/// Write counter shared by every tensor that views one allocation.
#[derive(Debug, Default)]
pub struct StorageVersion(AtomicU64);

impl StorageVersion {
    /// Returns the current version.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    /// Records one completed write.
    pub fn bump(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests;
