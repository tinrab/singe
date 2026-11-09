pub(crate) mod buffer;
#[cfg(feature = "testing")]
pub(crate) mod profiling;

use std::{
    fmt::{self, Debug, Formatter},
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};

#[cfg(feature = "cuda")]
use singe_cuda::memory::PinnedHostMemory;

#[cfg(feature = "testing")]
pub use self::profiling::CpuStorageCopyStats;
use crate::{
    activity,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    observation::check_storage_readback,
    storage::{
        StorageVersion,
        cpu::buffer::{CpuBuffer, HostBytes},
    },
    tensor::{TensorDataType, validate_boolean_bytes},
};

/// Untyped host storage behind a read-write lock.
///
/// Tensors that share the allocation can be written through `assign` while other threads read it.
pub struct CpuStorage {
    buffer: RwLock<CpuBuffer>,
    byte_size: usize,
    version: StorageVersion,
}

impl CpuStorage {
    pub(crate) fn from_buffer(buffer: CpuBuffer) -> Self {
        activity::storage_request();
        Self::wrap(buffer)
    }

    pub(crate) fn wrap(buffer: CpuBuffer) -> Self {
        Self {
            byte_size: buffer.as_bytes().len(),
            buffer: RwLock::new(buffer),
            version: StorageVersion::default(),
        }
    }

    /// Takes ownership of `values` without copying them.
    pub(crate) fn from_values<T: Copy>(values: Vec<T>) -> Self {
        Self::from_buffer(CpuBuffer::Heap(HostBytes::from_vec(values)))
    }

    /// Allocates zero-filled CPU storage for `element_count` elements of `data_type`.
    ///
    /// # Errors
    ///
    /// Returns an error if `element_count` times the element byte size overflows `usize`.
    pub fn create(data_type: DataType, element_count: usize) -> Result<Self> {
        activity::storage_request();
        let byte_size = element_count
            .checked_mul(data_type.size_bytes())
            .ok_or_else(|| Error::out_of_range("cpu storage byte size"))?;
        Ok(Self::wrap(CpuBuffer::zeroed(byte_size)))
    }

    /// Allocates zero-filled CPU storage, rejecting other devices.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not CPU or the requested byte count overflows `usize`.
    pub fn create_zeroed(
        device: Device,
        data_type: DataType,
        element_count: usize,
    ) -> Result<Self> {
        validate_cpu_device(device)?;
        Self::create(data_type, element_count)
    }

    /// Imports native-endian bytes holding whole `data_type` elements.
    ///
    /// Boolean bytes must be 0 or 1.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not CPU, the byte length is not a whole number of elements, or a boolean byte is neither zero nor one.
    pub fn create_from_bytes(device: Device, data_type: DataType, data: &[u8]) -> Result<Self> {
        validate_cpu_device(device)?;
        Self::from_bytes(data_type, data)
    }

    pub(crate) fn from_bytes(data_type: DataType, data: impl AsRef<[u8]>) -> Result<Self> {
        activity::storage_request();
        let data = data.as_ref();
        validate_elements(data_type, data, "cpu storage bytes")?;
        activity::cpu_import(data.len());
        Ok(Self::wrap(CpuBuffer::from_bytes(data)))
    }

    /// Wraps page-locked bytes holding whole `data_type` elements, such as those `Tensor::pin_memory` allocates.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length is not a whole number of elements or a boolean byte is neither 0 nor 1.
    #[cfg(feature = "cuda")]
    pub fn create_pinned(data_type: DataType, bytes: PinnedHostMemory<u8>) -> Result<Self> {
        validate_elements(data_type, bytes.as_slice(), "cpu pinned storage bytes")?;
        activity::storage_request();
        Ok(Self::wrap(CpuBuffer::Pinned(bytes)))
    }

    /// Returns whether the allocation is page-locked, so CUDA copies read and write it directly.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.read().is_ok_and(|buffer| buffer.is_pinned())
    }

    /// Returns the initialized storage span in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Returns retained allocation capacity, including spare host vector capacity.
    #[must_use]
    pub fn allocation_byte_size(&self) -> usize {
        self.read()
            .map_or(self.byte_size, |buffer| buffer.allocation_byte_size())
    }

    /// Returns the number of writes applied to this allocation since it was created.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version.get()
    }

    /// Records a completed write to this allocation.
    pub fn bump_version(&self) {
        self.version.bump();
    }

    /// Returns a process-local identity for this storage owner.
    #[must_use]
    pub fn storage_identity(&self) -> usize {
        ptr::from_ref(self).addr()
    }

    /// Returns a process-local identity for the allocation data pointer.
    #[must_use]
    pub fn data_pointer_identity(&self) -> Option<usize> {
        self.as_ptr().ok().map(<*const u8>::addr)
    }

    /// Copies the allocation's bytes into `dst`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, capture rules prohibit reading this allocation, or a writer panicked while holding its lock.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() != self.byte_size {
            return Err(Error::length_mismatch(
                "cpu copy_to_host destination",
                self.byte_size,
                dst.len(),
            ));
        }

        dst.copy_from_slice(&self.as_slice()?);
        activity::cpu_export(dst.len());
        Ok(())
    }

    /// Copies this allocation into new host storage.
    ///
    /// # Errors
    ///
    /// Returns an error if capture rules prohibit reading this allocation or a writer panicked while holding its lock.
    pub fn clone_on_host(&self) -> Result<Self> {
        check_storage_readback(self.storage_identity())?;
        activity::cpu_clone(self.byte_size());
        activity::storage_request();
        Ok(Self::wrap(self.read()?.clone()))
    }

    /// Fills the allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if a writer panicked while holding the allocation's lock.
    pub fn fill_zeroes(&mut self) -> Result<()> {
        self.buffer
            .get_mut()
            .map_err(|_| Error::poisoned("cpu storage"))?
            .fill_zeroes();
        Ok(())
    }
}

/// Checks that `data` holds whole `data_type` elements, and that boolean bytes are 0 or 1.
fn validate_elements(data_type: DataType, data: &[u8], name: &'static str) -> Result<()> {
    if !data.len().is_multiple_of(data_type.size_bytes()) {
        return Err(Error::byte_length_mismatch(
            name,
            data_type.size_bytes(),
            data.len(),
        ));
    }
    validate_boolean_bytes(data_type, data)
}

fn validate_cpu_device(device: Device) -> Result<()> {
    if !device.is_cpu() {
        return Err(Error::device_mismatch("device", Device::Cpu, device));
    }
    Ok(())
}

impl Debug for CpuStorage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuStorage")
            .field("byte_size", &self.byte_size)
            .finish_non_exhaustive()
    }
}

/// Native-endian bytes of a CPU allocation, borrowed under its read lock.
#[derive(Debug)]
pub struct CpuBytes<'a> {
    guard: RwLockReadGuard<'a, CpuBuffer>,
}

impl Deref for CpuBytes<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.guard.as_bytes()
    }
}

/// Native-endian bytes of a CPU allocation, borrowed under its write lock.
#[derive(Debug)]
pub struct CpuBytesMut<'a> {
    guard: RwLockWriteGuard<'a, CpuBuffer>,
}

impl Deref for CpuBytesMut<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.guard.as_bytes()
    }
}

impl DerefMut for CpuBytesMut<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.guard.as_bytes_mut()
    }
}

/// Typed elements of a CPU allocation, borrowed under its read lock.
#[derive(Debug)]
pub struct CpuSlice<'a, T> {
    _guard: RwLockReadGuard<'a, CpuBuffer>,
    values: NonNull<[T]>,
}

// SAFETY: a `CpuSlice` only reads the buffer its read guard locks, so sharing it between threads is as safe as sharing `&[T]`.
unsafe impl<T: Sync> Sync for CpuSlice<'_, T> {}

impl<T> Deref for CpuSlice<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `values` points into the buffer that `_guard` keeps read-locked, so the allocation is neither written, resized, nor freed while it lives.
        unsafe { self.values.as_ref() }
    }
}

impl CpuStorage {
    pub(crate) fn read(&self) -> Result<RwLockReadGuard<'_, CpuBuffer>> {
        self.buffer
            .read()
            .map_err(|_| Error::poisoned("cpu storage"))
    }

    /// Returns a pointer to the allocation's native-endian bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if a writer panicked while holding the allocation's lock.
    pub fn as_ptr(&self) -> Result<*const u8> {
        Ok(self.read()?.as_bytes().as_ptr())
    }

    /// Borrows the allocation's native-endian bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if capture rules prohibit reading this tracked allocation, or a writer panicked while holding its lock.
    pub fn as_slice(&self) -> Result<CpuBytes<'_>> {
        check_storage_readback(self.storage_identity())?;
        Ok(CpuBytes {
            guard: self.read()?,
        })
    }

    /// Borrows the bytes as `T` elements without copying.
    ///
    /// # Errors
    ///
    /// Returns an error if capture rules prohibit reading this tracked allocation, a writer panicked while holding its lock, or the bytes are not a whole, aligned run of valid `T` values.
    pub fn as_typed_slice<T: TensorDataType>(&self) -> Result<CpuSlice<'_, T>> {
        check_storage_readback(self.storage_identity())?;
        let guard = self.read()?;
        let values = NonNull::from(T::from_bytes(guard.as_bytes())?);
        Ok(CpuSlice {
            _guard: guard,
            values,
        })
    }

    /// Borrows the allocation's bytes for writing, blocking until no other borrow is active.
    ///
    /// Writers must keep every element valid for the stored data type.
    ///
    /// # Errors
    ///
    /// Returns an error if a writer panicked while holding the allocation's lock.
    pub(crate) fn write_bytes(&self) -> Result<CpuBytesMut<'_>> {
        let guard = self
            .buffer
            .write()
            .map_err(|_| Error::poisoned("cpu storage"))?;
        Ok(CpuBytesMut { guard })
    }

    /// Mutably borrows the bytes as `T` elements before the allocation is shared by tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if a writer panicked while holding the allocation's lock, or the bytes are not a whole, aligned run of valid `T` values.
    pub fn as_typed_slice_mut<T: TensorDataType>(&mut self) -> Result<&mut [T]> {
        T::from_bytes_mut(
            self.buffer
                .get_mut()
                .map_err(|_| Error::poisoned("cpu storage"))?
                .as_bytes_mut(),
        )
    }
}

#[cfg(test)]
mod tests;
