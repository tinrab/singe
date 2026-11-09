use std::ptr;

use crate::{
    activity,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    observation::check_storage_readback,
    storage::cuda::{CudaStorage, cuda_device_id, map_cuda_allocation_error},
};

impl CudaStorage {
    /// Returns the CUDA device that owns this allocation.
    #[must_use]
    pub fn device(&self) -> Device {
        Device::Cuda(self.context().device().id())
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
        Some(self.as_ptr().addr())
    }

    /// Allocates zero-filled storage for `element_count` elements of `data_type` on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device, the byte size overflows, or allocation fails.
    pub fn create_zeroed(
        device: Device,
        data_type: DataType,
        element_count: usize,
    ) -> Result<Self> {
        let shape = i64::try_from(element_count).ok().map(|count| vec![count]);
        Self::create_zeroed_with_shape_context(device, data_type, element_count, shape)
    }

    /// Copies host bytes holding whole `data_type` elements to `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is not a whole number of valid elements, `device` is not a valid CUDA device, or allocation or copying fails.
    pub fn create_from_bytes(device: Device, data_type: DataType, data: &[u8]) -> Result<Self> {
        let data_type_size = data_type.size_bytes();
        if !data.len().is_multiple_of(data_type_size) {
            return Err(Error::byte_length_mismatch(
                "cuda storage bytes",
                data_type_size,
                data.len(),
            ));
        }

        let shape = data
            .len()
            .checked_div(data_type_size)
            .and_then(|count| i64::try_from(count).ok())
            .map(|count| vec![count]);
        Self::create_from_bytes_with_shape_context(device, data_type, data, shape)
    }

    /// Copies the allocation's bytes into `dst`.
    ///
    /// # Errors
    ///
    /// Returns an error if capture rules prohibit reading this allocation, `dst` has the wrong length, or the copy fails.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        check_storage_readback(self.storage_identity())?;
        if dst.len() != self.byte_size() {
            return Err(Error::length_mismatch(
                "cuda copy_to_host destination",
                self.byte_size(),
                dst.len(),
            ));
        }

        self.context().bind()?;
        activity::cuda_readback();
        self.data.copy_to_host(dst).map_err(Error::from)
    }

    /// Clones this allocation to the CUDA device `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device or allocation or copying fails.
    pub fn clone_to_device(&self, device: Device) -> Result<Self> {
        let device_id = cuda_device_id(device)?;
        if device_id == self.context().device().id() {
            let data = self.data.try_clone().map_err(|error| {
                map_cuda_allocation_error(error, device, self.byte_size(), DataType::U8, None)
            })?;
            // SAFETY: `data` is a byte-for-byte clone of this storage, so every reader sees the same values.
            return unsafe { Self::from_raw(DataType::U8, data) };
        }

        // Storage-level cross-device cloning explicitly stages through initialized host bytes.
        // Direct peer transfers are provided separately by the CUDA memory API.
        let mut bytes = vec![0_u8; self.byte_size()];
        self.copy_to_host(&mut bytes)?;
        Self::create_from_bytes(device, DataType::U8, &bytes)
    }

    /// Fills the allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the context or the zero-fill fails.
    pub fn fill_zeroes(&mut self) -> Result<()> {
        self.context().bind()?;
        self.data.set_zeroes().map_err(Error::from)
    }
}
