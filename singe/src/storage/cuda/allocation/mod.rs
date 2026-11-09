use singe_cuda::{
    error::{Error as CudaError, Status as CudaStatus},
    memory::DeviceMemory,
};

use crate::{
    activity,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    storage::cuda::{CudaStorage, cuda_context_for_device, cuda_device_id},
    tensor::validate_boolean_bytes,
};

impl CudaStorage {
    /// Allocates zero-filled CUDA storage with optional shape context for errors.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device, the byte-size calculation overflows, or CUDA allocation fails.
    pub(crate) fn create_zeroed_with_shape_context(
        device: Device,
        data_type: DataType,
        element_count: usize,
        shape: Option<Vec<i64>>,
    ) -> Result<Self> {
        activity::storage_request();
        let device_id = cuda_device_id(device)?;

        let context = cuda_context_for_device(device_id)?;
        context.bind()?;

        let byte_size = element_count
            .checked_mul(data_type.size_bytes())
            .ok_or_else(|| Error::out_of_range("cuda storage byte size"))?;
        let data = DeviceMemory::<u8>::zeroed(&context, byte_size).map_err(|error| {
            map_cuda_allocation_error(error, device, byte_size, data_type, shape)
        })?;
        // SAFETY: the device buffer was just allocated with exactly the bytes for this data type.
        unsafe { Self::from_raw(data_type, data) }
    }

    /// Copies host bytes into CUDA storage with optional shape context for errors.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device or CUDA allocation/copying fails.
    /// Callers are responsible for validating that `data` contains a whole number of typed elements before calling this helper.
    pub(crate) fn create_from_bytes_with_shape_context(
        device: Device,
        data_type: DataType,
        data: &[u8],
        shape: Option<Vec<i64>>,
    ) -> Result<Self> {
        activity::storage_request();
        validate_boolean_bytes(data_type, data)?;
        let device_id = cuda_device_id(device)?;

        let context = cuda_context_for_device(device_id)?;
        context.bind()?;

        let data = DeviceMemory::<u8>::from_slice(&context, data).map_err(|error| {
            map_cuda_allocation_error(error, device, data.len(), data_type, shape)
        })?;
        // SAFETY: the device buffer was just allocated with exactly the bytes for this data type.
        unsafe { Self::from_raw(data_type, data) }
    }
}

pub fn map_cuda_allocation_error(
    error: CudaError,
    device: Device,
    byte_count: usize,
    data_type: DataType,
    shape: Option<Vec<i64>>,
) -> Error {
    match error {
        CudaError::Cuda {
            code: CudaStatus::OutOfMemory,
            ..
        } => Error::AllocationFailed {
            device,
            byte_count,
            data_type,
            shape,
        },
        error => Error::from(error),
    }
}

#[cfg(test)]
mod tests;
