mod cache;
mod registry;
mod stream;

use singe_cuda::{
    device::Device as NativeCudaDevice,
    error::{Error as CudaError, Status},
};

pub use self::{
    cache::{
        allocator_stats, autotune_cache_stats, clear_cache, cublaslt_matmul_algorithm_cache_stats,
        provider_cache_stats, reset_peak_memory, set_memory_limit, tune_cublaslt_matmul,
    },
    registry::{
        eager_execution_family, operation_execution_family, supports_operation,
        supports_operation_input, supports_operation_output,
    },
    stream::{current_stream, execution_context, set_current_stream, synchronize},
};
use crate::{
    device::Device,
    error::{Error, Result},
};

/// Returns the number of CUDA devices visible to the process.
///
/// # Errors
///
/// Returns an error if CUDA device enumeration fails for a reason other than no device being present.
pub fn device_count() -> Result<i32> {
    match NativeCudaDevice::count() {
        Ok(count) => Ok(count),
        Err(CudaError::Cuda {
            code: Status::NoDevice,
            ..
        }) => Ok(0),
        Err(error) => Err(Error::from(error)),
    }
}

/// Returns `true` when at least one CUDA device is available.
#[must_use]
pub fn is_available() -> bool {
    device_count().is_ok_and(|count| count > 0)
}

/// Returns a checked CUDA device handle for `id`.
///
/// # Errors
///
/// Returns an error if `id` is outside the visible CUDA device range or if CUDA device enumeration fails.
pub fn device(id: i32) -> Result<Device> {
    if id < 0 {
        return Err(Error::negative_cuda_device_id(id));
    }

    let count = device_count()?;
    if count == 0 {
        return Err(Error::no_cuda_devices());
    }

    if id >= count {
        return Err(Error::cuda_device_out_of_range(id, count));
    }
    Ok(Device::Cuda(id))
}

#[cfg(test)]
mod tests;
