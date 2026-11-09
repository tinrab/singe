//! Backend dispatch traits, backend facades, and operation support metadata.

pub(crate) mod cpu;
pub(crate) mod decomposition;
mod index_problem;
mod kernels;
pub(crate) mod meta;

/// Canonical operation schemas, inference, and backend capability metadata.
pub mod registry;

/// Checked operation geometry and data-type rules shared by backend implementations.
pub(crate) mod spec;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg_attr(
    not(feature = "cuda"),
    expect(
        dead_code,
        unused_imports,
        reason = "without CUDA only the CPU interpreter uses the lowering, for element-wise operations"
    )
)]
mod kernel_lowering;

pub(crate) use self::kernels::register_kernels;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
};

/// Allocates a zero-filled contiguous tensor on `device`.
///
/// # Errors
///
/// Returns an error if the device is unavailable in this build, the byte count overflows, or allocation fails.
pub(crate) fn zeros_on(device: Device, data_type: DataType, shape: &Shape) -> Result<Tensor> {
    match device {
        Device::Cpu => cpu::storage::zeros(device, data_type, shape),
        Device::Cuda(id) if id < 0 => Err(Error::negative_cuda_device_id(id)),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::eager::storage::cuda_zeroed_tensor(device, data_type, shape),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(Error::cuda_support_disabled()),
        Device::Meta => meta::zeros(device, data_type, shape),
    }
}

/// Copies native-endian host bytes into a contiguous tensor on `device`.
///
/// # Errors
///
/// Returns an error if the device is unavailable in this build, the byte length differs from the shape's, a Boolean byte is invalid, or allocation or the copy fails.
pub(crate) fn from_bytes_on(
    device: Device,
    data_type: DataType,
    shape: &Shape,
    data: &[u8],
) -> Result<Tensor> {
    match device {
        Device::Cpu => cpu::storage::create_from_bytes(device, data_type, shape, data),
        Device::Cuda(id) if id < 0 => Err(Error::negative_cuda_device_id(id)),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::eager::storage::tensor_from_bytes(device, data_type, shape, data),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(Error::cuda_support_disabled()),
        Device::Meta => meta::from_bytes(device, data_type, shape, data),
    }
}
