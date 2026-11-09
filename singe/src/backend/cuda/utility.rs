use singe_cuda::types::DevicePtr;

use crate::{
    backend::cuda::tensor_region::CudaTensorSource,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    storage::cuda::CudaStorage,
    tensor::Tensor,
};

pub fn cuda_storage(tensor: &Tensor) -> Result<&CudaStorage> {
    tensor.storage().as_cuda().ok_or_else(|| {
        Error::device_mismatch(
            "cuda storage device",
            tensor.device(),
            tensor.storage().device(),
        )
    })
}

pub fn ensure_same_device(
    a: &(impl CudaTensorSource + ?Sized),
    b: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    if a.device() != b.device() {
        return Err(Error::device_mismatch(
            "cuda tensor device",
            a.device(),
            b.device(),
        ));
    }

    if !a.device().is_cuda() {
        return Err(Error::device_mismatch(
            "cuda tensor device",
            Device::Cuda(0),
            a.device(),
        ));
    }
    Ok(())
}

pub fn ensure_f32(tensor: &(impl CudaTensorSource + ?Sized)) -> Result<()> {
    if tensor.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cuda tensor operation"));
    }
    Ok(())
}

pub fn ensure_same_data_type(
    lhs: &(impl CudaTensorSource + ?Sized),
    rhs: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    if lhs.data_type() != rhs.data_type() {
        return Err(Error::data_type_mismatch(
            "cuda tensor operand",
            lhs.data_type(),
            rhs.data_type(),
        ));
    }
    Ok(())
}

/// Wraps a raw CUDA device pointer for FFI calls.
///
/// # Safety
///
/// `ptr` must be a valid CUDA device pointer for the duration of the FFI call that receives the returned [`DevicePtr`].
pub unsafe fn device_ptr<T>(ptr: *const T) -> DevicePtr {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { DevicePtr::new(ptr.cast_mut().cast::<()>()) }
}
