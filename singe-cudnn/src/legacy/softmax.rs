use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    error::Error as CudaError,
    view::{DeviceInput, DeviceOutput},
};
use singe_cudnn_sys as sys;

use crate::{
    backend::tensor::TensorDescriptor,
    context::Context,
    data_type::FloatingElement,
    error::{Error, Result},
    try_ffi,
    utility::check_range,
};

/// Selects the softmax implementation used by [`softmax_forward`] and
/// [`softmax_backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SoftmaxAlgorithm {
    /// Applies the straightforward softmax operation.
    Fast = sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST as _,
    /// Scales each point of the softmax input domain by its maximum value to avoid
    /// potential floating point overflows in the softmax evaluation.
    Accurate = sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE as _,
    /// This entry performs the log softmax operation, avoiding overflows by scaling each point in the input domain as in [`SoftmaxAlgorithm::Accurate`].
    Log = sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG as _,
}

impl_enum_conversion!(sys::cudnnSoftmaxAlgorithm_t, SoftmaxAlgorithm);

impl_enum_display!(SoftmaxAlgorithm, {
    Self::Fast => "CUDNN_SOFTMAX_FAST",
    Self::Accurate => "CUDNN_SOFTMAX_ACCURATE",
    Self::Log => "CUDNN_SOFTMAX_LOG",
});

/// Selects the dimensions over which [`softmax_forward`] and
/// [`softmax_backward`] compute their results.
///
/// **Values**
///
/// [`SoftmaxMode::Instance`] computes softmax per image (`N`) across dimensions `C,H,W`.
///
/// [`SoftmaxMode::Channel`] computes softmax per spatial location (`H,W`) per image (`N`) across dimension `C`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SoftmaxMode {
    ///< Compute softmax over all C, H, W for each image (N). @since cuDNN 9.0.0
    Instance = sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE as _,
    ///< Compute softmax over channel (C) for each spatial location (H, W) and image (N). @since cuDNN 9.0.0
    Channel = sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL as _,
}

impl_enum_conversion!(sys::cudnnSoftmaxMode_t, SoftmaxMode);

impl_enum_display!(SoftmaxMode, {
    Self::Instance => "CUDNN_SOFTMAX_MODE_INSTANCE",
    Self::Channel => "CUDNN_SOFTMAX_MODE_CHANNEL",
});

fn require_capacity<T>(descriptor: &TensorDescriptor<T>, length: usize) -> Result<()> {
    let bytes = length
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::out_of_range("softmax buffer bytes"))?;
    check_range!(
        "softmax descriptor storage span",
        bytes >= descriptor.byte_len()?
    )?;
    Ok(())
}

/// Enqueues standalone softmax or stable log-softmax over the descriptor's elements.
/// Destinations may be pending when beta is zero; padding and tails are not results.
///
/// # Safety
/// Retain the context, device ranges, and exclusive writes through completion and replay.
/// Do not race handle reconfiguration with submission.
/// Input elements must be valid when read; order their producers before this operation.
/// When beta is nonzero, all accessed
/// destination elements must already be initialized.
/// Descriptors must match
/// their tensor element types and scaling coefficients are consumed during the call.
///
/// # Errors
///
/// Returns an error if context binding fails, a device range belongs to another CUDA context, tensor dimensions disagree, a storage byte count overflows or is too small for its descriptor, or cuDNN rejects the operation.
pub unsafe fn softmax_forward<T: FloatingElement>(
    context: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if x.region().context() != context.cuda_context()
        || y.region().context() != context.cuda_context()
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "softmax forward",
        x_desc.dimensions() == y_desc.dimensions()
    )?;
    require_capacity(x_desc, x.len())?;
    require_capacity(y_desc, y.len())?;
    // SAFETY: checked spans and the caller's queued access contract cover native use.
    unsafe {
        try_ffi!(sys::cudnnSoftmaxForward(
            context.as_raw(),
            algorithm.into(),
            mode.into(),
            ptr::from_ref(alpha).cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            ptr::from_ref(beta).cast(),
            y_desc.as_raw(),
            y.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Enqueues standalone softmax or log-softmax gradient production.
/// Only descriptor elements are produced; destination padding remains untouched.
///
/// # Safety
/// Retain the context, input values, output storage, and exclusive writes through completion
/// and captured replay. Order input producers before use. Nonzero beta requires
/// initialized destination elements. Descriptors must match their element types.
///
/// # Errors
///
/// Returns an error if context binding fails, a device range belongs to another CUDA context, tensor dimensions disagree, a storage byte count overflows or is too small for its descriptor, or cuDNN rejects the operation.
pub unsafe fn softmax_backward<T: FloatingElement>(
    context: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &(impl DeviceInput<T> + ?Sized),
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceInput<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [
        y.region(),
        output_gradient.region(),
        input_gradient.region(),
    ]
    .iter()
    .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "softmax backward",
        y_desc.dimensions() == output_gradient_desc.dimensions()
            && y_desc.dimensions() == input_gradient_desc.dimensions()
    )?;
    require_capacity(y_desc, y.len())?;
    require_capacity(output_gradient_desc, output_gradient.len())?;
    require_capacity(input_gradient_desc, input_gradient.len())?;
    // SAFETY: checked spans and the caller's queued access contract cover native use.
    unsafe {
        try_ffi!(sys::cudnnSoftmaxBackward(
            context.as_raw(),
            algorithm.into(),
            mode.into(),
            ptr::from_ref(alpha).cast(),
            y_desc.as_raw(),
            y.as_device_ptr().cast(),
            output_gradient_desc.as_raw(),
            output_gradient.as_device_ptr().cast(),
            ptr::from_ref(beta).cast(),
            input_gradient_desc.as_raw(),
            input_gradient.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}
