use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::memory::DeviceMemory;
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::DataTypeLike,
    error::{Error, Result},
    tensor::TensorDescriptor,
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
    Instance = sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE as _,
    Channel = sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL as _,
}

impl_enum_conversion!(sys::cudnnSoftmaxMode_t, SoftmaxMode);

impl_enum_display!(SoftmaxMode, {
    Self::Instance => "CUDNN_SOFTMAX_MODE_INSTANCE",
    Self::Channel => "CUDNN_SOFTMAX_MODE_CHANNEL",
});

/// Computes the softmax function.
///
/// In-place operation is allowed; `x` and `y` may refer to the same
/// allocation.
/// However, this requires `x_desc` and `y_desc` to be identical, particularly their strides.
///
/// All tensor formats are supported for all modes and algorithms with 4 and 5D tensors.
/// Performance is expected to be highest with NCHW fully-packed tensors.
/// For more than 5 dimensions, tensors must be packed in their spatial dimensions.
///
/// # Errors
///
/// Returns an error if the input and output tensor dimensions or data types
/// differ, if `algorithm` or `mode` is rejected by cuDNN, or if cuDNN does not
/// support the provided configuration.
pub fn softmax_forward<T: DataTypeLike>(
    ctx: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    beta: &T,
    y_desc: &TensorDescriptor<T>,
    y: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!(
        "softmax forward",
        x_desc.dimensions() == y_desc.dimensions()
    )?;

    unsafe {
        try_ffi!(sys::cudnnSoftmaxForward(
            ctx.as_raw(),
            algorithm.into(),
            mode.into(),
            alpha as *const T as _,
            x_desc.as_raw(),
            x.as_ptr() as _,
            beta as *const T as _,
            y_desc.as_raw(),
            y.as_mut_ptr() as _,
        ))?;
    }
    Ok(())
}

/// Computes the gradient of the softmax function.
///
/// In-place operation is allowed; `dy` and `dx` may refer to the same
/// allocation.
/// However, this requires `output_gradient_desc` and `input_gradient_desc` to be identical, particularly their strides.
///
/// All tensor formats are supported for all modes and algorithms with 4 and 5D tensors.
/// Performance is expected to be highest with NCHW fully-packed tensors.
/// For more than 5 dimensions, tensors must be packed in their spatial dimensions.
///
/// # Errors
///
/// Returns an error if the tensor dimensions, strides, or data types are
/// incompatible, if the operation fails to launch on the GPU, or if cuDNN does
/// not support the provided configuration.
pub fn softmax_backward<T: DataTypeLike>(
    ctx: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T,
    y_desc: &TensorDescriptor<T>,
    y: &DeviceMemory<T>,
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &DeviceMemory<T>,
    beta: &T,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!(
        "softmax backward",
        y_desc.dimensions() == output_gradient_desc.dimensions()
            && y_desc.dimensions() == input_gradient_desc.dimensions(),
    )?;

    unsafe {
        try_ffi!(sys::cudnnSoftmaxBackward(
            ctx.as_raw(),
            algorithm.into(),
            mode.into(),
            alpha as *const T as _,
            y_desc.as_raw(),
            y.as_ptr() as _,
            output_gradient_desc.as_raw(),
            output_gradient.as_ptr() as _,
            beta as *const T as _,
            input_gradient_desc.as_raw(),
            input_gradient.as_mut_ptr() as _,
        ))?;
    }
    Ok(())
}
