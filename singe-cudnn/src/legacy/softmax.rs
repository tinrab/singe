//! Standalone softmax and log-softmax modes and forward/backward operations.

use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda::view::{DeviceOutput, DeviceRange, DeviceSlice, DeviceSliceMut};
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::FloatingElement,
    error::Result,
    legacy::tensor::TensorDescriptor,
    try_ffi,
    utility::{check_range, checked_byte_len},
};

singe_core::native_enum! {
    /// Selects the softmax implementation used by [`softmax_forward`] and [`softmax_backward`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SoftmaxAlgorithm: sys::cudnnSoftmaxAlgorithm_t {
        /// Applies the straightforward softmax operation.
        Fast = CUDNN_SOFTMAX_FAST,
        /// Scales each point of the softmax input domain by its maximum value to avoid potential floating point overflows in the softmax evaluation.
        Accurate = CUDNN_SOFTMAX_ACCURATE,
        /// This entry performs the log softmax operation, avoiding overflows by scaling each point in the input domain as in [`SoftmaxAlgorithm::Accurate`].
        Log = CUDNN_SOFTMAX_LOG,
    }
}

impl_enum_display!(SoftmaxAlgorithm, {
    Self::Fast => "CUDNN_SOFTMAX_FAST",
    Self::Accurate => "CUDNN_SOFTMAX_ACCURATE",
    Self::Log => "CUDNN_SOFTMAX_LOG",
});

singe_core::native_enum! {
    /// Selects the dimensions over which [`softmax_forward`] and [`softmax_backward`] compute their results.
    ///
    /// **Values**
    ///
    /// [`SoftmaxMode::Instance`] computes softmax per image (`N`) across dimensions `C,H,W`.
    ///
    /// [`SoftmaxMode::Channel`] computes softmax per spatial location (`H,W`) per image (`N`) across dimension `C`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SoftmaxMode: sys::cudnnSoftmaxMode_t {
        /// Compute softmax over all C, H, W for each image (N).
        Instance = CUDNN_SOFTMAX_MODE_INSTANCE,
        /// Compute softmax over channel (C) for each spatial location (H, W) and image (N).
        Channel = CUDNN_SOFTMAX_MODE_CHANNEL,
    }
}

impl_enum_display!(SoftmaxMode, {
    Self::Instance => "CUDNN_SOFTMAX_MODE_INSTANCE",
    Self::Channel => "CUDNN_SOFTMAX_MODE_CHANNEL",
});

fn require_capacity<T>(descriptor: &TensorDescriptor<T>, length: usize) -> Result<()> {
    let bytes = checked_byte_len::<T>(length, "softmax buffer bytes")?;
    check_range!(
        "softmax descriptor storage span",
        bytes >= descriptor.byte_len()?
    )?;
    Ok(())
}

/// Enqueues standalone softmax or stable log-softmax over the descriptor's elements on the handle's stream.
/// Destinations may be pending when beta is zero; padding and tails are not results.
///
/// # Safety
///
/// The crate's raw execution contract applies.
/// `x` must be initialized, and `y` too when `beta` is nonzero.
///
/// # Errors
///
/// Returns an error if context binding fails, a device range belongs to another CUDA context, tensor dimensions disagree, a storage byte count overflows or is too small for its descriptor, or cuDNN rejects the operation.
pub unsafe fn softmax_forward_async<T: FloatingElement>(
    context: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    check_range!(
        "softmax forward",
        x_desc.dimensions() == y_desc.dimensions()
    )?;
    require_capacity(x_desc, x.len())?;
    require_capacity(y_desc, y.len())?;
    context.prepare_regions(&[x.region(), y.region()])?;
    // SAFETY: the buffers belong to the handle's context and cover their descriptors; the caller keeps them live and ordered through completion.
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

/// Computes standalone softmax or stable log-softmax and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`softmax_forward_async`] fails, or the wait fails.
pub fn softmax_forward<T: FloatingElement>(
    context: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: `x` and `y` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            softmax_forward_async(context, algorithm, mode, alpha, x_desc, x, beta, y_desc, y)
        }
    })
}

/// Enqueues standalone softmax or log-softmax gradient production on the handle's stream.
/// Only descriptor elements are produced; destination padding remains untouched.
///
/// # Safety
///
/// The crate's raw execution contract applies.
/// `y` and `output_gradient` must be initialized, and `input_gradient` too when `beta` is nonzero.
///
/// # Errors
///
/// Returns an error if context binding fails, a device range belongs to another CUDA context, tensor dimensions disagree, a storage byte count overflows or is too small for its descriptor, or cuDNN rejects the operation.
pub unsafe fn softmax_backward_async<T: FloatingElement>(
    context: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &(impl DeviceRange<T> + ?Sized),
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceRange<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    check_range!(
        "softmax backward",
        y_desc.dimensions() == output_gradient_desc.dimensions()
            && y_desc.dimensions() == input_gradient_desc.dimensions()
    )?;
    require_capacity(y_desc, y.len())?;
    require_capacity(output_gradient_desc, output_gradient.len())?;
    require_capacity(input_gradient_desc, input_gradient.len())?;
    context.prepare_regions(&[
        y.region(),
        output_gradient.region(),
        input_gradient.region(),
    ])?;
    // SAFETY: the buffers belong to the handle's context and cover their descriptors; the caller keeps them live and ordered through completion.
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

/// Computes standalone softmax or log-softmax gradients and waits for them.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`softmax_backward_async`] fails, or the wait fails.
pub fn softmax_backward<T: FloatingElement>(
    context: &Context,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
    alpha: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &(impl DeviceSlice<T> + ?Sized),
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceSlice<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: the inputs and `input_gradient` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            softmax_backward_async(
                context,
                algorithm,
                mode,
                alpha,
                y_desc,
                y,
                output_gradient_desc,
                output_gradient,
                beta,
                input_gradient_desc,
                input_gradient,
            )
        }
    })
}
