use std::{mem, ptr};

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

/// Operation mode for [`lrn_cross_channel_forward`] and [`lrn_cross_channel_backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LrnMode {
    /// LRN across the tensor's channel dimension.
    CrossChannelDim1 = sys::cudnnLRNMode_t::CUDNN_LRN_CROSS_CHANNEL_DIM1 as _,
}

impl_enum_conversion!(sys::cudnnLRNMode_t, LrnMode);

impl_enum_display!(LrnMode, {
    LrnMode::CrossChannelDim1 => "CUDNN_LRN_CROSS_CHANNEL_DIM1",
});

/// Operation mode for [`divisive_normalization_forward`] and [`divisive_normalization_backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DivisiveNormalizationMode {
    /// The means tensor is expected to contain precomputed means or other kernel
    /// convolution values.
    /// cuDNN treats an omitted means buffer as zero-filled.
    /// Equivalent to spatial LRN.
    ///
    /// In the backward pass, means are independent inputs with independent gradients.
    /// In this mode, to yield a net gradient over the entire LCN computational graph,
    /// backpropagate `means_gradient` through the means layer (which can be implemented
    /// using average pooling) and add it to the `data_gradient` tensor produced by
    /// [`divisive_normalization_backward`].
    PrecomputedMeans = sys::cudnnDivNormMode_t::CUDNN_DIVNORM_PRECOMPUTED_MEANS as _,
}

impl_enum_conversion!(sys::cudnnDivNormMode_t, DivisiveNormalizationMode);

impl_enum_display!(DivisiveNormalizationMode, {
    DivisiveNormalizationMode::PrecomputedMeans => "CUDNN_DIVNORM_PRECOMPUTED_MEANS",
});

/// Window and normalization parameters shared by local response and divisive normalization.
#[derive(Debug, Clone, Copy)]
pub struct LrnConfig {
    /// Window width in elements; divisive normalization uses this width along every spatial axis.
    pub n: u32,
    /// Variance scaling coefficient, divided by the window width for LRN or by the spatial window volume for divisive normalization.
    pub alpha: f64,
    /// Exponent applied to the normalization denominator.
    pub beta: f64,
    /// Additive constant in the normalization denominator.
    pub k: f64,
}

/// Owns a native local response normalization descriptor with retained window and scaling parameters.
#[derive(Debug)]
pub struct LrnDescriptor {
    handle: sys::cudnnLRNDescriptor_t,
    config: LrnConfig,
}

impl LrnDescriptor {
    /// Creates an LRN descriptor with explicit window and normalization parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the window width is outside the native range, `k` or `beta` fails its native minimum bound, cuDNN rejects creation or configuration, or the returned handle is null.
    pub fn create(config: LrnConfig) -> Result<Self> {
        check_range!(
            "lrn n",
            (sys::CUDNN_LRN_MIN_N..=sys::CUDNN_LRN_MAX_N).contains(&config.n),
        )?;
        check_range!("lrn k", config.k >= sys::CUDNN_LRN_MIN_K)?;
        check_range!("lrn beta", config.beta >= sys::CUDNN_LRN_MIN_BETA)?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cudnnCreateLRNDescriptor` receives valid output locations.
        unsafe {
            try_ffi!(sys::cudnnCreateLRNDescriptor(&raw mut handle))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        let descriptor = Self { handle, config };
        // SAFETY: `cudnnSetLRNDescriptor` receives only plain values.
        unsafe {
            try_ffi!(sys::cudnnSetLRNDescriptor(
                descriptor.handle,
                config.n,
                config.alpha,
                config.beta,
                config.k,
            ))?;
        }

        Ok(descriptor)
    }

    /// Returns the retained normalization parameters without querying cuDNN.
    #[must_use]
    pub fn config(&self) -> LrnConfig {
        self.config
    }

    /// Returns the borrowed native LRN descriptor handle, valid while this wrapper lives.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnLRNDescriptor_t {
        self.handle
    }

    /// Takes ownership of a raw cuDNN LRN descriptor handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cudnnLRNDescriptor_t` created by cuDNN. The
    /// returned wrapper takes ownership and will destroy it with
    /// `cudnnDestroyLRNDescriptor`; no other owner may destroy or keep using it.
    /// `config` must accurately describe the descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(handle: sys::cudnnLRNDescriptor_t, config: LrnConfig) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, config })
    }

    /// Releases ownership and returns the raw cuDNN LRN descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> sys::cudnnLRNDescriptor_t {
        mem::replace(&mut self.handle, ptr::null_mut())
    }
}

impl Drop for LrnDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: `cudnnDestroyLRNDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroyLRNDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn lrn descriptor: {error}");
            }
        }
    }
}

/// Performs the forward LRN layer computation.
///
/// Supported formats are: `positive-strided`, NCHW and NHWC for 4D `x` and `y`, and only NCDHW DHW-packed for 5D (for both `x` and `y`).
/// Only non-overlapping 4D and 5D tensors are supported.
/// NCHW layout is preferred for performance.
///
/// # Errors
///
/// Returns an error if the input and output descriptors are incompatible, if the
/// LRN descriptor parameters are outside their valid ranges, if a 5D tensor is
/// not in NCDHW DHW-packed format, or if cuDNN does not support the provided
/// tensor types, dimensions, or strides.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn lrn_cross_channel_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [x.region(), y.region()]
        .iter()
        .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "y storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;

    check_range!("lrn forward", x_desc.dimensions() == y_desc.dimensions())?;

    // SAFETY: `cudnnLRNCrossChannelForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cudnnLRNCrossChannelForward(
            context.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            ptr::from_ref(alpha).cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            ptr::from_ref(beta).cast(),
            y_desc.as_raw(),
            y.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Performs the backward LRN layer computation.
///
/// Supported formats are: `positive-strided`, NCHW and NHWC for 4D `x` and `y`, and only NCDHW DHW-packed for 5D (for both `x` and `y`).
/// Only non-overlapping 4D and 5D tensors are supported.
/// NCHW layout is preferred for performance.
///
/// # Errors
///
/// Returns an error if the data and gradient descriptors are incompatible, if
/// the LRN descriptor parameters are outside their valid ranges, if a 5D tensor
/// is not in NCDHW DHW-packed format, or if cuDNN does not support the provided
/// tensor types, dimensions, or strides.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn lrn_cross_channel_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &(impl DeviceInput<T> + ?Sized),
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceInput<T> + ?Sized),
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [
        y.region(),
        output_gradient.region(),
        x.region(),
        input_gradient.region(),
    ]
    .iter()
    .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "y storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;
    check_range!(
        "output_gradient storage span",
        output_gradient.region().byte_len() >= output_gradient_desc.byte_len()?
    )?;
    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "input_gradient storage span",
        input_gradient.region().byte_len() >= input_gradient_desc.byte_len()?
    )?;

    check_range!(
        "lrn backward",
        x_desc.dimensions() == y_desc.dimensions()
            && x_desc.dimensions() == output_gradient_desc.dimensions()
            && x_desc.dimensions() == input_gradient_desc.dimensions(),
    )?;

    // SAFETY: `cudnnLRNCrossChannelBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cudnnLRNCrossChannelBackward(
            context.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            ptr::from_ref(alpha).cast(),
            y_desc.as_raw(),
            y.as_device_ptr().cast(),
            output_gradient_desc.as_raw(),
            output_gradient.as_device_ptr().cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            ptr::from_ref(beta).cast(),
            input_gradient_desc.as_raw(),
            input_gradient.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Performs the forward spatial `DivisiveNormalization` layer computation.
/// It divides every value in a layer by the standard deviation of its spatial neighbors.
/// `DivisiveNormalization` only implements the `x/max(c, sigma_x)` portion of the computation, where `sigma_x` is the variance over the spatial neighborhood of `x`.
///
/// The full LCN (Local Contrastive Normalization) computation can be implemented as a two-step process:
///
/// The `x - mean(x)` portion, often called subtractive normalization, can be implemented using a
/// cuDNN average pooling layer followed by a tensor add operation.
///
/// Supported tensor formats are NCHW for 4D and NCDHW for 5D with any non-overlapping non-negative strides.
/// Only 4D and 5D tensors are supported.
///
/// # Errors
///
/// Returns an error if the input and output tensor descriptors are not 4D or 5D, if their dimensions are incompatible, if an in-place call uses
/// mismatched strides, if the LRN descriptor parameters are outside their valid
/// ranges, or if cuDNN does not support the provided strides or configuration.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn divisive_normalization_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    means: &(impl DeviceInput<T> + ?Sized),
    temp: &mut (impl DeviceOutput<T> + ?Sized),
    temp2: &mut (impl DeviceOutput<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [
        x.region(),
        means.region(),
        temp.region(),
        temp2.region(),
        y.region(),
    ]
    .iter()
    .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "means storage span",
        means.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "temp storage span",
        temp.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "temp2 storage span",
        temp2.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "y storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;

    check_range!(
        "divisive normalization forward",
        x_desc.dimensions() == y_desc.dimensions(),
    )?;

    // SAFETY: `cudnnDivisiveNormalizationForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cudnnDivisiveNormalizationForward(
            context.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            ptr::from_ref(alpha).cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            means.as_device_ptr().cast(),
            temp.as_mut_ptr().cast(),
            temp2.as_mut_ptr().cast(),
            ptr::from_ref(beta).cast(),
            y_desc.as_raw(),
            y.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Performs the backward `DivisiveNormalization` layer computation.
///
/// Supported tensor formats are NCHW for 4D and NCDHW for 5D with any non-overlapping non-negative strides.
/// Only 4D and 5D tensors are supported.
///
/// # Errors
///
/// Returns an error if the input or output tensor descriptors are not 4D or 5D, if `x_desc` and `dxdmeans_desc` dimensions are incompatible, if
/// the LRN descriptor parameters are outside their valid ranges, or if cuDNN
/// does not support the provided strides or configuration.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn divisive_normalization_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    means: &(impl DeviceInput<T> + ?Sized),
    dy: &(impl DeviceInput<T> + ?Sized),
    temp: &mut (impl DeviceOutput<T> + ?Sized),
    temp2: &mut (impl DeviceOutput<T> + ?Sized),
    beta: &T::Scale,
    dxdmeans_desc: &TensorDescriptor<T>,
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dmeans: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [
        x.region(),
        means.region(),
        dy.region(),
        temp.region(),
        temp2.region(),
        dx.region(),
        dmeans.region(),
    ]
    .iter()
    .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "means storage span",
        means.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "dy storage span",
        dy.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "temp storage span",
        temp.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "temp2 storage span",
        temp2.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "dx storage span",
        dx.region().byte_len() >= dxdmeans_desc.byte_len()?
    )?;
    check_range!(
        "dmeans storage span",
        dmeans.region().byte_len() >= dxdmeans_desc.byte_len()?
    )?;

    check_range!(
        "divisive normalization backward",
        x_desc.dimensions() == dxdmeans_desc.dimensions(),
    )?;

    // SAFETY: `cudnnDivisiveNormalizationBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cudnnDivisiveNormalizationBackward(
            context.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            ptr::from_ref(alpha).cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            means.as_device_ptr().cast(),
            dy.as_device_ptr().cast(),
            temp.as_mut_ptr().cast(),
            temp2.as_mut_ptr().cast(),
            ptr::from_ref(beta).cast(),
            dxdmeans_desc.as_raw(),
            dx.as_mut_ptr().cast(),
            dmeans.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::{memory::DeviceMemory, stream::StreamBinding};

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_lrn_descriptor_create() -> Result<()> {
        let descriptor = LrnDescriptor::create(LrnConfig {
            n: 5,
            alpha: 1e-4,
            beta: 0.75,
            k: 2.0,
        })?;

        assert_eq!(descriptor.config().n, 5);
        assert_eq!(descriptor.config().alpha, 1e-4);
        assert_eq!(descriptor.config().beta, 0.75);
        assert_eq!(descriptor.config().k, 2.0);

        Ok(())
    }

    #[test]
    fn test_lrn_forward_backward_smoke() -> Result<()> {
        let test_context = setup_context()?;

        let descriptor = LrnDescriptor::create(LrnConfig {
            n: 5,
            alpha: 1e-4,
            beta: 0.75,
            k: 2.0,
        })?;
        let tensor_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 4, 2, 2])?;
        let count = tensor_desc.element_count()?;

        let x_host = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];
        let dy_host = vec![1.0f32; count];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &dy_host)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;

        StreamBinding::Borrowed(test_context.stream().to_borrowed()).sync_scope(
            || -> Result<()> {
                // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
                unsafe {
                    lrn_cross_channel_forward(
                        &test_context,
                        &descriptor,
                        LrnMode::CrossChannelDim1,
                        &1.0f32,
                        &tensor_desc,
                        &x,
                        &0.0f32,
                        &tensor_desc,
                        &mut y,
                    )
                }?;
                // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
                unsafe {
                    lrn_cross_channel_backward(
                        &test_context,
                        &descriptor,
                        LrnMode::CrossChannelDim1,
                        &1.0f32,
                        &tensor_desc,
                        &y,
                        &tensor_desc,
                        &dy,
                        &tensor_desc,
                        &x,
                        &0.0f32,
                        &tensor_desc,
                        &mut dx,
                    )
                }?;

                Ok(())
            },
        )?;

        let mut y_host = vec![0.0f32; count];
        let mut dx_host = vec![0.0f32; count];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let y = unsafe { y.assume_init() };
        y.copy_to_host(&mut y_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut dx_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(dx_host.iter().all(|value| value.is_finite()));

        Ok(())
    }

    #[test]
    fn test_divisive_normalization_forward_backward_smoke() -> Result<()> {
        let test_context = setup_context()?;

        let descriptor = LrnDescriptor::create(LrnConfig {
            n: 5,
            alpha: 1e-4,
            beta: 0.75,
            k: 2.0,
        })?;
        let tensor_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 4, 2, 2])?;
        let count = tensor_desc.element_count()?;

        let x_host = vec![0.1f32; count];
        let means_host = vec![0.0f32; count];
        let dy_host = vec![1.0f32; count];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let means = DeviceMemory::from_slice(test_context.cuda_context(), &means_host)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &dy_host)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let mut dmeans = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let mut temp = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let mut temp2 = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;

        StreamBinding::Borrowed(test_context.stream().to_borrowed()).sync_scope(
            || -> Result<()> {
                // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
                unsafe {
                    divisive_normalization_forward(
                        &test_context,
                        &descriptor,
                        DivisiveNormalizationMode::PrecomputedMeans,
                        &1.0f32,
                        &tensor_desc,
                        &x,
                        &means,
                        &mut temp,
                        &mut temp2,
                        &0.0f32,
                        &tensor_desc,
                        &mut y,
                    )
                }?;
                // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
                unsafe {
                    divisive_normalization_backward(
                        &test_context,
                        &descriptor,
                        DivisiveNormalizationMode::PrecomputedMeans,
                        &1.0f32,
                        &tensor_desc,
                        &x,
                        &means,
                        &dy,
                        &mut temp,
                        &mut temp2,
                        &0.0f32,
                        &tensor_desc,
                        &mut dx,
                        &mut dmeans,
                    )
                }?;

                Ok(())
            },
        )?;

        let mut y_host = vec![0.0f32; count];
        let mut dx_host = vec![0.0f32; count];
        let mut dmeans_host = vec![0.0f32; count];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let y = unsafe { y.assume_init() };
        y.copy_to_host(&mut y_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut dx_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dmeans = unsafe { dmeans.assume_init() };
        dmeans.copy_to_host(&mut dmeans_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(dx_host.iter().all(|value| value.is_finite()));
        assert!(dmeans_host.iter().all(|value| value.is_finite()));

        Ok(())
    }
}
