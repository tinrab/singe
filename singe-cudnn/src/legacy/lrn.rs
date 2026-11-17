//! Local response and divisive normalization descriptors and tensor operations.

use std::{mem, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda::view::{DeviceOutput, DeviceRange, DeviceSlice, DeviceSliceMut};
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::FloatingElement,
    error::{Error, Result},
    legacy::tensor::TensorDescriptor,
    try_ffi,
    utility::check_range,
};

singe_core::native_enum! {
    /// Operation mode for [`lrn_cross_channel_forward`] and [`lrn_cross_channel_backward`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum LrnMode: sys::cudnnLRNMode_t {
        /// LRN across the tensor's channel dimension.
        CrossChannelDim1 = CUDNN_LRN_CROSS_CHANNEL_DIM1,
    }
}

impl_enum_display!(LrnMode, {
    LrnMode::CrossChannelDim1 => "CUDNN_LRN_CROSS_CHANNEL_DIM1",
});

singe_core::native_enum! {
    /// Operation mode for [`divisive_normalization_forward`] and [`divisive_normalization_backward`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DivisiveNormalizationMode: sys::cudnnDivNormMode_t {
        /// The means tensor is expected to contain precomputed means or other kernel convolution values.
        /// cuDNN treats an omitted means buffer as zero-filled.
        /// Equivalent to spatial LRN.
        ///
        /// In the backward pass, means are independent inputs with independent gradients.
        /// In this mode, to yield a net gradient over the entire LCN computational graph, backpropagate `means_gradient` through the means layer (which can be implemented using average pooling) and add it to the `data_gradient` tensor produced by [`divisive_normalization_backward`].
        PrecomputedMeans = CUDNN_DIVNORM_PRECOMPUTED_MEANS,
    }
}

impl_enum_display!(DivisiveNormalizationMode, {
    DivisiveNormalizationMode::PrecomputedMeans => "CUDNN_DIVNORM_PRECOMPUTED_MEANS",
});

/// Window and normalization parameters shared by local response and divisive normalization.
#[derive(Debug, Clone, Copy)]
pub struct LrnOptions {
    /// Window width in elements; divisive normalization uses this width along every spatial axis.
    pub n: u32,
    /// Variance scaling coefficient, divided by the window width for LRN or by the spatial window volume for divisive normalization.
    pub alpha: f64,
    /// Exponent applied to the normalization denominator.
    pub beta: f64,
    /// Additive constant in the normalization denominator.
    pub k: f64,
}

impl Default for LrnOptions {
    // These are the values `cudnnCreateLRNDescriptor` installs.
    fn default() -> Self {
        Self {
            n: 5,
            alpha: 1e-4,
            beta: 0.75,
            k: 2.0,
        }
    }
}

/// Owns a native local response normalization descriptor with retained window and scaling parameters.
#[derive(Debug)]
pub struct LrnDescriptor {
    handle: sys::cudnnLRNDescriptor_t,
    config: LrnOptions,
}

/// A transferred native LRN descriptor with the parameters it was configured with.
#[derive(Debug)]
pub struct RawLrnDescriptor {
    /// The owned native LRN descriptor handle.
    pub handle: sys::cudnnLRNDescriptor_t,
    /// The window and normalization parameters configured on `handle`.
    pub options: LrnOptions,
}

impl LrnDescriptor {
    /// Creates an LRN descriptor with explicit window and normalization parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the window width is outside the native range, `k` or `beta` fails its native minimum bound, cuDNN rejects creation or configuration, or the returned handle is null.
    pub fn create(config: LrnOptions) -> Result<Self> {
        check_range!(
            "lrn n",
            (sys::CUDNN_LRN_MIN_N..=sys::CUDNN_LRN_MAX_N).contains(&config.n),
        )?;
        check_range!("lrn k", config.k >= sys::CUDNN_LRN_MIN_K)?;
        check_range!("lrn beta", config.beta >= sys::CUDNN_LRN_MIN_BETA)?;

        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a live local that cuDNN writes the new descriptor into.
        unsafe {
            try_ffi!(sys::cudnnCreateLRNDescriptor(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let descriptor = Self { handle, config };
        // SAFETY: `descriptor` owns the non-null handle created above, and the parameters are plain values checked against cuDNN's bounds.
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
    pub fn options(&self) -> LrnOptions {
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
    /// `raw.handle` must be a live LRN descriptor created by cuDNN.
    /// The returned wrapper takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    /// `raw.options` must accurately describe the descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null ([`Error::NullHandle`]).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the raw value transfers ownership of its handle even though its fields are `Copy`"
    )]
    pub unsafe fn from_raw(raw: RawLrnDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            config: raw.options,
        })
    }

    /// Releases ownership and returns the raw cuDNN LRN descriptor handle with its parameters.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> RawLrnDescriptor {
        RawLrnDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            options: self.config,
        }
    }
}

impl Drop for LrnDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: the handle is non-null and owned by this wrapper, and nothing uses it after drop.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroyLRNDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cudnn lrn descriptor", &error);
            }
        }
    }
}

/// Enqueues the forward LRN layer computation on the handle's stream.
///
/// Supported formats are: `positive-strided`, NCHW and NHWC for 4D `x` and `y`, and only NCDHW DHW-packed for 5D (for both `x` and `y`).
/// Only non-overlapping 4D and 5D tensors are supported.
/// NCHW layout is preferred for performance.
///
/// # Safety
///
/// The crate's raw execution contract applies.
/// `x` must be initialized, and `y` too when `beta` is nonzero.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small for its descriptor, if the input and output descriptors are incompatible, or if cuDNN rejects the descriptors or parameters.
pub unsafe fn lrn_cross_channel_forward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "y storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;

    check_range!("lrn forward", x_desc.dimensions() == y_desc.dimensions())?;
    context.prepare_regions(&[x.region(), y.region()])?;

    // SAFETY: the buffers belong to the handle's context and cover their descriptors; the caller keeps them live and ordered through completion.
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

/// Performs the forward LRN layer computation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`lrn_cross_channel_forward_async`] fails, or the wait fails.
pub fn lrn_cross_channel_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: the inputs and `y` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            lrn_cross_channel_forward_async(
                context, descriptor, mode, alpha, x_desc, x, beta, y_desc, y,
            )
        }
    })
}

/// Enqueues the backward LRN layer computation on the handle's stream.
///
/// Supported formats are: `positive-strided`, NCHW and NHWC for 4D `x` and `y`, and only NCDHW DHW-packed for 5D (for both `x` and `y`).
/// Only non-overlapping 4D and 5D tensors are supported.
/// NCHW layout is preferred for performance.
///
/// # Safety
///
/// The crate's raw execution contract applies.
/// `y`, `output_gradient`, and `x` must be initialized, and `input_gradient` too when `beta` is nonzero.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small for its descriptor, if the data and gradient descriptors are incompatible, or if cuDNN rejects the descriptors or parameters.
pub unsafe fn lrn_cross_channel_backward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &(impl DeviceRange<T> + ?Sized),
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceRange<T> + ?Sized),
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
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
    context.prepare_regions(&[
        y.region(),
        output_gradient.region(),
        x.region(),
        input_gradient.region(),
    ])?;

    // SAFETY: the buffers belong to the handle's context and cover their descriptors; the caller keeps them live and ordered through completion.
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

/// Performs the backward LRN layer computation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`lrn_cross_channel_backward_async`] fails, or the wait fails.
pub fn lrn_cross_channel_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &(impl DeviceSlice<T> + ?Sized),
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceSlice<T> + ?Sized),
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: the inputs and `input_gradient` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            lrn_cross_channel_backward_async(
                context,
                descriptor,
                mode,
                alpha,
                y_desc,
                y,
                output_gradient_desc,
                output_gradient,
                x_desc,
                x,
                beta,
                input_gradient_desc,
                input_gradient,
            )
        }
    })
}

/// Enqueues the forward spatial `DivisiveNormalization` layer computation on the handle's stream.
///
/// It divides every value in a layer by the standard deviation of its spatial neighbors.
/// `DivisiveNormalization` only implements the `x/max(c, sigma_x)` portion of the computation, where `sigma_x` is the variance over the spatial neighborhood of `x`.
/// The `x - mean(x)` portion of local contrastive normalization, often called subtractive normalization, can be implemented using a cuDNN average pooling layer followed by a tensor add operation.
///
/// Supported tensor formats are NCHW for 4D and NCDHW for 5D with any non-overlapping non-negative strides.
/// Only 4D and 5D tensors are supported.
///
/// # Safety
///
/// The crate's raw execution contract applies.
/// `x` and `means` must be initialized, and `y` too when `beta` is nonzero.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small for its descriptor, if the tensor descriptors are incompatible, or if cuDNN rejects the descriptors, strides, or parameters.
pub unsafe fn divisive_normalization_forward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    means: &(impl DeviceRange<T> + ?Sized),
    temp: &mut (impl DeviceOutput<T> + ?Sized),
    temp2: &mut (impl DeviceOutput<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
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
    context.prepare_regions(&[
        x.region(),
        means.region(),
        temp.region(),
        temp2.region(),
        y.region(),
    ])?;

    // SAFETY: the buffers belong to the handle's context and cover their descriptors; the caller keeps them live and ordered through completion.
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

/// Performs the forward spatial `DivisiveNormalization` layer computation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`divisive_normalization_forward_async`] fails, or the wait fails.
pub fn divisive_normalization_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    means: &(impl DeviceSlice<T> + ?Sized),
    temp: &mut (impl DeviceOutput<T> + ?Sized),
    temp2: &mut (impl DeviceOutput<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: the inputs and `y` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            divisive_normalization_forward_async(
                context, descriptor, mode, alpha, x_desc, x, means, temp, temp2, beta, y_desc, y,
            )
        }
    })
}

/// Enqueues the backward `DivisiveNormalization` layer computation on the handle's stream.
///
/// Supported tensor formats are NCHW for 4D and NCDHW for 5D with any non-overlapping non-negative strides.
/// Only 4D and 5D tensors are supported.
///
/// # Safety
///
/// The crate's raw execution contract applies.
/// `x`, `means`, `dy`, and the scratch written by the matching forward call must be initialized, and `dx` and `dmeans` too when `beta` is nonzero.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small for its descriptor, if `x_desc` and `dxdmeans_desc` are incompatible, or if cuDNN rejects the descriptors, strides, or parameters.
pub unsafe fn divisive_normalization_backward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    means: &(impl DeviceRange<T> + ?Sized),
    dy: &(impl DeviceRange<T> + ?Sized),
    temp: &mut (impl DeviceOutput<T> + ?Sized),
    temp2: &mut (impl DeviceOutput<T> + ?Sized),
    beta: &T::Scale,
    dxdmeans_desc: &TensorDescriptor<T>,
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dmeans: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
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
    context.prepare_regions(&[
        x.region(),
        means.region(),
        dy.region(),
        temp.region(),
        temp2.region(),
        dx.region(),
        dmeans.region(),
    ])?;

    // SAFETY: the buffers belong to the handle's context and cover their descriptors; the caller keeps them live and ordered through completion.
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

/// Performs the backward `DivisiveNormalization` layer computation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`divisive_normalization_backward_async`] fails, or the wait fails.
pub fn divisive_normalization_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    means: &(impl DeviceSlice<T> + ?Sized),
    dy: &(impl DeviceSlice<T> + ?Sized),
    temp: &mut (impl DeviceSliceMut<T> + ?Sized),
    temp2: &mut (impl DeviceSliceMut<T> + ?Sized),
    beta: &T::Scale,
    dxdmeans_desc: &TensorDescriptor<T>,
    dx: &mut (impl DeviceSliceMut<T> + ?Sized),
    dmeans: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: every buffer the operation reads is initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            divisive_normalization_backward_async(
                context,
                descriptor,
                mode,
                alpha,
                x_desc,
                x,
                means,
                dy,
                temp,
                temp2,
                beta,
                dxdmeans_desc,
                dx,
                dmeans,
            )
        }
    })
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_lrn_descriptor_create() -> Result<()> {
        let descriptor = LrnDescriptor::create(LrnOptions {
            n: 5,
            alpha: 1e-4,
            beta: 0.75,
            k: 2.0,
        })?;

        assert_eq!(descriptor.options().n, 5);
        assert_eq!(descriptor.options().alpha, 1e-4);
        assert_eq!(descriptor.options().beta, 0.75);
        assert_eq!(descriptor.options().k, 2.0);

        Ok(())
    }

    #[test]
    fn test_lrn_forward_backward_smoke() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let descriptor = LrnDescriptor::create(LrnOptions {
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
        let y_grad_host = vec![1.0f32; count];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &y_grad_host)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;

        test.stream().complete(|| -> Result<()> {
            // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
            unsafe {
                lrn_cross_channel_forward_async(
                    test_context,
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
                lrn_cross_channel_backward_async(
                    test_context,
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
        })?;

        let mut y_host = vec![0.0f32; count];
        let mut x_grad_host = vec![0.0f32; count];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let y = unsafe { y.assume_init() };
        y.copy_to_host(&mut y_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut x_grad_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(x_grad_host.iter().all(|value| value.is_finite()));

        Ok(())
    }

    #[test]
    fn test_divisive_normalization_forward_backward_smoke() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let descriptor = LrnDescriptor::create(LrnOptions {
            n: 5,
            alpha: 1e-4,
            beta: 0.75,
            k: 2.0,
        })?;
        let tensor_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 4, 2, 2])?;
        let count = tensor_desc.element_count()?;

        let x_host = vec![0.1f32; count];
        let means_host = vec![0.0f32; count];
        let y_grad_host = vec![1.0f32; count];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let means = DeviceMemory::from_slice(test_context.cuda_context(), &means_host)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &y_grad_host)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let mut dmeans = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let mut temp = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let mut temp2 = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;

        test.stream().complete(|| -> Result<()> {
            // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
            unsafe {
                divisive_normalization_forward_async(
                    test_context,
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
                divisive_normalization_backward_async(
                    test_context,
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
        })?;

        let mut y_host = vec![0.0f32; count];
        let mut x_grad_host = vec![0.0f32; count];
        let mut dmeans_host = vec![0.0f32; count];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let y = unsafe { y.assume_init() };
        y.copy_to_host(&mut y_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut x_grad_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dmeans = unsafe { dmeans.assume_init() };
        dmeans.copy_to_host(&mut dmeans_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(x_grad_host.iter().all(|value| value.is_finite()));
        assert!(dmeans_host.iter().all(|value| value.is_finite()));

        Ok(())
    }
}
