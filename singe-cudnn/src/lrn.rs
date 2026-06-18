use std::ptr;

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

#[derive(Debug, Clone, Copy)]
pub struct LrnConfig {
    pub n: u32,
    pub alpha: f64,
    pub beta: f64,
    pub k: f64,
}

#[derive(Debug)]
pub struct LrnDescriptor {
    handle: sys::cudnnLRNDescriptor_t,
    config: LrnConfig,
}

impl LrnDescriptor {
    pub fn create(config: LrnConfig) -> Result<Self> {
        check_range!(
            "lrn n",
            (sys::CUDNN_LRN_MIN_N..=sys::CUDNN_LRN_MAX_N).contains(&config.n),
        )?;
        check_range!("lrn k", config.k >= sys::CUDNN_LRN_MIN_K)?;
        check_range!("lrn beta", config.beta >= sys::CUDNN_LRN_MIN_BETA)?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreateLRNDescriptor(&raw mut handle))?;
            try_ffi!(sys::cudnnSetLRNDescriptor(
                handle,
                config.n,
                config.alpha,
                config.beta,
                config.k,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, config })
    }

    pub fn config(&self) -> LrnConfig {
        self.config
    }

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
    pub unsafe fn from_raw(handle: sys::cudnnLRNDescriptor_t, config: LrnConfig) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, config })
    }

    /// Releases ownership and returns the raw cuDNN LRN descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cudnnLRNDescriptor_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl Drop for LrnDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudnnDestroyLRNDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy lrn descriptor: {err}");
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
pub fn lrn_cross_channel_forward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    beta: &T,
    y_desc: &TensorDescriptor<T>,
    y: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!("lrn forward", x_desc.dimensions() == y_desc.dimensions())?;

    unsafe {
        try_ffi!(sys::cudnnLRNCrossChannelForward(
            ctx.as_raw(),
            descriptor.as_raw(),
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
pub fn lrn_cross_channel_backward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &LrnDescriptor,
    mode: LrnMode,
    alpha: &T,
    y_desc: &TensorDescriptor<T>,
    y: &DeviceMemory<T>,
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &DeviceMemory<T>,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    beta: &T,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!(
        "lrn backward",
        x_desc.dimensions() == y_desc.dimensions()
            && x_desc.dimensions() == output_gradient_desc.dimensions()
            && x_desc.dimensions() == input_gradient_desc.dimensions(),
    )?;

    unsafe {
        try_ffi!(sys::cudnnLRNCrossChannelBackward(
            ctx.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            alpha as *const T as _,
            y_desc.as_raw(),
            y.as_ptr() as _,
            output_gradient_desc.as_raw(),
            output_gradient.as_ptr() as _,
            x_desc.as_raw(),
            x.as_ptr() as _,
            beta as *const T as _,
            input_gradient_desc.as_raw(),
            input_gradient.as_mut_ptr() as _,
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
pub fn divisive_normalization_forward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    means: &DeviceMemory<T>,
    temp: &mut DeviceMemory<T>,
    temp2: &mut DeviceMemory<T>,
    beta: &T,
    y_desc: &TensorDescriptor<T>,
    y: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!(
        "divisive normalization forward",
        x_desc.dimensions() == y_desc.dimensions(),
    )?;

    unsafe {
        try_ffi!(sys::cudnnDivisiveNormalizationForward(
            ctx.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            alpha as *const T as _,
            x_desc.as_raw(),
            x.as_ptr() as _,
            means.as_ptr() as _,
            temp.as_mut_ptr() as _,
            temp2.as_mut_ptr() as _,
            beta as *const T as _,
            y_desc.as_raw(),
            y.as_mut_ptr() as _,
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
pub fn divisive_normalization_backward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &LrnDescriptor,
    mode: DivisiveNormalizationMode,
    alpha: &T,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    means: &DeviceMemory<T>,
    dy: &DeviceMemory<T>,
    temp: &mut DeviceMemory<T>,
    temp2: &mut DeviceMemory<T>,
    beta: &T,
    dxdmeans_desc: &TensorDescriptor<T>,
    dx: &mut DeviceMemory<T>,
    dmeans: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!(
        "divisive normalization backward",
        x_desc.dimensions() == dxdmeans_desc.dimensions(),
    )?;

    unsafe {
        try_ffi!(sys::cudnnDivisiveNormalizationBackward(
            ctx.as_raw(),
            descriptor.as_raw(),
            mode.into(),
            alpha as *const T as _,
            x_desc.as_raw(),
            x.as_ptr() as _,
            means.as_ptr() as _,
            dy.as_ptr() as _,
            temp.as_mut_ptr() as _,
            temp2.as_mut_ptr() as _,
            beta as *const T as _,
            dxdmeans_desc.as_raw(),
            dx.as_mut_ptr() as _,
            dmeans.as_mut_ptr() as _,
        ))?;
    }

    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    use crate::testing::setup_context;

    #[test]
    fn lrn_descriptor_create() -> Result<()> {
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
    fn lrn_forward_backward_smoke() -> Result<()> {
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

        let mut x = DeviceMemory::<f32>::create(count)?;
        let mut y = DeviceMemory::<f32>::create(count)?;
        let mut dy = DeviceMemory::<f32>::create(count)?;
        let mut dx = DeviceMemory::<f32>::create(count)?;

        x.copy_from_host(&x_host)?;
        dy.copy_from_host(&dy_host)?;

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
        )?;
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
        )?;

        test_context.stream().synchronize()?;

        let mut y_host = vec![0.0f32; count];
        let mut dx_host = vec![0.0f32; count];
        y.copy_to_host(&mut y_host)?;
        dx.copy_to_host(&mut dx_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(dx_host.iter().all(|value| value.is_finite()));

        Ok(())
    }

    #[test]
    fn divisive_normalization_forward_backward_smoke() -> Result<()> {
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

        let mut x = DeviceMemory::<f32>::create(count)?;
        let mut y = DeviceMemory::<f32>::create(count)?;
        let mut means = DeviceMemory::<f32>::create(count)?;
        let mut dy = DeviceMemory::<f32>::create(count)?;
        let mut dx = DeviceMemory::<f32>::create(count)?;
        let mut dmeans = DeviceMemory::<f32>::create(count)?;
        let mut temp = DeviceMemory::<f32>::create(count)?;
        let mut temp2 = DeviceMemory::<f32>::create(count)?;

        x.copy_from_host(&x_host)?;
        means.copy_from_host(&means_host)?;
        dy.copy_from_host(&dy_host)?;

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
        )?;
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
        )?;

        test_context.stream().synchronize()?;

        let mut y_host = vec![0.0f32; count];
        let mut dx_host = vec![0.0f32; count];
        let mut dmeans_host = vec![0.0f32; count];
        y.copy_to_host(&mut y_host)?;
        dx.copy_to_host(&mut dx_host)?;
        dmeans.copy_to_host(&mut dmeans_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(dx_host.iter().all(|value| value.is_finite()));
        assert!(dmeans_host.iter().all(|value| value.is_finite()));

        Ok(())
    }
}
