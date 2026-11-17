//! Spatial transformation descriptors, coordinate-grid generation, and bilinear sampling.

use std::{mem, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda::view::{DeviceOutput, DeviceRange, DeviceSlice, DeviceSliceMut};
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::{DataType, FloatingElement},
    error::{Error, Result},
    legacy::tensor::TensorDescriptor,
    try_ffi,
    utility::{check_range, to_i32, to_i32_vec, to_usize},
};

singe_core::native_enum! {
    /// [`SamplerType`] selects the sampler type used by [`SpatialTransformerDescriptor`], [`spatial_tf_sampler_forward`], and [`spatial_tf_sampler_backward`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SamplerType: sys::cudnnSamplerType_t {
        /// Selects the bilinear sampler.
        Bilinear = CUDNN_SAMPLER_BILINEAR,
    }
}

impl_enum_display!(SamplerType, {
    SamplerType::Bilinear => "CUDNN_SAMPLER_BILINEAR",
});

/// Sampler and element type for a native spatial transformer descriptor.
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerOptions {
    /// Interpolation sampler used for spatial transformation.
    pub sampler_type: SamplerType,
    /// Element type of the transformed tensor.
    pub data_type: DataType,
}

/// Owns a configured native spatial transformer descriptor and retains the transformed tensor dimensions.
#[derive(Debug)]
pub struct SpatialTransformerDescriptor {
    handle: sys::cudnnSpatialTransformerDescriptor_t,
    config: SpatialTransformerOptions,
    shape: Vec<u32>,
}

/// A transferred native spatial transformer descriptor with the settings and shape it was configured with.
#[derive(Debug)]
pub struct RawSpatialTransformerDescriptor {
    /// The owned native spatial transformer descriptor handle.
    pub handle: sys::cudnnSpatialTransformerDescriptor_t,
    /// The sampler and element type configured on `handle`.
    pub options: SpatialTransformerOptions,
    /// The transformed tensor dimensions configured on `handle`.
    pub shape: Vec<u32>,
}

impl SpatialTransformerDescriptor {
    /// Creates a spatial transformer descriptor for the supplied transformed tensor shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape has fewer than three dimensions, its rank or dimensions cannot fit native `i32` values, cuDNN rejects creation or configuration, or the returned handle is null.
    pub fn create(config: SpatialTransformerOptions, shape: &[u32]) -> Result<Self> {
        if shape.len() < 3 {
            return Err(Error::InvalidDataShape);
        }

        let dims = to_i32_vec(shape, "spatial transformer dimension")?;

        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a live local that cuDNN writes the new descriptor into.
        unsafe {
            try_ffi!(sys::cudnnCreateSpatialTransformerDescriptor(
                &raw mut handle
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let descriptor = Self {
            handle,
            config,
            shape: shape.to_vec(),
        };
        // SAFETY: `descriptor` owns the non-null handle created above, and `dims` holds exactly the rank's `i32` dimensions for the call.
        unsafe {
            try_ffi!(sys::cudnnSetSpatialTransformerNdDescriptor(
                descriptor.handle,
                config.sampler_type.into(),
                config.data_type.into(),
                to_i32(shape.len(), "spatial transformer rank")?,
                dims.as_ptr(),
            ))?;
        }

        Ok(descriptor)
    }

    /// Returns the retained sampler and tensor element type.
    #[must_use]
    pub fn options(&self) -> SpatialTransformerOptions {
        self.config
    }

    /// Borrows the declared transformed tensor dimensions in native axis order.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Returns the borrowed native descriptor handle, valid while this wrapper lives.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnSpatialTransformerDescriptor_t {
        self.handle
    }

    /// Takes ownership of a raw cuDNN spatial transformer descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a live spatial transformer descriptor created by cuDNN.
    /// The returned wrapper takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    /// `raw.options` and `raw.shape` must accurately describe the descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null ([`Error::NullHandle`]).
    pub unsafe fn from_raw(raw: RawSpatialTransformerDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            config: raw.options,
            shape: raw.shape,
        })
    }

    /// Releases ownership and returns the raw cuDNN spatial transformer descriptor handle with its settings and shape.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> RawSpatialTransformerDescriptor {
        RawSpatialTransformerDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            options: self.config,
            shape: mem::take(&mut self.shape),
        }
    }
}

impl Drop for SpatialTransformerDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: the handle is non-null and owned by this wrapper, and nothing uses it after drop.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroySpatialTransformerDescriptor(self.handle))
            {
                report_cleanup_failure("destroy", "cudnn spatial transformer descriptor", &error);
            }
        }
    }
}

/// Generates a grid of input-tensor coordinates corresponding to each output-tensor pixel.
///
/// Only 2D transformation is supported.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Nonzero beta requires initialized destination elements.
/// Backward scratch must contain the state required by the matching forward operation.
///
/// # Errors
///
/// Returns an error if `descriptor`, `theta`, or `grid` is invalid, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration, such as a transformed tensor dimension greater than 4.
pub unsafe fn spatial_tf_grid_generator_forward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    theta: &(impl DeviceRange<T> + ?Sized),
    grid: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    context.prepare_regions(&[theta.region(), grid.region()])?;

    check_range!(
        "spatial transformer data type",
        descriptor.options().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(to_usize(d, "spatial transformer dimension")?)
                .ok_or(Error::InvalidDataShape)
        })?;
    check_range!(
        "theta span",
        theta.len()
            >= to_usize(shape[0], "spatial transformer batch")?
                .checked_mul(6)
                .ok_or(Error::InvalidDataShape)?
    )?;
    check_range!("grid span", grid.len() >= grid_elements)?;

    // SAFETY: the descriptor matches `T` and rank 4, `theta` and `grid` share the handle's context and hold the six-coefficient and grid spans checked above.
    // The caller retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnSpatialTfGridGeneratorForward(
            context.as_raw(),
            descriptor.as_raw(),
            theta.as_device_ptr().cast(),
            grid.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Computes the gradient of a grid generation operation.
///
/// Only 2D transformation is supported.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Nonzero beta requires initialized destination elements.
/// Backward scratch must contain the state required by the matching forward operation.
///
/// # Errors
///
/// Returns an error if `descriptor`, `dgrid`, or `dtheta` is invalid, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration, such as a transformed tensor dimension greater than 4.
pub unsafe fn spatial_tf_grid_generator_backward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    dgrid: &(impl DeviceRange<T> + ?Sized),
    dtheta: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    context.prepare_regions(&[dgrid.region(), dtheta.region()])?;

    check_range!(
        "spatial transformer data type",
        descriptor.options().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(to_usize(d, "spatial transformer dimension")?)
                .ok_or(Error::InvalidDataShape)
        })?;
    check_range!("dgrid span", dgrid.len() >= grid_elements)?;
    check_range!(
        "dtheta span",
        dtheta.len()
            >= to_usize(shape[0], "spatial transformer batch")?
                .checked_mul(6)
                .ok_or(Error::InvalidDataShape)?
    )?;

    // SAFETY: the descriptor matches `T` and rank 4, `dgrid` and `dtheta` share the handle's context and hold the grid and six-coefficient spans checked above.
    // The caller retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnSpatialTfGridGeneratorBackward(
            context.as_raw(),
            descriptor.as_raw(),
            dgrid.as_device_ptr().cast(),
            dtheta.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Performs a sampler operation and generates the output tensor from the grid generator output.
///
/// Only 2D transformation is supported.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Nonzero beta requires initialized destination elements.
/// Backward scratch must contain the state required by the matching forward operation.
///
/// # Errors
///
/// Returns an error if `y_desc` dimensions differ from those specified in `descriptor`, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration, such as a transformed tensor dimension greater than 4.
pub unsafe fn spatial_tf_sampler_forward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    grid: &(impl DeviceRange<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    context.prepare_regions(&[x.region(), grid.region(), y.region()])?;

    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "y storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;
    check_range!(
        "spatial transformer data type",
        descriptor.options().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(to_usize(d, "spatial transformer dimension")?)
                .ok_or(Error::InvalidDataShape)
        })?;
    check_range!("grid span", grid.len() >= grid_elements)?;

    // SAFETY: the descriptor matches `T` and rank 4, the buffers share the handle's context and cover their descriptors and the grid span, and the scalars are live host values of `T::Scale`.
    // The caller retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnSpatialTfSamplerForward(
            context.as_raw(),
            descriptor.as_raw(),
            ptr::from_ref(alpha).cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            grid.as_device_ptr().cast(),
            ptr::from_ref(beta).cast(),
            y_desc.as_raw(),
            y.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Computes the gradient of a sampling operation.
///
/// Only 2D transformation is supported.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Nonzero beta requires initialized destination elements.
/// Backward scratch must contain the state required by the matching forward operation.
///
/// # Errors
///
/// Returns an error if `output_gradient_desc` dimensions differ from those specified in `descriptor`, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration, such as a transformed tensor dimension greater than 4.
pub unsafe fn spatial_tf_sampler_backward_async<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
    alpha_dgrid: &T::Scale,
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceRange<T> + ?Sized),
    grid: &(impl DeviceRange<T> + ?Sized),
    beta_dgrid: &T::Scale,
    dgrid: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    context.prepare_regions(&[
        x.region(),
        input_gradient.region(),
        output_gradient.region(),
        grid.region(),
        dgrid.region(),
    ])?;

    check_range!(
        "x storage span",
        x.region().byte_len() >= x_desc.byte_len()?
    )?;
    check_range!(
        "input_gradient storage span",
        input_gradient.region().byte_len() >= input_gradient_desc.byte_len()?
    )?;
    check_range!(
        "output_gradient storage span",
        output_gradient.region().byte_len() >= output_gradient_desc.byte_len()?
    )?;
    check_range!(
        "spatial transformer data type",
        descriptor.options().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(to_usize(d, "spatial transformer dimension")?)
                .ok_or(Error::InvalidDataShape)
        })?;
    check_range!("grid span", grid.len() >= grid_elements)?;
    check_range!("dgrid span", dgrid.len() >= grid_elements)?;

    check_range!(
        "spatial transformer backward",
        x_desc.dimensions() == input_gradient_desc.dimensions(),
    )?;

    // SAFETY: the descriptor matches `T` and rank 4, the buffers share the handle's context and cover their descriptors and grid spans, and the scalars are live host values of `T::Scale`.
    // The caller retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnSpatialTfSamplerBackward(
            context.as_raw(),
            descriptor.as_raw(),
            ptr::from_ref(alpha).cast(),
            x_desc.as_raw(),
            x.as_device_ptr().cast(),
            ptr::from_ref(beta).cast(),
            input_gradient_desc.as_raw(),
            input_gradient.as_mut_ptr().cast(),
            ptr::from_ref(alpha_dgrid).cast(),
            output_gradient_desc.as_raw(),
            output_gradient.as_device_ptr().cast(),
            grid.as_device_ptr().cast(),
            ptr::from_ref(beta_dgrid).cast(),
            dgrid.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Generates the sampling grid for each output pixel and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`spatial_tf_grid_generator_forward_async`] fails, or the wait fails.
pub fn spatial_tf_grid_generator_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    theta: &(impl DeviceSlice<T> + ?Sized),
    grid: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: `theta` is initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe { spatial_tf_grid_generator_forward_async(context, descriptor, theta, grid) }
    })
}

/// Computes the gradient of a grid generation operation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`spatial_tf_grid_generator_backward_async`] fails, or the wait fails.
pub fn spatial_tf_grid_generator_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    dgrid: &(impl DeviceSlice<T> + ?Sized),
    dtheta: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: `dgrid` is initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe { spatial_tf_grid_generator_backward_async(context, descriptor, dgrid, dtheta) }
    })
}

/// Samples the output tensor from the grid generator output and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`spatial_tf_sampler_forward_async`] fails, or the wait fails.
pub fn spatial_tf_sampler_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    grid: &(impl DeviceSlice<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: the inputs and `y` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            spatial_tf_sampler_forward_async(
                context, descriptor, alpha, x_desc, x, grid, beta, y_desc, y,
            )
        }
    })
}

/// Computes the gradient of a sampling operation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`spatial_tf_sampler_backward_async`] fails, or the wait fails.
pub fn spatial_tf_sampler_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceSliceMut<T> + ?Sized),
    alpha_dgrid: &T::Scale,
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceSlice<T> + ?Sized),
    grid: &(impl DeviceSlice<T> + ?Sized),
    beta_dgrid: &T::Scale,
    dgrid: &mut (impl DeviceSliceMut<T> + ?Sized),
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: every buffer the operation reads is initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            spatial_tf_sampler_backward_async(
                context,
                descriptor,
                alpha,
                x_desc,
                x,
                beta,
                input_gradient_desc,
                input_gradient,
                alpha_dgrid,
                output_gradient_desc,
                output_gradient,
                grid,
                beta_dgrid,
                dgrid,
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
    fn test_spatial_transformer_descriptor_create() -> Result<()> {
        let descriptor = SpatialTransformerDescriptor::create(
            SpatialTransformerOptions {
                sampler_type: SamplerType::Bilinear,
                data_type: DataType::F32,
            },
            &[1, 1, 2, 2],
        )?;

        assert_eq!(descriptor.options().sampler_type, SamplerType::Bilinear);
        assert_eq!(descriptor.options().data_type, DataType::F32);
        assert_eq!(descriptor.shape(), &[1, 1, 2, 2]);

        Ok(())
    }

    #[test]
    fn test_spatial_transformer_forward_backward_smoke() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let descriptor = SpatialTransformerDescriptor::create(
            SpatialTransformerOptions {
                sampler_type: SamplerType::Bilinear,
                data_type: DataType::F32,
            },
            &[1, 1, 2, 2],
        )?;
        let x_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 1, 2, 2])?;
        let y_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 1, 2, 2])?;

        let x_host = vec![1.0f32, 2.0, 3.0, 4.0];
        let theta_host = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let y_grad_host = vec![1.0f32, 1.0, 1.0, 1.0];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 4)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 4)?;
        let theta = DeviceMemory::from_slice(test_context.cuda_context(), &theta_host)?;
        let mut dtheta = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 6)?;
        let mut grid = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 8)?;
        let mut dgrid = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 8)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &y_grad_host)?;

        test.stream().complete(|| -> Result<()> {
            // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
            unsafe {
                spatial_tf_grid_generator_forward_async(
                    test_context,
                    &descriptor,
                    &theta,
                    &mut grid,
                )
            }?;
            // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
            unsafe {
                spatial_tf_sampler_forward_async(
                    test_context,
                    &descriptor,
                    &1.0f32,
                    &x_desc,
                    &x,
                    &grid,
                    &0.0f32,
                    &y_desc,
                    &mut y,
                )
            }?;
            // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
            unsafe {
                spatial_tf_sampler_backward_async(
                    test_context,
                    &descriptor,
                    &1.0f32,
                    &x_desc,
                    &x,
                    &0.0f32,
                    &x_desc,
                    &mut dx,
                    &1.0f32,
                    &y_desc,
                    &dy,
                    &grid,
                    &0.0f32,
                    &mut dgrid,
                )
            }?;
            // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
            unsafe {
                spatial_tf_grid_generator_backward_async(
                    test_context,
                    &descriptor,
                    &dgrid,
                    &mut dtheta.output(),
                )
            }?;

            Ok(())
        })?;

        let mut y_host = vec![0.0f32; 4];
        let mut x_grad_host = vec![0.0f32; 4];
        let mut dtheta_host = vec![0.0f32; 6];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let y = unsafe { y.assume_init() };
        y.copy_to_host(&mut y_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut x_grad_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dtheta = unsafe { dtheta.assume_init() };
        dtheta.copy_to_host(&mut dtheta_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(x_grad_host.iter().all(|value| value.is_finite()));
        assert!(dtheta_host.iter().all(|value| value.is_finite()));

        Ok(())
    }
}
