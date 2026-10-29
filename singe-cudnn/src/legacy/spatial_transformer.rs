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
    data_type::{DataType, FloatingElement},
    error::{Error, Result},
    try_ffi,
    utility::{check_range, to_i32, to_i32_vec},
};

/// [`SamplerType`] selects the sampler type used by [`SpatialTransformerDescriptor`],
/// [`spatial_tf_sampler_forward`], and [`spatial_tf_sampler_backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SamplerType {
    /// Selects the bilinear sampler.
    Bilinear = sys::cudnnSamplerType_t::CUDNN_SAMPLER_BILINEAR as _,
}

impl_enum_conversion!(sys::cudnnSamplerType_t, SamplerType);

impl_enum_display!(SamplerType, {
    SamplerType::Bilinear => "CUDNN_SAMPLER_BILINEAR",
});

/// Sampler and element type for a native spatial transformer descriptor.
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    /// Interpolation sampler used for spatial transformation.
    pub sampler_type: SamplerType,
    /// Element type of the transformed tensor.
    pub data_type: DataType,
}

/// Owns a configured native spatial transformer descriptor and retains the transformed tensor dimensions.
#[derive(Debug)]
pub struct SpatialTransformerDescriptor {
    handle: sys::cudnnSpatialTransformerDescriptor_t,
    config: SpatialTransformerConfig,
    shape: Vec<u32>,
}

impl SpatialTransformerDescriptor {
    /// Creates a spatial transformer descriptor for the supplied transformed tensor shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape has fewer than three dimensions, its rank or dimensions cannot fit native `i32` values, cuDNN rejects creation or configuration, or the returned handle is null.
    pub fn create(config: SpatialTransformerConfig, shape: &[u32]) -> Result<Self> {
        if shape.len() < 3 {
            return Err(Error::InvalidDataShape);
        }

        let dims = to_i32_vec(shape, "spatial transformer dimension")?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cudnnCreateSpatialTransformerDescriptor` receives valid output locations.
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
        // SAFETY: `cudnnSetSpatialTransformerNdDescriptor` receives pointers into buffers borrowed for the call.
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
    pub fn config(&self) -> SpatialTransformerConfig {
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
    /// `handle` must be a valid `cudnnSpatialTransformerDescriptor_t` created by
    /// cuDNN. The returned wrapper takes ownership and will destroy it with
    /// `cudnnDestroySpatialTransformerDescriptor`; no other owner may destroy
    /// or keep using it. `config` and `shape` must accurately describe the
    /// descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(
        handle: sys::cudnnSpatialTransformerDescriptor_t,
        config: SpatialTransformerConfig,
        shape: Vec<u32>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            config,
            shape,
        })
    }

    /// Releases ownership and returns the raw cuDNN spatial transformer descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> sys::cudnnSpatialTransformerDescriptor_t {
        mem::replace(&mut self.handle, ptr::null_mut())
    }
}

impl Drop for SpatialTransformerDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: `cudnnDestroySpatialTransformerDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroySpatialTransformerDescriptor(self.handle))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn spatial transformer descriptor: {error}");
            }
        }
    }
}

/// Generates a grid of input-tensor coordinates corresponding to each output-tensor pixel.
///
/// Only 2D transformation is supported.
///
/// # Errors
///
/// Returns an error if `descriptor`, `theta`, or `grid` is invalid, if the
/// operation fails to launch on the GPU, or if cuDNN does not support the
/// provided configuration, such as a transformed tensor dimension greater than
/// 4.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn spatial_tf_grid_generator_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    theta: &(impl DeviceInput<T> + ?Sized),
    grid: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [theta.region(), grid.region()]
        .iter()
        .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "spatial transformer data type",
        descriptor.config().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(d as usize).ok_or(Error::InvalidDataShape)
        })?;
    check_range!(
        "theta span",
        theta.len()
            >= (shape[0] as usize)
                .checked_mul(6)
                .ok_or(Error::InvalidDataShape)?
    )?;
    check_range!("grid span", grid.len() >= grid_elements)?;

    // SAFETY: `cudnnSpatialTfGridGeneratorForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
/// # Errors
///
/// Returns an error if `descriptor`, `dgrid`, or `dtheta` is invalid, if the
/// operation fails to launch on the GPU, or if cuDNN does not support the
/// provided configuration, such as a transformed tensor dimension greater than
/// 4.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn spatial_tf_grid_generator_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    dgrid: &(impl DeviceInput<T> + ?Sized),
    dtheta: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [dgrid.region(), dtheta.region()]
        .iter()
        .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "spatial transformer data type",
        descriptor.config().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(d as usize).ok_or(Error::InvalidDataShape)
        })?;
    check_range!("dgrid span", dgrid.len() >= grid_elements)?;
    check_range!(
        "dtheta span",
        dtheta.len()
            >= (shape[0] as usize)
                .checked_mul(6)
                .ok_or(Error::InvalidDataShape)?
    )?;

    // SAFETY: `cudnnSpatialTfGridGeneratorBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
/// # Errors
///
/// Returns an error if `y_desc` dimensions differ from those specified in
/// `descriptor`, if the operation fails to launch on the GPU, or if cuDNN does
/// not support the provided configuration, such as a transformed tensor
/// dimension greater than 4.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn spatial_tf_sampler_forward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    grid: &(impl DeviceInput<T> + ?Sized),
    beta: &T::Scale,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [x.region(), grid.region(), y.region()]
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
    check_range!(
        "spatial transformer data type",
        descriptor.config().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(d as usize).ok_or(Error::InvalidDataShape)
        })?;
    check_range!("grid span", grid.len() >= grid_elements)?;

    // SAFETY: `cudnnSpatialTfSamplerForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
/// # Errors
///
/// Returns an error if `output_gradient_desc` dimensions differ from those
/// specified in `descriptor`, if the operation fails to launch on the GPU, or if
/// cuDNN does not support the provided configuration, such as a transformed
/// tensor dimension greater than 4.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Nonzero beta requires initialized destination elements. Backward scratch
/// must contain the state required by the matching forward operation.
pub unsafe fn spatial_tf_sampler_backward<T: FloatingElement>(
    context: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T::Scale,
    x_desc: &TensorDescriptor<T>,
    x: &(impl DeviceInput<T> + ?Sized),
    beta: &T::Scale,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
    alpha_dgrid: &T::Scale,
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceInput<T> + ?Sized),
    grid: &(impl DeviceInput<T> + ?Sized),
    beta_dgrid: &T::Scale,
    dgrid: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    if [
        x.region(),
        input_gradient.region(),
        output_gradient.region(),
        grid.region(),
        dgrid.region(),
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
        "input_gradient storage span",
        input_gradient.region().byte_len() >= input_gradient_desc.byte_len()?
    )?;
    check_range!(
        "output_gradient storage span",
        output_gradient.region().byte_len() >= output_gradient_desc.byte_len()?
    )?;
    check_range!(
        "spatial transformer data type",
        descriptor.config().data_type == T::data_type()
    )?;
    check_range!("spatial transformer rank", descriptor.shape().len() == 4)?;
    let shape = descriptor.shape();
    let grid_elements = [shape[0], shape[2], shape[3], 2]
        .iter()
        .try_fold(1usize, |n, &d| {
            n.checked_mul(d as usize).ok_or(Error::InvalidDataShape)
        })?;
    check_range!("grid span", grid.len() >= grid_elements)?;
    check_range!("dgrid span", dgrid.len() >= grid_elements)?;

    check_range!(
        "spatial transformer backward",
        x_desc.dimensions() == input_gradient_desc.dimensions(),
    )?;

    // SAFETY: `cudnnSpatialTfSamplerBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::{memory::DeviceMemory, stream::StreamBinding};

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_spatial_transformer_descriptor_create() -> Result<()> {
        let descriptor = SpatialTransformerDescriptor::create(
            SpatialTransformerConfig {
                sampler_type: SamplerType::Bilinear,
                data_type: DataType::F32,
            },
            &[1, 1, 2, 2],
        )?;

        assert_eq!(descriptor.config().sampler_type, SamplerType::Bilinear);
        assert_eq!(descriptor.config().data_type, DataType::F32);
        assert_eq!(descriptor.shape(), &[1, 1, 2, 2]);

        Ok(())
    }

    #[test]
    fn test_spatial_transformer_forward_backward_smoke() -> Result<()> {
        let test_context = setup_context()?;

        let descriptor = SpatialTransformerDescriptor::create(
            SpatialTransformerConfig {
                sampler_type: SamplerType::Bilinear,
                data_type: DataType::F32,
            },
            &[1, 1, 2, 2],
        )?;
        let x_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 1, 2, 2])?;
        let y_desc = TensorDescriptor::<f32>::create_contiguous(&[1, 1, 2, 2])?;

        let x_host = vec![1.0f32, 2.0, 3.0, 4.0];
        let theta_host = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let dy_host = vec![1.0f32, 1.0, 1.0, 1.0];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 4)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 4)?;
        let theta = DeviceMemory::from_slice(test_context.cuda_context(), &theta_host)?;
        let mut dtheta = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 6)?;
        let mut grid = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 8)?;
        let mut dgrid = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), 8)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &dy_host)?;

        StreamBinding::Borrowed(test_context.stream().to_borrowed()).sync_scope(
            || -> Result<()> {
                // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
                unsafe {
                    spatial_tf_grid_generator_forward(&test_context, &descriptor, &theta, &mut grid)
                }?;
                // SAFETY: the descriptors and buffers are test locals that outlive the draining scope.
                unsafe {
                    spatial_tf_sampler_forward(
                        &test_context,
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
                    spatial_tf_sampler_backward(
                        &test_context,
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
                    spatial_tf_grid_generator_backward(
                        &test_context,
                        &descriptor,
                        &dgrid,
                        &mut dtheta.output(),
                    )
                }?;

                Ok(())
            },
        )?;

        let mut y_host = vec![0.0f32; 4];
        let mut dx_host = vec![0.0f32; 4];
        let mut dtheta_host = vec![0.0f32; 6];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let y = unsafe { y.assume_init() };
        y.copy_to_host(&mut y_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut dx_host)?;
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dtheta = unsafe { dtheta.assume_init() };
        dtheta.copy_to_host(&mut dtheta_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(dx_host.iter().all(|value| value.is_finite()));
        assert!(dtheta_host.iter().all(|value| value.is_finite()));

        Ok(())
    }
}
