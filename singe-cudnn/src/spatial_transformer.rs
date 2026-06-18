use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::memory::DeviceMemory;
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::{DataType, DataTypeLike},
    error::{Error, Result},
    tensor::TensorDescriptor,
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

#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub sampler_type: SamplerType,
    pub data_type: DataType,
}

#[derive(Debug)]
pub struct SpatialTransformerDescriptor {
    handle: sys::cudnnSpatialTransformerDescriptor_t,
    config: SpatialTransformerConfig,
    shape: Vec<u32>,
}

impl SpatialTransformerDescriptor {
    pub fn create(config: SpatialTransformerConfig, shape: &[u32]) -> Result<Self> {
        if shape.len() < 3 {
            return Err(Error::InvalidDataShape);
        }

        let dims = to_i32_vec(shape, "spatial transformer dimension")?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreateSpatialTransformerDescriptor(
                &raw mut handle
            ))?;
            try_ffi!(sys::cudnnSetSpatialTransformerNdDescriptor(
                handle,
                config.sampler_type.into(),
                config.data_type.into(),
                to_i32(shape.len(), "spatial transformer rank")?,
                dims.as_ptr(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            config,
            shape: shape.to_vec(),
        })
    }

    pub fn config(&self) -> SpatialTransformerConfig {
        self.config
    }

    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

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
    pub fn into_raw(self) -> sys::cudnnSpatialTransformerDescriptor_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl Drop for SpatialTransformerDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudnnDestroySpatialTransformerDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy spatial transformer descriptor: {err}");
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
pub fn spatial_tf_grid_generator_forward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &SpatialTransformerDescriptor,
    theta: &DeviceMemory<T>,
    grid: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    unsafe {
        try_ffi!(sys::cudnnSpatialTfGridGeneratorForward(
            ctx.as_raw(),
            descriptor.as_raw(),
            theta.as_ptr() as _,
            grid.as_mut_ptr() as _,
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
pub fn spatial_tf_grid_generator_backward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &SpatialTransformerDescriptor,
    dgrid: &DeviceMemory<T>,
    dtheta: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    unsafe {
        try_ffi!(sys::cudnnSpatialTfGridGeneratorBackward(
            ctx.as_raw(),
            descriptor.as_raw(),
            dgrid.as_ptr() as _,
            dtheta.as_mut_ptr() as _,
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
pub fn spatial_tf_sampler_forward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    grid: &DeviceMemory<T>,
    beta: &T,
    y_desc: &TensorDescriptor<T>,
    y: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    unsafe {
        try_ffi!(sys::cudnnSpatialTfSamplerForward(
            ctx.as_raw(),
            descriptor.as_raw(),
            alpha as *const T as _,
            x_desc.as_raw(),
            x.as_ptr() as _,
            grid.as_ptr() as _,
            beta as *const T as _,
            y_desc.as_raw(),
            y.as_mut_ptr() as _,
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
pub fn spatial_tf_sampler_backward<T: DataTypeLike>(
    ctx: &Context,
    descriptor: &SpatialTransformerDescriptor,
    alpha: &T,
    x_desc: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    beta: &T,
    input_gradient_desc: &TensorDescriptor<T>,
    input_gradient: &mut DeviceMemory<T>,
    alpha_dgrid: &T,
    output_gradient_desc: &TensorDescriptor<T>,
    output_gradient: &DeviceMemory<T>,
    grid: &DeviceMemory<T>,
    beta_dgrid: &T,
    dgrid: &mut DeviceMemory<T>,
) -> Result<()> {
    ctx.bind()?;

    check_range!(
        "spatial transformer backward",
        x_desc.dimensions() == input_gradient_desc.dimensions(),
    )?;

    unsafe {
        try_ffi!(sys::cudnnSpatialTfSamplerBackward(
            ctx.as_raw(),
            descriptor.as_raw(),
            alpha as *const T as _,
            x_desc.as_raw(),
            x.as_ptr() as _,
            beta as *const T as _,
            input_gradient_desc.as_raw(),
            input_gradient.as_mut_ptr() as _,
            alpha_dgrid as *const T as _,
            output_gradient_desc.as_raw(),
            output_gradient.as_ptr() as _,
            grid.as_ptr() as _,
            beta_dgrid as *const T as _,
            dgrid.as_mut_ptr() as _,
        ))?;
    }

    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    use crate::testing::setup_context;

    #[test]
    fn spatial_transformer_descriptor_create() -> Result<()> {
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
    fn spatial_transformer_forward_backward_smoke() -> Result<()> {
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

        let mut x = DeviceMemory::<f32>::create(4)?;
        let mut y = DeviceMemory::<f32>::create(4)?;
        let mut dx = DeviceMemory::<f32>::create(4)?;
        let mut theta = DeviceMemory::<f32>::create(6)?;
        let mut dtheta = DeviceMemory::<f32>::create(6)?;
        let mut grid = DeviceMemory::<f32>::create(8)?;
        let mut dgrid = DeviceMemory::<f32>::create(8)?;
        let mut dy = DeviceMemory::<f32>::create(4)?;

        x.copy_from_host(&x_host)?;
        theta.copy_from_host(&theta_host)?;
        dy.copy_from_host(&dy_host)?;

        spatial_tf_grid_generator_forward(&test_context, &descriptor, &theta, &mut grid)?;
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
        )?;
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
        )?;
        spatial_tf_grid_generator_backward(&test_context, &descriptor, &dgrid, &mut dtheta)?;

        test_context.stream().synchronize()?;

        let mut y_host = vec![0.0f32; 4];
        let mut dx_host = vec![0.0f32; 4];
        let mut dtheta_host = vec![0.0f32; 6];
        y.copy_to_host(&mut y_host)?;
        dx.copy_to_host(&mut dx_host)?;
        dtheta.copy_to_host(&mut dtheta_host)?;

        assert!(y_host.iter().all(|value| value.is_finite()));
        assert!(dx_host.iter().all(|value| value.is_finite()));
        assert!(dtheta_host.iter().all(|value| value.is_finite()));

        Ok(())
    }
}
