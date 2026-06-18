use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub trait DeviceConstantArithmeticImage<T, L> {
    type Constant;

    fn add_device_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<Self::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn add_device_constant_image_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<Self::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_device_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<Self::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_device_constant_image_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<Self::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_device_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<Self::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_device_constant_image_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<Self::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_device_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<Self::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_device_constant_image_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<Self::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait MultiplyDeviceConstantScaleImage<T, L> {
    type Constant;

    fn multiply_device_constant_scale_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<Self::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn multiply_device_constant_scale_image_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<Self::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
