use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, operation_traits::DeviceConstantArithmeticImage};

#[path = "operations_device_constant_multiplicative_into_methods.rs"]
mod multiplicative_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DeviceConstantArithmeticImage<T, L>,
{
    pub fn add_device_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    pub fn add_device_constant_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }

    pub fn subtract_device_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    pub fn subtract_device_constant_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }
}
