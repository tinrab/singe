use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::super::{ImagePipeline, operation_traits::DeviceConstantArithmeticImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DeviceConstantArithmeticImage<T, L>,
{
    pub fn multiply_device_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::multiply_device_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_device_constant_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::multiply_device_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }

    pub fn divide_device_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::divide_device_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    pub fn divide_device_constant_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::divide_device_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }
}
