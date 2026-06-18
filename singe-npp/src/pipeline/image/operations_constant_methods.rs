use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::{ImagePipeline, operation_traits::ConstantArithmeticImage};

#[path = "operations_multiplicative_constant_methods.rs"]
mod multiplicative_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ConstantArithmeticImage<T, L>,
    <Self as ConstantArithmeticImage<T, L>>::Constant: Copy,
{
    pub fn add_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::add_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    pub fn add_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::add_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }

    pub fn subtract_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::subtract_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    pub fn subtract_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::subtract_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }
}
