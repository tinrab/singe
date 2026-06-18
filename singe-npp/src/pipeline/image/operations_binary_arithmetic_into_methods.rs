use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

#[path = "operations_binary_multiplicative_into_methods.rs"]
mod multiplicative_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BinaryArithmeticImage<T, L>,
{
    pub fn add_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::add_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn add_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::add_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }

    pub fn subtract_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::subtract_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn subtract_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::subtract_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }
}
