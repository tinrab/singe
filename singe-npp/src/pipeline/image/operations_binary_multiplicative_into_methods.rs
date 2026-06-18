use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BinaryArithmeticImage<T, L>,
{
    pub fn multiply_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::multiply_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::multiply_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }

    pub fn divide_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::divide_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn divide_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::divide_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }
}
