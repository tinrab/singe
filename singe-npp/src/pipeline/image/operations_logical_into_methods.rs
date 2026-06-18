use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, operation_traits::LogicalBinaryImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: LogicalBinaryImage<T, L>,
{
    pub fn logical_and_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_and_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn logical_and_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_and_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }

    pub fn logical_or_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_or_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn logical_or_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_or_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }

    pub fn logical_xor_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_xor_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn logical_xor_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_xor_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }
}
