use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point},
};

use super::super::{ImagePipeline, filtering::HighLowGaussBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HighLowGaussBorderFilterImage<T, L>,
{
    pub fn filter_high_pass_border_partial_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HighLowGaussBorderFilterImage<T, L>>::filter_high_pass_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    pub fn filter_low_pass_border_partial_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HighLowGaussBorderFilterImage<T, L>>::filter_low_pass_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    pub fn filter_gauss_border_partial_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HighLowGaussBorderFilterImage<T, L>>::filter_gauss_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
