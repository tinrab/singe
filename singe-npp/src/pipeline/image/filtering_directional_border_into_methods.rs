use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point},
};

use super::super::{ImagePipeline, filtering::*};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: EdgeDirectionalBorderFilterImage<T, L>,
{
    pub fn filter_prewitt_horizontal_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_horizontal_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn filter_prewitt_vertical_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_vertical_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn filter_roberts_down_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_down_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn filter_roberts_up_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_up_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn filter_sobel_horizontal_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_horizontal_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn filter_sobel_vertical_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_vertical_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
