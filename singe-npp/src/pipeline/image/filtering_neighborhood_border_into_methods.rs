use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

use super::{BoxBorderFilterImage, ImagePipeline, MedianBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MedianBorderFilterImage<T, L>,
{
    pub fn filter_median_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as MedianBorderFilterImage<T, L>>::filter_median_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BoxBorderFilterImage<T, L>,
{
    pub fn filter_box_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as BoxBorderFilterImage<T, L>>::filter_box_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }
}
