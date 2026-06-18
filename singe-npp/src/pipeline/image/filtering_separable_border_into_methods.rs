use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point},
};

use super::super::super::{ImagePipeline, filtering::IntegerSeparableBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IntegerSeparableBorderFilterImage<T, L>,
{
    pub fn filter_column_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as IntegerSeparableBorderFilterImage<T, L>>::filter_column_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            anchor,
            divisor,
            border_type,
        )
    }

    pub fn filter_row_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as IntegerSeparableBorderFilterImage<T, L>>::filter_row_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            anchor,
            divisor,
            border_type,
        )
    }
}
