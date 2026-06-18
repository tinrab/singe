use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point},
};

use super::{FloatSeparableBorderFilterImage, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: FloatSeparableBorderFilterImage<T, L>,
{
    pub fn filter_column_border32f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as FloatSeparableBorderFilterImage<T, L>>::filter_column_border32f_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            anchor,
            border_type,
        )
    }

    pub fn filter_row_border32f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as FloatSeparableBorderFilterImage<T, L>>::filter_row_border32f_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            anchor,
            border_type,
        )
    }
}
