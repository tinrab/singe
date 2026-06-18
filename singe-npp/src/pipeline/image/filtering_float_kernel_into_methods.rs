use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

use super::{FloatKernelFilterImage, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: FloatKernelFilterImage<T, L>,
{
    pub fn filter_kernel32f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
        )
    }

    pub fn filter_kernel32f_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            kernel_size,
            anchor,
            border_type,
        )
    }
}
