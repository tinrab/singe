use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point},
};

use super::super::{ImagePipeline, filtering::GaussPyramidBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: GaussPyramidBorderFilterImage<T, L>,
{
    pub fn filter_gauss_pyramid_layer_down_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()> {
        <Self as GaussPyramidBorderFilterImage<T, L>>::filter_gauss_pyramid_layer_down_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            rate,
            kernel,
            border_type,
        )
    }

    pub fn filter_gauss_pyramid_layer_up_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()> {
        <Self as GaussPyramidBorderFilterImage<T, L>>::filter_gauss_pyramid_layer_up_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            rate,
            kernel,
            border_type,
        )
    }
}
