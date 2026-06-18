use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ColorTwistMatrix, ImagePipeline, color::ColorTwistBatchImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ColorTwistBatchImage<T, L>,
{
    pub fn color_twist_batch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, L>],
        twists: &[ColorTwistMatrix],
        destinations: &mut [ImageViewMut<'_, T, L>],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchImage<T, L>>::color_twist_batch_image(
            stream_context,
            sources,
            twists,
            destinations,
            min,
            max,
        )
    }

    pub fn color_twist_batch_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, T, L>],
        twists: &[ColorTwistMatrix],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchImage<T, L>>::color_twist_batch_image_in_place(
            stream_context,
            images,
            twists,
            min,
            max,
        )
    }
}
