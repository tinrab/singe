use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C4, ImageView, ImageViewMut},
};

use super::super::super::{
    ColorTwistBatchConstantsMatrix4, ImagePipeline, color::ColorTwistBatchWithConstantsImage,
};

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
    Self: ColorTwistBatchWithConstantsImage<T>,
{
    pub fn color_twist_batch_with_constants(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        destinations: &mut [ImageViewMut<'_, T, C4>],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchWithConstantsImage<T>>::color_twist_batch_with_constants_image(
            stream_context,
            sources,
            twists,
            destinations,
            min,
            max,
        )
    }

    pub fn color_twist_batch_with_constants_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchWithConstantsImage<
            T,
        >>::color_twist_batch_with_constants_image_in_place(stream_context, images, twists, min, max)
    }
}
