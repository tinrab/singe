use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, template::SquareDistanceNormScaledImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareDistanceNormScaledImage<T, L>,
{
    pub fn square_distance_full_norm_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareDistanceNormScaledImage<T, L>>::full(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }

    pub fn square_distance_same_norm_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareDistanceNormScaledImage<T, L>>::same(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }

    pub fn square_distance_valid_norm_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareDistanceNormScaledImage<T, L>>::valid(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }
}
