use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, template::CrossCorrelationNormScaledImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormScaledImage<T, L>,
{
    pub fn cross_correlation_full_norm_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as CrossCorrelationNormScaledImage<T, L>>::full(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }

    pub fn cross_correlation_same_norm_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as CrossCorrelationNormScaledImage<T, L>>::same(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }

    pub fn cross_correlation_valid_norm_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as CrossCorrelationNormScaledImage<T, L>>::valid(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }
}
