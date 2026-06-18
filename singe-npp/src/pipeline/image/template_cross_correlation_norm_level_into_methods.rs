use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, template::CrossCorrelationNormLevelImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormLevelImage<T, L>,
{
    pub fn cross_correlation_full_norm_level_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelImage<T, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_same_norm_level_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelImage<T, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_valid_norm_level_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelImage<T, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}
