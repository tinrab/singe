use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, template::CrossCorrelationNormLevelToImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn cross_correlation_full_norm_level_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: CrossCorrelationNormLevelToImage<T, D, L>,
    {
        <Self as CrossCorrelationNormLevelToImage<T, D, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_same_norm_level_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: CrossCorrelationNormLevelToImage<T, D, L>,
    {
        <Self as CrossCorrelationNormLevelToImage<T, D, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_valid_norm_level_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: CrossCorrelationNormLevelToImage<T, D, L>,
    {
        <Self as CrossCorrelationNormLevelToImage<T, D, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}
