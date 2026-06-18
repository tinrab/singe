use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, template::CrossCorrelationToImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationToImage<T, f32, L>,
{
    pub fn cross_correlation_valid_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationToImage<T, f32, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn cross_correlation_valid_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: CrossCorrelationToImage<T, D, L>,
    {
        <Self as CrossCorrelationToImage<T, D, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}
