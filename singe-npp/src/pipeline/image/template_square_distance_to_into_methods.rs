use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, template::SquareDistanceNormToImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn square_distance_full_norm_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        <Self as SquareDistanceNormToImage<T, D, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn square_distance_same_norm_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        <Self as SquareDistanceNormToImage<T, D, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn square_distance_valid_norm_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        <Self as SquareDistanceNormToImage<T, D, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}
