use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{
    ImagePipeline, SquareDistanceNormToImage, template_full_size, template_same_size,
    template_valid_size,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareDistanceNormToImage<T, f32, L>,
{
    pub fn square_distance_full_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormToImage<T, f32, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn square_distance_same_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormToImage<T, f32, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn square_distance_valid_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormToImage<T, f32, L>>::valid(
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
    Workspace: ImageAllocator<f32, L>,
    Self: SquareDistanceNormToImage<T, f32, L>,
{
    pub fn square_distance_full_norm_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.template_match_to_f32(
            template,
            template_full_size,
            <Self as SquareDistanceNormToImage<T, f32, L>>::full,
        )
    }

    pub fn square_distance_same_norm_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.template_match_to_f32(
            template,
            template_same_size,
            <Self as SquareDistanceNormToImage<T, f32, L>>::same,
        )
    }

    pub fn square_distance_valid_norm_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.template_match_to_f32(
            template,
            template_valid_size,
            <Self as SquareDistanceNormToImage<T, f32, L>>::valid,
        )
    }

    fn template_match_to_f32(
        self,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, f32, L>,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.template_match_to(template, destination_size, operation)
    }
}
