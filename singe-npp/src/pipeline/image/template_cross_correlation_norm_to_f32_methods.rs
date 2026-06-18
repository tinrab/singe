use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{
    CrossCorrelationNormToImage, ImagePipeline, template_full_size, template_same_size,
    template_valid_size,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormToImage<T, f32, L>,
{
    pub fn cross_correlation_full_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormToImage<T, f32, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_same_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormToImage<T, f32, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_valid_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormToImage<T, f32, L>>::valid(
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
    Self: CrossCorrelationNormToImage<T, f32, L>,
{
    pub fn cross_correlation_full_norm_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to_f32(
            template,
            template_full_size,
            <Self as CrossCorrelationNormToImage<T, f32, L>>::full,
        )
    }

    pub fn cross_correlation_same_norm_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to_f32(
            template,
            template_same_size,
            <Self as CrossCorrelationNormToImage<T, f32, L>>::same,
        )
    }

    pub fn cross_correlation_valid_norm_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to_f32(
            template,
            template_valid_size,
            <Self as CrossCorrelationNormToImage<T, f32, L>>::valid,
        )
    }

    fn cross_correlation_to_f32(
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
        self.cross_correlation_to(template, destination_size, operation)
    }
}
