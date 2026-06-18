use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::template::{
    CrossCorrelationNormLevelAdvancedImage, template_full_size, template_same_size,
    template_valid_size,
};
use super::{ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormLevelAdvancedImage<T, L>,
{
    pub fn cross_correlation_full_norm_level_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_same_norm_level_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    pub fn cross_correlation_valid_norm_level_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::valid(
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
    Workspace: ImageAllocator<T, L>,
    Self: CrossCorrelationNormLevelAdvancedImage<T, L>,
{
    pub fn cross_correlation_full_norm_level_advanced(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_advanced(
            template,
            template_full_size,
            <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::full,
        )
    }

    pub fn cross_correlation_same_norm_level_advanced(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_advanced(
            template,
            template_same_size,
            <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::same,
        )
    }

    pub fn cross_correlation_valid_norm_level_advanced(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_advanced(
            template,
            template_valid_size,
            <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::valid,
        )
    }

    fn cross_correlation_norm_level_advanced(
        self,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
        ) -> Result<()>,
    ) -> Result<Self> {
        let size = destination_size(self.size(), template.size())?;
        let mut destination = self.workspace.image::<T, L>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                template,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
