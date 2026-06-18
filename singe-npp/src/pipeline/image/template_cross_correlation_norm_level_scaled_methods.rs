use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::super::{
    ImageBacking, ImagePipeline,
    template::{
        CrossCorrelationNormLevelScaledImage, template_full_size, template_same_size,
        template_valid_size,
    },
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CrossCorrelationNormLevelScaledImage<T, L>,
{
    pub fn cross_correlation_full_norm_level_scaled(
        self,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_scaled(
            template,
            template_full_size,
            scale_factor,
            <Self as CrossCorrelationNormLevelScaledImage<T, L>>::full,
        )
    }

    pub fn cross_correlation_same_norm_level_scaled(
        self,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_scaled(
            template,
            template_same_size,
            scale_factor,
            <Self as CrossCorrelationNormLevelScaledImage<T, L>>::same,
        )
    }

    pub fn cross_correlation_valid_norm_level_scaled(
        self,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_scaled(
            template,
            template_valid_size,
            scale_factor,
            <Self as CrossCorrelationNormLevelScaledImage<T, L>>::valid,
        )
    }

    fn cross_correlation_norm_level_scaled(
        self,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        scale_factor: i32,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            i32,
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
                scale_factor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
