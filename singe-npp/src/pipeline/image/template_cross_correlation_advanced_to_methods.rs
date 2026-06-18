use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::template::{
    CrossCorrelationNormLevelAdvancedToImage, template_full_size, template_same_size,
    template_valid_size,
};
use super::{ImageBacking, ImagePipeline};

#[path = "template_cross_correlation_advanced_to_f32_methods.rs"]
mod f32_methods;
#[path = "template_cross_correlation_advanced_to_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn cross_correlation_full_norm_level_advanced_to<D>(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, L>,
        Self: CrossCorrelationNormLevelAdvancedToImage<T, D, L>,
    {
        self.cross_correlation_norm_level_advanced_to(
            template,
            template_full_size,
            <Self as CrossCorrelationNormLevelAdvancedToImage<T, D, L>>::full,
        )
    }

    pub fn cross_correlation_same_norm_level_advanced_to<D>(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, L>,
        Self: CrossCorrelationNormLevelAdvancedToImage<T, D, L>,
    {
        self.cross_correlation_norm_level_advanced_to(
            template,
            template_same_size,
            <Self as CrossCorrelationNormLevelAdvancedToImage<T, D, L>>::same,
        )
    }

    pub fn cross_correlation_valid_norm_level_advanced_to<D>(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, L>,
        Self: CrossCorrelationNormLevelAdvancedToImage<T, D, L>,
    {
        self.cross_correlation_norm_level_advanced_to(
            template,
            template_valid_size,
            <Self as CrossCorrelationNormLevelAdvancedToImage<T, D, L>>::valid,
        )
    }

    fn cross_correlation_norm_level_advanced_to<D>(
        self,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, D, L>,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, L>,
    {
        let size = destination_size(self.size(), template.size())?;
        let mut destination = self.workspace.image::<D, L>(size)?;

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

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
