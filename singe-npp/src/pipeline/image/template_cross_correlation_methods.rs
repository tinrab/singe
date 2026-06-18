use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::template::{
    CrossCorrelationImage, CrossCorrelationNormLevelImage, template_full_size, template_same_size,
    template_valid_size,
};
use super::{ImageBacking, ImagePipeline};

#[path = "template_cross_correlation_into_methods.rs"]
mod into_methods;
#[path = "template_cross_correlation_norm_level_scaled_methods.rs"]
mod norm_level_scaled_methods;
#[path = "template_cross_correlation_norm_methods.rs"]
mod norm_methods;
#[path = "template_cross_correlation_to_methods.rs"]
mod to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CrossCorrelationNormLevelImage<T, L>,
{
    pub fn cross_correlation_full_norm_level(self, template: &ImageView<'_, T, L>) -> Result<Self> {
        self.cross_correlation_norm_level(
            template,
            template_full_size,
            <Self as CrossCorrelationNormLevelImage<T, L>>::full,
        )
    }

    pub fn cross_correlation_same_norm_level(self, template: &ImageView<'_, T, L>) -> Result<Self> {
        self.cross_correlation_norm_level(
            template,
            template_same_size,
            <Self as CrossCorrelationNormLevelImage<T, L>>::same,
        )
    }

    pub fn cross_correlation_valid_norm_level(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level(
            template,
            template_valid_size,
            <Self as CrossCorrelationNormLevelImage<T, L>>::valid,
        )
    }

    fn cross_correlation_norm_level(
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

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CrossCorrelationImage<T, L>,
{
    pub fn cross_correlation_valid(self, template: &ImageView<'_, T, L>) -> Result<Self> {
        self.cross_correlation_single_output(template, <Self as CrossCorrelationImage<T, L>>::valid)
    }

    fn cross_correlation_single_output(
        self,
        template: &ImageView<'_, T, L>,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
        ) -> Result<()>,
    ) -> Result<Self> {
        let size = template_valid_size(self.size(), template.size())?;
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
