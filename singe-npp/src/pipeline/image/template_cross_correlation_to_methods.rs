use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::template::{CrossCorrelationToImage, template_valid_size};
use super::super::{ImageBacking, ImagePipeline};

#[path = "template_cross_correlation_to_into_methods.rs"]
mod into_methods;
#[path = "template_cross_correlation_norm_to_methods.rs"]
mod norm_to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn cross_correlation_valid_to<D>(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, L>,
        Self: CrossCorrelationToImage<T, D, L>,
    {
        self.cross_correlation_single_output_to(
            template,
            <Self as CrossCorrelationToImage<T, D, L>>::valid,
        )
    }

    fn cross_correlation_single_output_to<D>(
        self,
        template: &ImageView<'_, T, L>,
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
        let size = template_valid_size(self.size(), template.size())?;
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

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<f32, L>,
    Self: CrossCorrelationToImage<T, f32, L>,
{
    pub fn cross_correlation_valid_to_f32(
        self,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_single_output_to_f32(
            template,
            <Self as CrossCorrelationToImage<T, f32, L>>::valid,
        )
    }

    fn cross_correlation_single_output_to_f32(
        self,
        template: &ImageView<'_, T, L>,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, f32, L>,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_single_output_to(template, operation)
    }
}
