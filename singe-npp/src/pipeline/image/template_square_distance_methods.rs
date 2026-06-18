use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::template::{
    SquareDistanceNormImage, template_full_size, template_same_size, template_valid_size,
};
use super::{ImageBacking, ImagePipeline};

#[path = "template_square_distance_scaled_methods.rs"]
mod scaled_methods;
#[path = "template_square_distance_to_methods.rs"]
mod to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareDistanceNormImage<T, L>,
{
    pub fn square_distance_full_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormImage<T, L>>::full(stream_context, source, template, destination)
    }

    pub fn square_distance_same_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormImage<T, L>>::same(stream_context, source, template, destination)
    }

    pub fn square_distance_valid_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormImage<T, L>>::valid(
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
    Self: SquareDistanceNormImage<T, L>,
{
    pub fn square_distance_full_norm(self, template: &ImageView<'_, T, L>) -> Result<Self> {
        self.template_match(
            template,
            template_full_size,
            <Self as SquareDistanceNormImage<T, L>>::full,
        )
    }

    pub fn square_distance_same_norm(self, template: &ImageView<'_, T, L>) -> Result<Self> {
        self.template_match(
            template,
            template_same_size,
            <Self as SquareDistanceNormImage<T, L>>::same,
        )
    }

    pub fn square_distance_valid_norm(self, template: &ImageView<'_, T, L>) -> Result<Self> {
        self.template_match(
            template,
            template_valid_size,
            <Self as SquareDistanceNormImage<T, L>>::valid,
        )
    }

    fn template_match(
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
