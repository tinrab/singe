use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::super::template::{
    SquareDistanceNormScaledImage, template_full_size, template_same_size, template_valid_size,
};
use super::{ImageBacking, ImagePipeline};

#[path = "template_square_distance_scaled_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SquareDistanceNormScaledImage<T, L>,
{
    pub fn square_distance_full_norm_scaled(
        self,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.template_match_scaled(
            template,
            template_full_size,
            scale_factor,
            <Self as SquareDistanceNormScaledImage<T, L>>::full,
        )
    }

    pub fn square_distance_same_norm_scaled(
        self,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.template_match_scaled(
            template,
            template_same_size,
            scale_factor,
            <Self as SquareDistanceNormScaledImage<T, L>>::same,
        )
    }

    pub fn square_distance_valid_norm_scaled(
        self,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.template_match_scaled(
            template,
            template_valid_size,
            scale_factor,
            <Self as SquareDistanceNormScaledImage<T, L>>::valid,
        )
    }

    fn template_match_scaled(
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
