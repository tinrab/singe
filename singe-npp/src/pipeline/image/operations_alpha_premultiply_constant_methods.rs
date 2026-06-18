use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{AlphaPremultiplyConstantImage, ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AlphaPremultiplyConstantImage<T, L>,
    <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha: Copy,
{
    pub fn alpha_premultiply_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        alpha: <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyConstantImage<T, L>>::alpha_premultiply_constant_image(
            stream_context,
            source,
            alpha,
            destination,
        )
    }

    pub fn alpha_premultiply_constant_in_place(
        stream_context: &StreamContext,
        alpha: <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyConstantImage<T, L>>::alpha_premultiply_constant_image_in_place(
            stream_context,
            alpha,
            source_destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: AlphaPremultiplyConstantImage<T, L>,
    <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha: Copy,
{
    pub fn alpha_premultiply_constant(
        mut self,
        alpha: <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as AlphaPremultiplyConstantImage<
                    T,
                    L,
                >>::alpha_premultiply_constant_image_in_place(
                    self.stream_context,
                    alpha,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as AlphaPremultiplyConstantImage<T, L>>::alpha_premultiply_constant_image(
                    self.stream_context,
                    source,
                    alpha,
                    &mut destination_view,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
