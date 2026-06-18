use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, operation_traits::AlphaPremultiplyImage};

impl<'a, T> ImagePipeline<'a, T, AC4>
where
    T: Copy,
    Self: AlphaPremultiplyImage<T>,
{
    pub fn alpha_premultiply_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn alpha_premultiply_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image_in_place(
            stream_context,
            source_destination,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, AC4>
where
    T: Copy,
    Workspace: ImageAllocator<T, AC4>,
    Self: AlphaPremultiplyImage<T>,
{
    pub fn alpha_premultiply(mut self) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, AC4>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
