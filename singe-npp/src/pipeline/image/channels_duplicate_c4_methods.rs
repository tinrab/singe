use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C4, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::ImageBacking;
use super::{DuplicateToC4Image, ImagePipeline};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: DuplicateToC4Image<T>,
{
    pub fn duplicate_to_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C4>,
    ) -> Result<()> {
        <Self as DuplicateToC4Image<T>>::duplicate_to_c4_image(stream_context, source, destination)
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C4>,
    Self: DuplicateToC4Image<T>,
{
    pub fn duplicate_to_c4(self) -> Result<ImagePipeline<'a, T, C4>> {
        let mut destination = self.workspace.image::<T, C4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as DuplicateToC4Image<T>>::duplicate_to_c4_image(
                self.stream_context,
                &source,
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
