use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::ImageBacking;
use super::{DuplicateToAC4Image, ImagePipeline};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: DuplicateToAC4Image<T>,
{
    pub fn duplicate_to_ac4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()> {
        <Self as DuplicateToAC4Image<T>>::duplicate_to_ac4_image(
            stream_context,
            source,
            destination,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, AC4>,
    Self: DuplicateToAC4Image<T>,
{
    pub fn duplicate_to_ac4(self) -> Result<ImagePipeline<'a, T, AC4>> {
        let mut destination = self.workspace.image::<T, AC4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as DuplicateToAC4Image<T>>::duplicate_to_ac4_image(
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
