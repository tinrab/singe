use crate::{
    error::Result,
    image::view::C1,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, threshold_dispatch::FixedThresholdImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: FixedThresholdImage<T>,
{
    pub fn threshold_greater(mut self, threshold: T) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as FixedThresholdImage<T>>::threshold_greater_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as FixedThresholdImage<T>>::threshold_greater_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn threshold_less(mut self, threshold: T) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as FixedThresholdImage<T>>::threshold_less_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as FixedThresholdImage<T>>::threshold_less_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
