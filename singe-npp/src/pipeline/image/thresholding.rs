use crate::{
    error::Result,
    image::view::C1,
    pipeline::{ImageAllocator, Workspace},
    types::ComparisonOperation,
};

use super::{
    ImageBacking, ImagePipeline,
    threshold_dispatch::{ThresholdImage, ValueThresholdImage},
};

#[path = "thresholding_fixed_methods.rs"]
mod fixed_methods;
#[path = "thresholding_fixed_value_methods.rs"]
mod fixed_value_methods;
#[path = "thresholding_into_methods.rs"]
mod into_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: ThresholdImage<T>,
{
    pub fn threshold(mut self, threshold: T, operation: ComparisonOperation) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ThresholdImage<T>>::threshold_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                    operation,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ThresholdImage<T>>::threshold_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                    operation,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: ValueThresholdImage<T>,
{
    pub fn threshold_value(
        mut self,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ValueThresholdImage<T>>::threshold_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                    value,
                    operation,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ValueThresholdImage<T>>::threshold_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                    value,
                    operation,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
