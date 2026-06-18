use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, threshold_dispatch::LessGreaterValueThresholdImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: LessGreaterValueThresholdImage<T>,
{
    pub fn threshold_less_greater_value_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<()> {
        <Self as LessGreaterValueThresholdImage<T>>::threshold_less_greater_value_image(
            stream_context,
            source,
            destination,
            lower_threshold,
            lower_value,
            upper_threshold,
            upper_value,
        )
    }

    pub fn threshold_less_greater_value_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<()> {
        <Self as LessGreaterValueThresholdImage<T>>::threshold_less_greater_value_image_in_place(
            stream_context,
            image,
            lower_threshold,
            lower_value,
            upper_threshold,
            upper_value,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: LessGreaterValueThresholdImage<T>,
{
    pub fn threshold_less_greater_value(
        mut self,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as LessGreaterValueThresholdImage<
                    T,
                >>::threshold_less_greater_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as LessGreaterValueThresholdImage<T>>::threshold_less_greater_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
