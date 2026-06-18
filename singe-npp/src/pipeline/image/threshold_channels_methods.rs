use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::ComparisonOperation,
};

use super::{ImageBacking, ImagePipeline, threshold_dispatch::PackedThresholdImage};

#[path = "threshold_channels_value_methods.rs"]
mod value_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn threshold_channels_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        Self: PackedThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedThresholdImage<T, L, CHANNELS>>::threshold_channels_image(
            stream_context,
            source,
            destination,
            thresholds,
            operation,
        )
    }

    pub fn threshold_channels_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        Self: PackedThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedThresholdImage<T, L, CHANNELS>>::threshold_channels_image_in_place(
            stream_context,
            image,
            thresholds,
            operation,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn threshold_channels<const CHANNELS: usize>(
        mut self,
        thresholds: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<Self>
    where
        Self: PackedThresholdImage<T, L, CHANNELS>,
    {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as PackedThresholdImage<T, L, CHANNELS>>::threshold_channels_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    thresholds,
                    operation,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as PackedThresholdImage<T, L, CHANNELS>>::threshold_channels_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    thresholds,
                    operation,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
