use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::ComparisonOperation,
};

use super::super::{ImageBacking, ImagePipeline, threshold_dispatch::PackedValueThresholdImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn threshold_channels_value_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        Self: PackedValueThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedValueThresholdImage<T, L, CHANNELS>>::threshold_channels_value_image(
            stream_context,
            source,
            destination,
            thresholds,
            values,
            operation,
        )
    }

    pub fn threshold_channels_value_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        Self: PackedValueThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedValueThresholdImage<T, L, CHANNELS>>::threshold_channels_value_image_in_place(
            stream_context,
            image,
            thresholds,
            values,
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
    pub fn threshold_channels_value<const CHANNELS: usize>(
        mut self,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<Self>
    where
        Self: PackedValueThresholdImage<T, L, CHANNELS>,
    {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as PackedValueThresholdImage<
                    T,
                    L,
                    CHANNELS,
                >>::threshold_channels_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    thresholds,
                    values,
                    operation,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as PackedValueThresholdImage<
                    T,
                    L,
                    CHANNELS,
                >>::threshold_channels_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    thresholds,
                    values,
                    operation,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
