use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline, threshold_dispatch::PackedLessGreaterValueThresholdImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn threshold_channels_less_greater_value_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        lower_thresholds: [T; CHANNELS],
        lower_values: [T; CHANNELS],
        upper_thresholds: [T; CHANNELS],
        upper_values: [T; CHANNELS],
    ) -> Result<()>
    where
        Self: PackedLessGreaterValueThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedLessGreaterValueThresholdImage<
            T,
            L,
            CHANNELS,
        >>::threshold_channels_less_greater_value_image(
            stream_context,
            source,
            destination,
            lower_thresholds,
            lower_values,
            upper_thresholds,
            upper_values,
        )
    }

    pub fn threshold_channels_less_greater_value_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        lower_thresholds: [T; CHANNELS],
        lower_values: [T; CHANNELS],
        upper_thresholds: [T; CHANNELS],
        upper_values: [T; CHANNELS],
    ) -> Result<()>
    where
        Self: PackedLessGreaterValueThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedLessGreaterValueThresholdImage<
            T,
            L,
            CHANNELS,
        >>::threshold_channels_less_greater_value_image_in_place(
            stream_context,
            image,
            lower_thresholds,
            lower_values,
            upper_thresholds,
            upper_values,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn threshold_channels_less_greater_value<const CHANNELS: usize>(
        mut self,
        lower_thresholds: [T; CHANNELS],
        lower_values: [T; CHANNELS],
        upper_thresholds: [T; CHANNELS],
        upper_values: [T; CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLessGreaterValueThresholdImage<T, L, CHANNELS>,
    {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as PackedLessGreaterValueThresholdImage<
                    T,
                    L,
                    CHANNELS,
                >>::threshold_channels_less_greater_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    lower_thresholds,
                    lower_values,
                    upper_thresholds,
                    upper_values,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as PackedLessGreaterValueThresholdImage<
                    T,
                    L,
                    CHANNELS,
                >>::threshold_channels_less_greater_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    lower_thresholds,
                    lower_values,
                    upper_thresholds,
                    upper_values,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
