use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, PackedFixedThresholdImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn threshold_channels_less_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
    ) -> Result<()>
    where
        Self: PackedFixedThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedFixedThresholdImage<T, L, CHANNELS>>::threshold_channels_less_image(
            stream_context,
            source,
            destination,
            thresholds,
        )
    }

    pub fn threshold_channels_less_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
    ) -> Result<()>
    where
        Self: PackedFixedThresholdImage<T, L, CHANNELS>,
    {
        <Self as PackedFixedThresholdImage<T, L, CHANNELS>>::threshold_channels_less_image_in_place(
            stream_context,
            image,
            thresholds,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn threshold_channels_less<const CHANNELS: usize>(
        mut self,
        thresholds: [T; CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedFixedThresholdImage<T, L, CHANNELS>,
    {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as PackedFixedThresholdImage<
                    T,
                    L,
                    CHANNELS,
                >>::threshold_channels_less_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    thresholds,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as PackedFixedThresholdImage<T, L, CHANNELS>>::threshold_channels_less_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    thresholds,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
