use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::threshold_channels_fixed_methods::{
        ImageBacking, ImagePipeline, PackedFixedThresholdImage,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Replaces samples strictly below their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// The provided C3 and AC4 implementations use three entries per parameter array,
    /// one per processed channel. AC4 alpha is excluded.
    /// Writes into the equally sized destination. AC4 alpha retains its previous
    /// initialization state; this call does not initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn threshold_channels_less_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
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

    /// Replaces samples strictly below their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// The provided C3 and AC4 implementations use three entries per parameter array,
    /// one per processed channel. AC4 alpha is excluded.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
{
    /// Replaces samples strictly below their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// The provided C3 and AC4 implementations use three entries per parameter array,
    /// one per processed channel. AC4 alpha is excluded.
    /// Reuses owned backing; borrowed input produces a new owned image. Preserves source
    /// alpha for AC4.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn threshold_channels_less<const CHANNELS: usize>(
        mut self,
        thresholds: [T; CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedFixedThresholdImage<T, L, CHANNELS>,
    {
        self.ensure_eager()?;

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
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                <Self as PackedFixedThresholdImage<T, L, CHANNELS>>::threshold_channels_less_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    thresholds,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
