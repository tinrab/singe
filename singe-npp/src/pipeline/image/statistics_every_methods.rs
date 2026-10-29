use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::image::{ImageBacking, ImagePipeline, statistics::EveryStatisticImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: EveryStatisticImage<T, L>,
{
    /// Combines corresponding active samples using the native per-sample maximum.
    /// Updates the supplied destination image in place; AC4 alpha remains unchanged.
    /// Prepares participating image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn max_every_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EveryStatisticImage<T, L>>::max_every(stream_context, source, source_destination)
    }

    /// Combines corresponding active samples using the native per-sample minimum.
    /// Updates the supplied destination image in place; AC4 alpha remains unchanged.
    /// Prepares participating image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn min_every_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EveryStatisticImage<T, L>>::min_every(stream_context, source, source_destination)
    }

    /// Combines corresponding active samples using the native per-sample maximum.
    /// Reuses owned backing; borrowed input is copied to a new owned image before updating
    /// it in place. Preserves AC4 alpha from the pipeline image.
    /// Prepares participating image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, copying, or view construction can also fail.
    pub fn max_every(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.every_statistic(other, <Self as EveryStatisticImage<T, L>>::max_every)
    }

    /// Combines corresponding active samples using the native per-sample minimum.
    /// Reuses owned backing; borrowed input is copied to a new owned image before updating
    /// it in place. Preserves AC4 alpha from the pipeline image.
    /// Prepares participating image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, copying, or view construction can also fail.
    pub fn min_every(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.every_statistic(other, <Self as EveryStatisticImage<T, L>>::min_every)
    }

    fn every_statistic(
        mut self,
        other: &ImageView<'_, T, L>,
        statistic: EveryStatisticCallback<L, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                statistic(self.stream_context, other, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::from_view(source)?;
                statistic(self.stream_context, other, &mut destination.view_mut()?)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

type EveryStatisticCallback<L, T> =
    fn(&StreamContext, &ImageView<'_, T, L>, &mut ImageViewMut<'_, T, L>) -> Result<()>;
