use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, threshold_dispatch::FusedAbsoluteDifferenceThresholdImage,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: FusedAbsoluteDifferenceThresholdImage<T, L>,
{
    /// Computes native absolute differences and replaces those strictly above `threshold` with `value`.
    /// For each active sample, returns `value` when the absolute difference exceeds the
    /// threshold; otherwise returns the absolute difference. Both inputs must have matching
    /// pixel dimensions.
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
    pub fn fused_absolute_difference_threshold_greater_value_into(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FusedAbsoluteDifferenceThresholdImage<
            T,
            L,
        >>::fused_absolute_difference_threshold_greater_value_image(
            stream_context,
            source1,
            source2,
            destination,
            threshold,
            value,
        )
    }

    /// Computes native absolute differences and replaces those strictly above `threshold` with `value`.
    /// For each active sample, returns `value` when the absolute difference exceeds the
    /// threshold; otherwise returns the absolute difference. Both inputs must have matching
    /// pixel dimensions.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn fused_absolute_difference_threshold_greater_value_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FusedAbsoluteDifferenceThresholdImage<
            T,
            L,
        >>::fused_absolute_difference_threshold_greater_value_image_in_place(
            stream_context,
            source_destination,
            source2,
            threshold,
            value,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: FusedAbsoluteDifferenceThresholdImage<T, L>,
{
    /// Computes native absolute differences and replaces those strictly above `threshold` with `value`.
    /// For each active sample, returns `value` when the absolute difference exceeds the
    /// threshold; otherwise returns the absolute difference. Both inputs must have matching
    /// pixel dimensions.
    /// Reuses owned backing; borrowed input produces a new owned image. Preserves source
    /// alpha for AC4.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn fused_absolute_difference_threshold_greater_value(
        mut self,
        other: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as FusedAbsoluteDifferenceThresholdImage<
                    T,
                    L,
                >>::fused_absolute_difference_threshold_greater_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    other,
                    threshold,
                    value,
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
                <Self as FusedAbsoluteDifferenceThresholdImage<
                    T,
                    L,
                >>::fused_absolute_difference_threshold_greater_value_image(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    threshold,
                    value,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
