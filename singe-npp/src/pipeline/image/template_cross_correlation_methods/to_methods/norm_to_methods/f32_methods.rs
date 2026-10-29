use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::template_cross_correlation_methods::to_methods::norm_to_methods::{
        CrossCorrelationNormToImage, ImagePipeline, template_full_size, template_same_size,
        template_valid_size,
    },
    types::Size,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormToImage<T, f32, L>,
{
    /// Computes full normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Writes `f32` response samples while retaining the channel layout.
    /// For alpha-ignoring layouts, destination alpha remains unwritten.
    /// Prepares source, template, and destination regions, waits for producers on other
    /// streams, rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked response-size arithmetic, native template matching, or
    /// stream synchronization fail. The destination must have the exact dimensions
    /// required by the selected response mode. NPP can reject native geometry or numerical
    /// parameters.
    pub fn cross_correlation_full_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormToImage<T, f32, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    /// Computes same normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Writes `f32` response samples while retaining the channel layout.
    /// For alpha-ignoring layouts, destination alpha remains unwritten.
    /// Prepares source, template, and destination regions, waits for producers on other
    /// streams, rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked response-size arithmetic, native template matching, or
    /// stream synchronization fail. The destination must have the exact dimensions
    /// required by the selected response mode. NPP can reject native geometry or numerical
    /// parameters.
    pub fn cross_correlation_same_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormToImage<T, f32, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    /// Computes valid normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Writes `f32` response samples while retaining the channel layout.
    /// For alpha-ignoring layouts, destination alpha remains unwritten.
    /// Prepares source, template, and destination regions, waits for producers on other
    /// streams, rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked response-size arithmetic, native template matching, or
    /// stream synchronization fail. The destination must have the exact dimensions
    /// required by the selected response mode. NPP can reject native geometry or numerical
    /// parameters.
    pub fn cross_correlation_valid_norm_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormToImage<T, f32, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormToImage<T, f32, L>,
{
    /// Computes full normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Writes `f32` response samples while retaining the channel layout.
    /// Takes ownership of the supplied initialized destination and returns it in the
    /// resulting pipeline. Its dimensions must exactly match the selected response mode.
    /// Ignored alpha retains its destination value.
    /// Prepares source, template, and destination regions, waits for producers on other
    /// streams, rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked response-size arithmetic, native template matching, or
    /// stream synchronization fail. The destination must have the exact dimensions
    /// required by the selected response mode. NPP can reject native geometry or numerical
    /// parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn cross_correlation_full_norm_to_f32(
        self,
        destination: Image<f32, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to_f32(
            destination,
            template,
            template_full_size,
            <Self as CrossCorrelationNormToImage<T, f32, L>>::full,
        )
    }

    /// Computes same normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Writes `f32` response samples while retaining the channel layout.
    /// Takes ownership of the supplied initialized destination and returns it in the
    /// resulting pipeline. Its dimensions must exactly match the selected response mode.
    /// Ignored alpha retains its destination value.
    /// Prepares source, template, and destination regions, waits for producers on other
    /// streams, rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked response-size arithmetic, native template matching, or
    /// stream synchronization fail. The destination must have the exact dimensions
    /// required by the selected response mode. NPP can reject native geometry or numerical
    /// parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn cross_correlation_same_norm_to_f32(
        self,
        destination: Image<f32, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to_f32(
            destination,
            template,
            template_same_size,
            <Self as CrossCorrelationNormToImage<T, f32, L>>::same,
        )
    }

    /// Computes valid normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Writes `f32` response samples while retaining the channel layout.
    /// Takes ownership of the supplied initialized destination and returns it in the
    /// resulting pipeline. Its dimensions must exactly match the selected response mode.
    /// Ignored alpha retains its destination value.
    /// Prepares source, template, and destination regions, waits for producers on other
    /// streams, rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked response-size arithmetic, native template matching, or
    /// stream synchronization fail. The destination must have the exact dimensions
    /// required by the selected response mode. NPP can reject native geometry or numerical
    /// parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn cross_correlation_valid_norm_to_f32(
        self,
        destination: Image<f32, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to_f32(
            destination,
            template,
            template_valid_size,
            <Self as CrossCorrelationNormToImage<T, f32, L>>::valid,
        )
    }

    fn cross_correlation_to_f32(
        self,
        destination: Image<f32, L>,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: CrossCorrelationToF32Callback<L, T>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_to(destination, template, destination_size, operation)
    }
}

type CrossCorrelationToF32Callback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, f32, L>,
) -> Result<()>;
