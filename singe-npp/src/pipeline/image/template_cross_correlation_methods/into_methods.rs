use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{
        ImagePipeline,
        template::{CrossCorrelationImage, CrossCorrelationNormImage},
    },
};

mod norm_level_into_methods;
mod norm_level_scaled_into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormImage<T, L>,
{
    /// Computes full normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
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
    pub fn cross_correlation_full_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormImage<T, L>>::full(
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
    pub fn cross_correlation_same_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormImage<T, L>>::same(
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
    pub fn cross_correlation_valid_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormImage<T, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationImage<T, L>,
{
    /// Computes valid cross-correlation responses against `template`.
    ///
    /// Sums corresponding source and template sample products over each valid placement.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
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
    pub fn cross_correlation_valid_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationImage<T, L>>::valid(stream_context, source, template, destination)
    }
}
