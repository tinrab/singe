use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, template::CrossCorrelationNormLevelScaledImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormLevelScaledImage<T, L>,
{
    /// Computes full mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Applies native integer result scaling by `2^(-scale_factor)`.
    /// Queries and allocates temporary device scratch for the native normalization.
    /// Retains scratch storage through native completion.
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
    /// Scratch-size queries, checked byte-size conversions, or scratch allocation can fail.
    pub fn cross_correlation_full_norm_level_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelScaledImage<T, L>>::full(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }

    /// Computes same mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Applies native integer result scaling by `2^(-scale_factor)`.
    /// Queries and allocates temporary device scratch for the native normalization.
    /// Retains scratch storage through native completion.
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
    /// Scratch-size queries, checked byte-size conversions, or scratch allocation can fail.
    pub fn cross_correlation_same_norm_level_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelScaledImage<T, L>>::same(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }

    /// Computes valid mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Applies native integer result scaling by `2^(-scale_factor)`.
    /// Queries and allocates temporary device scratch for the native normalization.
    /// Retains scratch storage through native completion.
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
    /// Scratch-size queries, checked byte-size conversions, or scratch allocation can fail.
    pub fn cross_correlation_valid_norm_level_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelScaledImage<T, L>>::valid(
            stream_context,
            source,
            template,
            destination,
            scale_factor,
        )
    }
}
