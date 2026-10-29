use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        Image,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        template::{
            CrossCorrelationNormLevelAdvancedImage, template_full_size, template_same_size,
            template_valid_size,
        },
    },
    types::Size,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormLevelAdvancedImage<T, L>,
{
    /// Computes full mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Uses the advanced native algorithm for large templates, with both ordinary and
    /// advanced device scratch buffers.
    /// Retains scratch storage through native completion.
    /// NPP recommends an output width divisible by four for maximum performance.
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
    pub fn cross_correlation_full_norm_level_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::full(
            stream_context,
            source,
            template,
            destination,
        )
    }

    /// Computes same mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Uses the advanced native algorithm for large templates, with both ordinary and
    /// advanced device scratch buffers.
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
    pub fn cross_correlation_same_norm_level_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::same(
            stream_context,
            source,
            template,
            destination,
        )
    }

    /// Computes valid mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Uses the advanced native algorithm for large templates, with both ordinary and
    /// advanced device scratch buffers.
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
    pub fn cross_correlation_valid_norm_level_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::valid(
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
    Self: CrossCorrelationNormLevelAdvancedImage<T, L>,
{
    /// Computes full mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Uses the advanced native algorithm for large templates, with both ordinary and
    /// advanced device scratch buffers.
    /// Retains scratch storage through native completion.
    /// NPP recommends an output width divisible by four for maximum performance.
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
    /// Scratch-size queries, checked byte-size conversions, or scratch allocation can fail.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn cross_correlation_full_norm_level_advanced(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_advanced(
            destination,
            template,
            template_full_size,
            <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::full,
        )
    }

    /// Computes same mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Uses the advanced native algorithm for large templates, with both ordinary and
    /// advanced device scratch buffers.
    /// Retains scratch storage through native completion.
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
    /// Scratch-size queries, checked byte-size conversions, or scratch allocation can fail.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn cross_correlation_same_norm_level_advanced(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_advanced(
            destination,
            template,
            template_same_size,
            <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::same,
        )
    }

    /// Computes valid mean-subtracted normalized cross-correlation coefficient responses against `template`.
    ///
    /// Uses NPP `NormLevel` normalization, which removes local source and template means
    /// before computing the normalized correlation coefficient.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Uses the advanced native algorithm for large templates, with both ordinary and
    /// advanced device scratch buffers.
    /// Retains scratch storage through native completion.
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
    /// Scratch-size queries, checked byte-size conversions, or scratch allocation can fail.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn cross_correlation_valid_norm_level_advanced(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_advanced(
            destination,
            template,
            template_valid_size,
            <Self as CrossCorrelationNormLevelAdvancedImage<T, L>>::valid,
        )
    }

    fn cross_correlation_norm_level_advanced(
        self,
        mut destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: CrossCorrelationNormLevelAdvancedCallback<L, T>,
    ) -> Result<Self> {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;
        let size = destination_size(self.size(), template.size())?;
        if destination.size() != size {
            return Err(Error::SizeMismatch {
                name: "template destination".into(),
                expected: size,
                actual: destination.size(),
            });
        }

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            operation(
                self.stream_context,
                &source,
                template,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

type CrossCorrelationNormLevelAdvancedCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
) -> Result<()>;
