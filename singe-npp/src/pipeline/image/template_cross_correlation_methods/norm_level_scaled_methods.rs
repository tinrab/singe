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
            CrossCorrelationNormLevelScaledImage, template_full_size, template_same_size,
            template_valid_size,
        },
    },
    types::Size,
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
    pub fn cross_correlation_full_norm_level_scaled(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_scaled(
            destination,
            template,
            template_full_size,
            scale_factor,
            <Self as CrossCorrelationNormLevelScaledImage<T, L>>::full,
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
    pub fn cross_correlation_same_norm_level_scaled(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_scaled(
            destination,
            template,
            template_same_size,
            scale_factor,
            <Self as CrossCorrelationNormLevelScaledImage<T, L>>::same,
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
    pub fn cross_correlation_valid_norm_level_scaled(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_norm_level_scaled(
            destination,
            template,
            template_valid_size,
            scale_factor,
            <Self as CrossCorrelationNormLevelScaledImage<T, L>>::valid,
        )
    }

    fn cross_correlation_norm_level_scaled(
        self,
        mut destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        scale_factor: i32,
        operation: CrossCorrelationNormLevelScaledCallback<L, T>,
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
                scale_factor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

type CrossCorrelationNormLevelScaledCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    i32,
) -> Result<()>;
