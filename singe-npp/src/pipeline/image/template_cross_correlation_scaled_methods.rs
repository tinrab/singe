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
            CrossCorrelationNormScaledImage, template_full_size, template_same_size,
            template_valid_size,
        },
    },
    types::Size,
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationNormScaledImage<T, L>,
{
    /// Computes full normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Applies native integer result scaling by `2^(-scale_factor)`.
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
    pub fn cross_correlation_full_norm_scaled(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_scaled(
            destination,
            template,
            template_full_size,
            scale_factor,
            <Self as CrossCorrelationNormScaledImage<T, L>>::full,
        )
    }

    /// Computes same normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Applies native integer result scaling by `2^(-scale_factor)`.
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
    pub fn cross_correlation_same_norm_scaled(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_scaled(
            destination,
            template,
            template_same_size,
            scale_factor,
            <Self as CrossCorrelationNormScaledImage<T, L>>::same,
        )
    }

    /// Computes valid normalized cross-correlation responses against `template`.
    ///
    /// Uses the native normalized sum of corresponding source/template products, without
    /// the mean subtraction used by `NormLevel`.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Applies native integer result scaling by `2^(-scale_factor)`.
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
    pub fn cross_correlation_valid_norm_scaled(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.cross_correlation_scaled(
            destination,
            template,
            template_valid_size,
            scale_factor,
            <Self as CrossCorrelationNormScaledImage<T, L>>::valid,
        )
    }

    fn cross_correlation_scaled(
        self,
        mut destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        scale_factor: i32,
        operation: CrossCorrelationScaledCallback<L, T>,
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

type CrossCorrelationScaledCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    i32,
) -> Result<()>;
