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
            SquareDistanceNormImage, template_full_size, template_same_size, template_valid_size,
        },
    },
    types::Size,
};

mod scaled_methods;
mod to_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareDistanceNormImage<T, L>,
{
    /// Computes full normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
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
    pub fn square_distance_full_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormImage<T, L>>::full(stream_context, source, template, destination)
    }

    /// Computes same normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
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
    pub fn square_distance_same_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormImage<T, L>>::same(stream_context, source, template, destination)
    }

    /// Computes valid normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
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
    pub fn square_distance_valid_norm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as SquareDistanceNormImage<T, L>>::valid(
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
    Self: SquareDistanceNormImage<T, L>,
{
    /// Computes full normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
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
    pub fn square_distance_full_norm(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.template_match(
            destination,
            template,
            template_full_size,
            <Self as SquareDistanceNormImage<T, L>>::full,
        )
    }

    /// Computes same normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
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
    pub fn square_distance_same_norm(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.template_match(
            destination,
            template,
            template_same_size,
            <Self as SquareDistanceNormImage<T, L>>::same,
        )
    }

    /// Computes valid normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
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
    pub fn square_distance_valid_norm(
        self,
        destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<Self> {
        self.template_match(
            destination,
            template,
            template_valid_size,
            <Self as SquareDistanceNormImage<T, L>>::valid,
        )
    }

    fn template_match(
        self,
        mut destination: Image<T, L>,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: TemplateMatchCallback<L, T>,
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

type TemplateMatchCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
) -> Result<()>;
