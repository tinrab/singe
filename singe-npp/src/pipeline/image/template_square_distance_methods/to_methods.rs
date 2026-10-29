use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        Image,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        template::{
            SquareDistanceNormToImage, template_full_size, template_same_size, template_valid_size,
        },
        template_square_distance_methods::{ImageBacking, ImagePipeline},
    },
    types::Size,
};

mod f32_methods;
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Computes full normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
    /// Each output dimension must equal source dimension plus template dimension minus one.
    /// NPP centers the template anchor and zero-pads source samples beyond the image.
    /// Uses the destination sample type selected by the typed dispatch and retains the
    /// channel layout.
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
    pub fn square_distance_full_norm_to<D>(
        self,
        destination: Image<D, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        self.template_match_to(
            destination,
            template,
            template_full_size,
            <Self as SquareDistanceNormToImage<T, D, L>>::full,
        )
    }

    /// Computes same normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
    /// Output dimensions must equal the source dimensions. NPP starts the centered template
    /// anchor at the top-left source pixel and zero-pads out-of-image source samples.
    /// Uses the destination sample type selected by the typed dispatch and retains the
    /// channel layout.
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
    pub fn square_distance_same_norm_to<D>(
        self,
        destination: Image<D, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        self.template_match_to(
            destination,
            template,
            template_same_size,
            <Self as SquareDistanceNormToImage<T, D, L>>::same,
        )
    }

    /// Computes valid normalized squared-distance responses against `template`.
    ///
    /// Uses NPP's normalized sum of squared sample differences; smaller responses indicate
    /// a closer match.
    /// Each output dimension must equal source dimension minus template dimension plus one.
    /// The template must fit within the source; only placements without out-of-image
    /// source reads are evaluated.
    /// Uses the destination sample type selected by the typed dispatch and retains the
    /// channel layout.
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
    pub fn square_distance_valid_norm_to<D>(
        self,
        destination: Image<D, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        self.template_match_to(
            destination,
            template,
            template_valid_size,
            <Self as SquareDistanceNormToImage<T, D, L>>::valid,
        )
    }

    fn template_match_to<D>(
        self,
        mut destination: Image<D, L>,
        template: &ImageView<'_, T, L>,
        destination_size: fn(Size, Size) -> Result<Size>,
        operation: TemplateMatchToCallback<D, L, T>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
    {
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

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

type TemplateMatchToCallback<D, L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, D, L>,
) -> Result<()>;
