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
        template::{CrossCorrelationToImage, template_valid_size},
    },
};

mod into_methods;
mod norm_to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Computes valid cross-correlation responses against `template`.
    ///
    /// Sums corresponding source and template sample products over each valid placement.
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
    pub fn cross_correlation_valid_to<D>(
        self,
        destination: Image<D, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        Self: CrossCorrelationToImage<T, D, L>,
    {
        self.cross_correlation_single_output_to(
            destination,
            template,
            <Self as CrossCorrelationToImage<T, D, L>>::valid,
        )
    }

    fn cross_correlation_single_output_to<D>(
        self,
        mut destination: Image<D, L>,
        template: &ImageView<'_, T, L>,
        operation: CrossCorrelationSingleOutputToCallback<D, L, T>,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
    {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;
        let size = template_valid_size(self.size(), template.size())?;
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

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CrossCorrelationToImage<T, f32, L>,
{
    /// Computes valid cross-correlation responses against `template`.
    ///
    /// Sums corresponding source and template sample products over each valid placement.
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
    pub fn cross_correlation_valid_to_f32(
        self,
        destination: Image<f32, L>,
        template: &ImageView<'_, T, L>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_single_output_to_f32(
            destination,
            template,
            <Self as CrossCorrelationToImage<T, f32, L>>::valid,
        )
    }

    fn cross_correlation_single_output_to_f32(
        self,
        destination: Image<f32, L>,
        template: &ImageView<'_, T, L>,
        operation: CrossCorrelationSingleOutputToF32Callback<L, T>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.cross_correlation_single_output_to(destination, template, operation)
    }
}

type CrossCorrelationSingleOutputToCallback<D, L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, D, L>,
) -> Result<()>;

type CrossCorrelationSingleOutputToF32Callback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut ImageOutputView<'_, f32, L>,
) -> Result<()>;
