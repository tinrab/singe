use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, template::SquareDistanceNormToImage},
};

impl<T, L> ImagePipeline<'_, T, L>
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
    pub fn square_distance_full_norm_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        <Self as SquareDistanceNormToImage<T, D, L>>::full(
            stream_context,
            source,
            template,
            destination,
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
    pub fn square_distance_same_norm_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        <Self as SquareDistanceNormToImage<T, D, L>>::same(
            stream_context,
            source,
            template,
            destination,
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
    pub fn square_distance_valid_norm_to_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, L>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SquareDistanceNormToImage<T, D, L>,
    {
        <Self as SquareDistanceNormToImage<T, D, L>>::valid(
            stream_context,
            source,
            template,
            destination,
        )
    }
}
