use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{
        ImagePipeline, WarpAffine, WarpPerspective, WarpQuad,
        geometry_dispatch::{
            WarpAffineBackImage, WarpAffineImage, WarpAffineQuadImage, WarpPerspectiveBackImage,
            WarpPerspectiveImage, WarpPerspectiveQuadImage,
        },
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies the affine transformation and interpolation configured by `warp`.
    /// Writes only the transformed source intersection into the configured destination ROI.
    /// Uncovered pixels and ignored AC4 alpha retain their previous initialization state;
    /// success does not establish initialization of the entire destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    /// Capture checks, preparing regions, producer waits, and stream synchronization can also fail.
    pub fn warp_affine_into(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpAffineImage<T, L>,
    {
        <Self as WarpAffineImage<T, L>>::warp_affine_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    /// Applies the NPP backward affine warp configured by `warp`.
    /// Writes only the transformed source intersection into the configured destination ROI.
    /// Uncovered pixels and ignored AC4 alpha retain their previous initialization state;
    /// success does not establish initialization of the entire destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    /// Capture checks, preparing regions, producer waits, and stream synchronization can also fail.
    pub fn warp_affine_back_into(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpAffineBackImage<T, L>,
    {
        <Self as WarpAffineBackImage<T, L>>::warp_affine_back_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    /// Applies an affine warp defined by the source and destination quadrangles in `warp`.
    /// Writes only the transformed source intersection into the configured destination ROI.
    /// Uncovered pixels and ignored AC4 alpha retain their previous initialization state;
    /// success does not establish initialization of the entire destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    /// Capture checks, preparing regions, producer waits, and stream synchronization can also fail.
    pub fn warp_affine_quad_into(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpAffineQuadImage<T, L>,
    {
        <Self as WarpAffineQuadImage<T, L>>::warp_affine_quad_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    /// Applies the perspective transformation and interpolation configured by `warp`.
    /// Writes only the transformed source intersection into the configured destination ROI.
    /// Uncovered pixels and ignored AC4 alpha retain their previous initialization state;
    /// success does not establish initialization of the entire destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    /// Capture checks, preparing regions, producer waits, and stream synchronization can also fail.
    pub fn warp_perspective_into(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpPerspectiveImage<T, L>,
    {
        <Self as WarpPerspectiveImage<T, L>>::warp_perspective_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    /// Applies the NPP backward perspective warp configured by `warp`.
    /// Writes only the transformed source intersection into the configured destination ROI.
    /// Uncovered pixels and ignored AC4 alpha retain their previous initialization state;
    /// success does not establish initialization of the entire destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    /// Capture checks, preparing regions, producer waits, and stream synchronization can also fail.
    pub fn warp_perspective_back_into(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpPerspectiveBackImage<T, L>,
    {
        <Self as WarpPerspectiveBackImage<T, L>>::warp_perspective_back_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    /// Applies a perspective warp defined by the source and destination quadrangles in `warp`.
    /// Writes only the transformed source intersection into the configured destination ROI.
    /// Uncovered pixels and ignored AC4 alpha retain their previous initialization state;
    /// success does not establish initialization of the entire destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    /// Capture checks, preparing regions, producer waits, and stream synchronization can also fail.
    pub fn warp_perspective_quad_into(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpPerspectiveQuadImage<T, L>,
    {
        <Self as WarpPerspectiveQuadImage<T, L>>::warp_perspective_quad_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }
}
