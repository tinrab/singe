use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    pipeline::image::{
        ImagePipeline, filtering_distance_traits::SignedDistanceTransformPbaAntialiasingImage,
    },
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
{
    /// Computes an antialiased signed Euclidean distance image with the parallel banding algorithm.
    ///
    /// Accepts C1 `f32` or `f64` coverage images: values at or below zero describe exterior
    /// pixels, fractional values between zero and one describe coverage, and values at or
    /// above one describe fully covered interior pixels. Values below `cutoff_value` are
    /// classified as fully outside regions. Distances are generally negative outside and
    /// positive inside. Horizontal and vertical subpixel offsets shift the final distances;
    /// fully covered pixel centers are at `(0.5, 0.5)`.
    /// Writes C1 `f64` distances and owns both transform and antialiasing device scratch
    /// through native completion.
    /// NPP requires width and height in `64..=32767`; the wrapper checks positive
    /// dimensions but does not enforce that native range.
    /// Requests only the Euclidean distance image; optional Manhattan and Voronoi outputs
    /// are disabled.
    /// Source and destination dimensions must match.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Prepares image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Writes every
    /// logical output sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, workspace-size queries, scratch allocation, native execution, or
    /// stream synchronization fail. Source and destination dimensions must match and
    /// dimensions must be positive. NPP may reject the supplied transform parameters.
    pub fn signed_distance_transform_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: SignedDistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as SignedDistanceTransformPbaAntialiasingImage<T>>::signed_distance_transform_pba_antialiasing_image(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination,
        )
    }

    /// Computes an antialiased signed Euclidean distance image with the parallel banding algorithm.
    ///
    /// Accepts C1 `f32` or `f64` coverage images: values at or below zero describe exterior
    /// pixels, fractional values between zero and one describe coverage, and values at or
    /// above one describe fully covered interior pixels. Values below `cutoff_value` are
    /// classified as fully outside regions. Distances are generally negative outside and
    /// positive inside. Horizontal and vertical subpixel offsets shift the final distances;
    /// fully covered pixel centers are at `(0.5, 0.5)`.
    /// Writes C1 `f64` distances and owns both transform and antialiasing device scratch
    /// through native completion.
    /// NPP requires width and height in `64..=32767`; the wrapper checks positive
    /// dimensions but does not enforce that native range.
    /// Uses the native absolute-Manhattan variant, but requests only the Euclidean distance
    /// image. The optional Manhattan and Voronoi outputs are disabled.
    /// Source and destination dimensions must match.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Prepares image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Writes every
    /// logical output sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, workspace-size queries, scratch allocation, native execution, or
    /// stream synchronization fail. Source and destination dimensions must match and
    /// dimensions must be positive. NPP may reject the supplied transform parameters.
    pub fn signed_distance_transform_abs_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: SignedDistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as SignedDistanceTransformPbaAntialiasingImage<T>>::signed_distance_transform_abs_pba_antialiasing_image(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination,
        )
    }
}
