use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering_traits::DistanceTransformPbaAntialiasingImage},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
{
    /// Computes an antialiased Euclidean distance image with the parallel banding algorithm.
    ///
    /// Source values in the inclusive interval `minimum_site_value..=maximum_site_value`
    /// are sites; NPP requires at least one site. The wrapper does not inspect source pixels
    /// or validate the site interval.
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
    pub fn distance_transform_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: DistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as DistanceTransformPbaAntialiasingImage<
            T,
        >>::distance_transform_pba_antialiasing_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }

    /// Computes an antialiased Euclidean distance image with the parallel banding algorithm.
    ///
    /// Source values in the inclusive interval `minimum_site_value..=maximum_site_value`
    /// are sites; NPP requires at least one site. The wrapper does not inspect source pixels
    /// or validate the site interval.
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
    pub fn distance_transform_abs_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: DistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as DistanceTransformPbaAntialiasingImage<
            T,
        >>::distance_transform_abs_pba_antialiasing_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }
}
