use crate::{
    error::Result,
    image::{Image, view::ChannelLayout},
    pipeline::image::{
        ImagePipeline, WarpAffine, WarpPerspective, WarpQuad,
        geometry_dispatch::{
            WarpAffineBackImage, WarpAffineImage, WarpAffineQuadImage, WarpPerspectiveBackImage,
            WarpPerspectiveImage, WarpPerspectiveQuadImage,
        },
    },
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WarpAffineImage<T, L>,
{
    /// Applies the affine transformation and interpolation configured by `warp`.
    /// Transforms onto the supplied initialized `destination` and returns it as the pipeline
    /// backing. Does not allocate a replacement image. Uncovered pixels and ignored AC4 alpha
    /// retain their previous values.
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
    /// Allocation-retention setup and constructing source or destination views can also fail.
    pub fn warp_affine(self, warp: WarpAffine, destination: Image<T, L>) -> Result<Self> {
        self.transform_geometry(
            destination,
            &warp,
            <Self as WarpAffineImage<T, L>>::warp_affine_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WarpAffineBackImage<T, L>,
{
    /// Applies the NPP backward affine warp configured by `warp`.
    /// Transforms onto the supplied initialized `destination` and returns it as the pipeline
    /// backing. Does not allocate a replacement image. Uncovered pixels and ignored AC4 alpha
    /// retain their previous values.
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
    /// Allocation-retention setup and constructing source or destination views can also fail.
    pub fn warp_affine_back(self, warp: WarpAffine, destination: Image<T, L>) -> Result<Self> {
        self.transform_geometry(
            destination,
            &warp,
            <Self as WarpAffineBackImage<T, L>>::warp_affine_back_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WarpAffineQuadImage<T, L>,
{
    /// Applies an affine warp defined by the source and destination quadrangles in `warp`.
    /// Transforms onto the supplied initialized `destination` and returns it as the pipeline
    /// backing. Does not allocate a replacement image. Uncovered pixels and ignored AC4 alpha
    /// retain their previous values.
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
    /// Allocation-retention setup and constructing source or destination views can also fail.
    pub fn warp_affine_quad(self, warp: WarpQuad, destination: Image<T, L>) -> Result<Self> {
        self.transform_geometry(
            destination,
            &warp,
            <Self as WarpAffineQuadImage<T, L>>::warp_affine_quad_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WarpPerspectiveImage<T, L>,
{
    /// Applies the perspective transformation and interpolation configured by `warp`.
    /// Transforms onto the supplied initialized `destination` and returns it as the pipeline
    /// backing. Does not allocate a replacement image. Uncovered pixels and ignored AC4 alpha
    /// retain their previous values.
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
    /// Allocation-retention setup and constructing source or destination views can also fail.
    pub fn warp_perspective(self, warp: WarpPerspective, destination: Image<T, L>) -> Result<Self> {
        self.transform_geometry(
            destination,
            &warp,
            <Self as WarpPerspectiveImage<T, L>>::warp_perspective_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WarpPerspectiveBackImage<T, L>,
{
    /// Applies the NPP backward perspective warp configured by `warp`.
    /// Transforms onto the supplied initialized `destination` and returns it as the pipeline
    /// backing. Does not allocate a replacement image. Uncovered pixels and ignored AC4 alpha
    /// retain their previous values.
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
    /// Allocation-retention setup and constructing source or destination views can also fail.
    pub fn warp_perspective_back(
        self,
        warp: WarpPerspective,
        destination: Image<T, L>,
    ) -> Result<Self> {
        self.transform_geometry(
            destination,
            &warp,
            <Self as WarpPerspectiveBackImage<T, L>>::warp_perspective_back_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WarpPerspectiveQuadImage<T, L>,
{
    /// Applies a perspective warp defined by the source and destination quadrangles in `warp`.
    /// Transforms onto the supplied initialized `destination` and returns it as the pipeline
    /// backing. Does not allocate a replacement image. Uncovered pixels and ignored AC4 alpha
    /// retain their previous values.
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
    /// Allocation-retention setup and constructing source or destination views can also fail.
    pub fn warp_perspective_quad(self, warp: WarpQuad, destination: Image<T, L>) -> Result<Self> {
        self.transform_geometry(
            destination,
            &warp,
            <Self as WarpPerspectiveQuadImage<T, L>>::warp_perspective_quad_image,
        )
    }
}
