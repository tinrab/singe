use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    pipeline::image::{
        Remap, ResizeSqrPixel, ResizeSqrPixelAdvanced, Rotate, WarpAffine, WarpPerspective,
        WarpQuad,
    },
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait RotateImage<T, L> {
    fn rotate_image(
        stream_context: &StreamContext,
        rotate: &Rotate,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait ResizeSqrPixelImage<T, L> {
    fn resize_sqr_pixel_image(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixel,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait ResizeSqrPixelAdvancedImage<T, L> {
    fn resize_sqr_pixel_advanced_image(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait RemapImage<T, M, L> {
    fn remap_image(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &ImageView<'_, T, L>,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait WarpAffineImage<T, L> {
    fn warp_affine_image(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait WarpAffineBackImage<T, L> {
    fn warp_affine_back_image(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait WarpAffineQuadImage<T, L> {
    fn warp_affine_quad_image(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait WarpPerspectiveImage<T, L> {
    fn warp_perspective_image(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait WarpPerspectiveBackImage<T, L> {
    fn warp_perspective_back_image(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait WarpPerspectiveQuadImage<T, L> {
    fn warp_perspective_quad_image(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}
