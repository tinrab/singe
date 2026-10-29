use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait DistanceTransformPbaImage<T, D> {
    fn distance_transform_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, D, C1>,
    ) -> Result<()>;

    fn distance_transform_abs_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, D, C1>,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait DistanceTransformPbaAntialiasingImage<T> {
    fn distance_transform_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>;

    fn distance_transform_abs_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait SignedDistanceTransformPbaImage<T, D> {
    fn signed_distance_transform_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination: &mut ImageOutputView<'_, D, C1>,
    ) -> Result<()>;

    fn signed_distance_transform_abs_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination: &mut ImageOutputView<'_, D, C1>,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait SignedDistanceTransformPbaAntialiasingImage<T> {
    fn signed_distance_transform_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>;

    fn signed_distance_transform_abs_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination: &mut ImageOutputView<'_, f64, C1>,
    ) -> Result<()>;
}
