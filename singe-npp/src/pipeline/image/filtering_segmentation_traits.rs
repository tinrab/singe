use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    types::{ImageNormalization, WatershedSegmentBoundaryType},
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait WatershedSegmentImage<T> {
    fn segment_watershed_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, C1>,
        marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
/// Establish native union-find provenance for the exact destination pointer, context, and ROI.
pub unsafe trait LabelMarkersUfImage<T> {
    fn label_markers_uf_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait LabelMarkersUfBatchImage<T> {
    fn label_markers_uf_batch_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;

    fn label_markers_uf_batch_advanced_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;
}
