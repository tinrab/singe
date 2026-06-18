use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::{ImageNormalization, WatershedSegmentBoundaryType},
};

pub trait WatershedSegmentImage<T> {
    fn segment_watershed_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, C1>,
        marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()>;
}

pub trait LabelMarkersUfImage<T> {
    fn label_markers_uf_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

pub trait LabelMarkersUfBatchImage<T> {
    fn label_markers_uf_batch_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;

    fn label_markers_uf_batch_advanced_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;
}
