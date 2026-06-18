use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{ImageNormalization, WatershedSegmentBoundaryType},
};

use super::ImagePipeline;
use super::filtering_traits::*;

macro_rules! impl_watershed_segment_image {
    ($ty:ty, $segment:path) => {
        impl<'a> WatershedSegmentImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn segment_watershed_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, C1>,
                marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
                norm: ImageNormalization,
                boundary_type: WatershedSegmentBoundaryType,
            ) -> Result<()> {
                $segment(
                    stream_context,
                    source_destination,
                    marker_labels,
                    norm,
                    boundary_type,
                )
            }
        }
    };
}

macro_rules! impl_label_markers_uf_image {
    ($ty:ty, $label:path) => {
        impl<'a> LabelMarkersUfImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn label_markers_uf_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, u32, C1>,
                norm: ImageNormalization,
            ) -> Result<()> {
                $label(stream_context, source, destination, norm)
            }
        }
    };
}

macro_rules! impl_label_markers_uf_batch_image {
    ($ty:ty, $label:path, $label_advanced:path) => {
        impl<'a> LabelMarkersUfBatchImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn label_markers_uf_batch_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, C1>],
                destinations: &mut [ImageViewMut<'_, u32, C1>],
                norm: ImageNormalization,
            ) -> Result<()> {
                $label(stream_context, sources, destinations, norm)
            }

            fn label_markers_uf_batch_advanced_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, C1>],
                destinations: &mut [ImageViewMut<'_, u32, C1>],
                norm: ImageNormalization,
            ) -> Result<()> {
                $label_advanced(stream_context, sources, destinations, norm)
            }
        }
    };
}

impl_watershed_segment_image!(u8, filtering::segment_watershed_u8_c1_in_place);
impl_watershed_segment_image!(u16, filtering::segment_watershed_u16_c1_in_place);

impl_label_markers_uf_image!(u8, filtering::label_markers_uf_u8_to_u32_c1);
impl_label_markers_uf_image!(u16, filtering::label_markers_uf_u16_to_u32_c1);
impl_label_markers_uf_image!(u32, filtering::label_markers_uf_u32_c1);

impl_label_markers_uf_batch_image!(
    u8,
    filtering::label_markers_uf_batch_u8_to_u32_c1,
    filtering::label_markers_uf_batch_u8_to_u32_c1_advanced
);
impl_label_markers_uf_batch_image!(
    u16,
    filtering::label_markers_uf_batch_u16_to_u32_c1,
    filtering::label_markers_uf_batch_u16_to_u32_c1_advanced
);
impl_label_markers_uf_batch_image!(
    u32,
    filtering::label_markers_uf_batch_u32_c1,
    filtering::label_markers_uf_batch_u32_c1_advanced
);
