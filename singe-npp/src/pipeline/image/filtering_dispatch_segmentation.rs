use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImagePipeline,
        filtering_traits::{LabelMarkersUfBatchImage, LabelMarkersUfImage, WatershedSegmentImage},
    },
    types::{ImageNormalization, WatershedSegmentBoundaryType},
};

macro_rules! impl_watershed_segment_image {
    ($ty:ty, $segment:path) => {
        unsafe impl<'a> WatershedSegmentImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn segment_watershed_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, C1>,
                marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
                norm: ImageNormalization,
                boundary_type: WatershedSegmentBoundaryType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                if let Some(image) = marker_labels.as_ref() {
                    stream_context.prepare_region(*image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe {
                    $segment(
                        stream_context,
                        source_destination,
                        marker_labels,
                        norm,
                        boundary_type,
                    )
                }
            }
        }
    };
}

macro_rules! impl_label_markers_uf_image {
    ($ty:ty, $label:path) => {
        unsafe impl<'a> LabelMarkersUfImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn label_markers_uf_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, u32, C1>,
                norm: ImageNormalization,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $label(stream_context, source, destination, norm) }
            }
        }
    };
}

macro_rules! impl_label_markers_uf_batch_image {
    ($ty:ty, $label:path, $label_advanced:path) => {
        unsafe impl<'a> LabelMarkersUfBatchImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn label_markers_uf_batch_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, C1>],
                destinations: &mut [ImageOutputView<'_, u32, C1>],
                norm: ImageNormalization,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $label(stream_context, sources, destinations, norm) }
            }

            fn label_markers_uf_batch_advanced_image(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, $ty, C1>],
                destinations: &mut [ImageOutputView<'_, u32, C1>],
                norm: ImageNormalization,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $label_advanced(stream_context, sources, destinations, norm) }
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
