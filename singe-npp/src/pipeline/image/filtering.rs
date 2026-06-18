use crate::{
    error::Result,
    image::{filtering, view::ChannelLayout},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, PointPolar, Size},
};

use super::{ImageBacking, ImagePipeline};

pub(super) use super::filtering_dispatch::*;

#[path = "filtering_compressed_marker_labels.rs"]
mod compressed_marker_labels;
#[path = "filtering_outputs.rs"]
mod outputs;

pub use compressed_marker_labels::CompressedMarkerLabels;
pub use outputs::{GradientVector, HistogramOfGradients, HoughLines};

impl<'a, T, L> ImagePipeline<'a, T, L> {
    pub fn label_markers_uf_buffer_size(roi: Size) -> Result<usize> {
        filtering::label_markers_uf_buffer_size_u32_c1(roi)
    }

    pub fn compress_marker_labels_uf_buffer_size(starting_number: i32) -> Result<usize> {
        filtering::compress_marker_labels_uf_buffer_size_u32_c1(starting_number)
    }

    pub fn flood_fill_buffer_size(roi: Size) -> Result<usize> {
        filtering::flood_fill_buffer_size(roi)
    }

    pub fn filter_canny_border_buffer_size(roi: Size) -> Result<usize> {
        filtering::filter_canny_border_buffer_size(roi)
    }

    pub fn filter_harris_corners_border_buffer_size(roi: Size) -> Result<usize> {
        filtering::filter_harris_corners_border_buffer_size(roi)
    }

    pub fn filter_hough_line_buffer_size(
        roi: Size,
        delta: PointPolar,
        max_line_count: i32,
    ) -> Result<usize> {
        filtering::filter_hough_line_buffer_size(roi, delta, max_line_count)
    }

    pub fn filter_gauss_pyramid_layer_down_border_destination_roi(
        source_roi: Size,
        rate: f32,
    ) -> Result<Size> {
        filtering::filter_gauss_pyramid_layer_down_border_dst_roi(source_roi, rate)
    }

    pub fn filter_gauss_pyramid_layer_up_border_destination_roi(
        source_roi: Size,
        rate: f32,
    ) -> Result<(Size, Size)> {
        filtering::filter_gauss_pyramid_layer_up_border_dst_roi(source_roi, rate)
    }

    pub fn distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
        filtering::distance_transform_pba_buffer_size(roi)
    }

    pub fn distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
        filtering::distance_transform_pba_antialiasing_buffer_size(roi)
    }

    pub fn signed_distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
        filtering::signed_distance_transform_pba_buffer_size(roi)
    }

    pub fn signed_distance_transform_pba_64f_buffer_size(roi: Size) -> Result<usize> {
        filtering::signed_distance_transform_pba_64f_buffer_size(roi)
    }

    pub fn signed_distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
        filtering::signed_distance_transform_pba_antialiasing_buffer_size(roi)
    }

    pub fn segment_watershed_u8_buffer_size(roi: Size) -> Result<usize> {
        filtering::segment_watershed_buffer_size_u8_c1(roi)
    }

    pub fn segment_watershed_u16_buffer_size(roi: Size) -> Result<usize> {
        filtering::segment_watershed_buffer_size_u16_c1(roi)
    }

    pub fn segment_watershed_buffer_size<U: filtering::SegmentWatershedC1InPlace>(
        roi: Size,
    ) -> Result<usize> {
        filtering::segment_watershed_buffer_size_for::<U>(roi)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: AdaptiveBoxThresholdBorderImage<T, L>,
{
    pub fn filter_threshold_adaptive_box_border(
        self,
        source_offset: Point,
        mask_size: Size,
        delta: f32,
        value_greater_than: T,
        value_less_or_equal: T,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AdaptiveBoxThresholdBorderImage<T, L>>::filter_threshold_adaptive_box_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                delta,
                value_greater_than,
                value_less_or_equal,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
