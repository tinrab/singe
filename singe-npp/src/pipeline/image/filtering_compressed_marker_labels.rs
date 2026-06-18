use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{
        filtering,
        view::{C1, ImageViewMut},
    },
    types::{
        CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelDirectionInfo,
        ContourPixelGeometryInfo, ContourTotalsInfo,
    },
};

use super::ImagePipeline;

pub struct CompressedMarkerLabels<'a> {
    pub labels: ImagePipeline<'a, u32, C1>,
    pub new_max_label_id: u32,
}

impl<'a> CompressedMarkerLabels<'a> {
    pub fn info_list_size(&self) -> Result<usize> {
        filtering::compressed_marker_labels_uf_info_list_size_u32_c1(self.new_max_label_id)
    }

    pub fn geometry_lists_size(max_contour_pixel_geometry_info_count: u32) -> Result<usize> {
        filtering::compressed_marker_labels_uf_geometry_lists_size_c1(
            max_contour_pixel_geometry_info_count,
        )
    }

    pub fn contours_block_segment_list_size(
        contours_pixel_counts_list_host: &mut [u32],
        total_image_pixel_contour_count: u32,
        compressed_label_count: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
    ) -> Result<usize> {
        filtering::compressed_marker_labels_uf_contours_block_segment_list_size_c1(
            contours_pixel_counts_list_host,
            total_image_pixel_contour_count,
            compressed_label_count,
            first_contour_geometry_list_id,
            last_contour_geometry_list_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn info_into(
        &self,
        marker_labels_info_list: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
        contours_image: &mut ImageViewMut<'_, u8, C1>,
        contours_direction_image: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
        contours_pixel_counts_list_host: &mut [u32],
        contours_pixel_starting_offset_dev: &mut DeviceMemory<u32>,
        contours_pixel_starting_offset_host: &mut [u32],
    ) -> Result<ContourTotalsInfo> {
        let source = self.labels.view()?;
        let mut contours_totals = ContourTotalsInfo::default();
        filtering::compressed_marker_labels_uf_info_u32_c1(
            self.labels.stream_context,
            &source,
            self.new_max_label_id,
            marker_labels_info_list,
            contours_image,
            contours_direction_image,
            &mut contours_totals,
            contours_pixel_counts_list_dev,
            contours_pixel_counts_list_host,
            contours_pixel_starting_offset_dev,
            contours_pixel_starting_offset_host,
        )?;
        Ok(contours_totals)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_contour_geometry_lists_into(
        &self,
        marker_labels_info_list_dev: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
        marker_labels_info_list_host: &mut [CompressedMarkerLabelsInfo],
        contours_direction_image_dev: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_geometry_lists_dev: &mut DeviceMemory<ContourPixelGeometryInfo>,
        contours_pixel_geometry_lists_host: &mut [ContourPixelGeometryInfo],
        contours_geometry_image_host: &mut [u8],
        contours_geometry_image_step: i32,
        contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
        contours_pixels_found_list_dev: &mut DeviceMemory<u32>,
        contours_pixels_found_list_host: &mut [u32],
        contours_pixels_starting_offset_dev: &mut DeviceMemory<u32>,
        contours_pixels_starting_offset_host: &mut [u32],
        total_image_pixel_contour_count: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
        contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
        contours_block_segment_list_host: &mut [ContourBlockSegment],
        output_in_counterclockwise_order: bool,
    ) -> Result<()> {
        filtering::compressed_marker_labels_uf_contours_generate_geometry_lists_c1(
            self.labels.stream_context,
            marker_labels_info_list_dev,
            marker_labels_info_list_host,
            contours_direction_image_dev,
            contours_pixel_geometry_lists_dev,
            contours_pixel_geometry_lists_host,
            contours_geometry_image_host,
            contours_geometry_image_step,
            contours_pixel_counts_list_dev,
            contours_pixels_found_list_dev,
            contours_pixels_found_list_host,
            contours_pixels_starting_offset_dev,
            contours_pixels_starting_offset_host,
            total_image_pixel_contour_count,
            self.new_max_label_id,
            first_contour_geometry_list_id,
            last_contour_geometry_list_id,
            contours_block_segment_list_dev,
            contours_block_segment_list_host,
            output_in_counterclockwise_order,
        )
    }
}
