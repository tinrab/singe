use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{
        ContourBlockSegment, ContourPixelDirectionInfo, ContourPixelGeometryInfo, Point32f,
        Point64f,
    },
};

use super::ImagePipeline;

impl<'a> ImagePipeline<'a, u8, C1> {
    #[allow(clippy::too_many_arguments)]
    pub fn contours_image_marching_squares_interpolation_32f(
        stream_context: &StreamContext,
        contours_image_dev: &ImageView<'_, u8, C1>,
        contours_interpolated_image_dev: &mut ImageViewMut<'_, Point32f, C1>,
        contours_direction_image_dev: &ImageView<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_geometry_lists_dev: &DeviceMemory<ContourPixelGeometryInfo>,
        contours_pixel_geometry_lists_host: &[ContourPixelGeometryInfo],
        contours_interpolated_geometry_lists_dev: &mut DeviceMemory<Point32f>,
        contours_pixels_found_list_host: &[u32],
        contours_pixels_starting_offset_dev: &DeviceMemory<u32>,
        contours_pixels_starting_offset_host: &[u32],
        total_image_pixel_contour_count: u32,
        max_marker_label_id: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
        contours_block_segment_list_dev: &DeviceMemory<ContourBlockSegment>,
        contours_block_segment_list_host: &[ContourBlockSegment],
    ) -> Result<()> {
        filtering::contours_image_marching_squares_interpolation_32f_c1(
            stream_context,
            contours_image_dev,
            contours_interpolated_image_dev,
            contours_direction_image_dev,
            contours_pixel_geometry_lists_dev,
            contours_pixel_geometry_lists_host,
            contours_interpolated_geometry_lists_dev,
            contours_pixels_found_list_host,
            contours_pixels_starting_offset_dev,
            contours_pixels_starting_offset_host,
            total_image_pixel_contour_count,
            max_marker_label_id,
            first_contour_geometry_list_id,
            last_contour_geometry_list_id,
            contours_block_segment_list_dev,
            contours_block_segment_list_host,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contours_image_marching_squares_interpolation_64f(
        stream_context: &StreamContext,
        contours_image_dev: &ImageView<'_, u8, C1>,
        contours_interpolated_image_dev: &mut ImageViewMut<'_, Point64f, C1>,
        contours_direction_image_dev: &ImageView<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_geometry_lists_dev: &DeviceMemory<ContourPixelGeometryInfo>,
        contours_pixel_geometry_lists_host: &[ContourPixelGeometryInfo],
        contours_interpolated_geometry_lists_dev: &mut DeviceMemory<Point64f>,
        contours_pixels_found_list_host: &[u32],
        contours_pixels_starting_offset_dev: &DeviceMemory<u32>,
        contours_pixels_starting_offset_host: &[u32],
        total_image_pixel_contour_count: u32,
        max_marker_label_id: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
        contours_block_segment_list_dev: &DeviceMemory<ContourBlockSegment>,
        contours_block_segment_list_host: &[ContourBlockSegment],
    ) -> Result<()> {
        filtering::contours_image_marching_squares_interpolation_64f_c1(
            stream_context,
            contours_image_dev,
            contours_interpolated_image_dev,
            contours_direction_image_dev,
            contours_pixel_geometry_lists_dev,
            contours_pixel_geometry_lists_host,
            contours_interpolated_geometry_lists_dev,
            contours_pixels_found_list_host,
            contours_pixels_starting_offset_dev,
            contours_pixels_starting_offset_host,
            total_image_pixel_contour_count,
            max_marker_label_id,
            first_contour_geometry_list_id,
            last_contour_geometry_list_id,
            contours_block_segment_list_dev,
            contours_block_segment_list_host,
        )
    }
}
