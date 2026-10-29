use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::image::filtering_segmentation_methods::ImagePipeline,
    types::{
        ContourBlockSegment, ContourPixelDirectionInfo, ContourPixelGeometryInfo, Point32f,
        Point64f,
    },
};

impl ImagePipeline<'_, u8, C1> {
    /// Interpolates matching contour geometry with marching squares using `Point32f` coordinates.
    ///
    /// Updates the supplied image and device geometry outputs for the selected contour
    /// ID range. All three images must have matching dimensions. Uses the corresponding
    /// host/device geometry, count, offset, and block-segment lists from completed native
    /// contour generation. Sparse output locations outside native writes retain their
    /// previous values. Prepares device regions and waits for their producers on other
    /// streams. Rejects capture and completes native use of host and device buffers before
    /// returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native interpolation, or synchronization fail. Image dimensions
    /// must match and IDs must satisfy `first < last <= max_marker_label_id`. The host
    /// offset sentinel at `max_marker_label_id` must exist. Host/device geometry lengths
    /// must match and cover the native geometry-size query for that sentinel. Found-count
    /// and starting-offset lists must cover all labels; the host offsets also need the
    /// sentinel. The interpolated device geometry list must have exactly
    /// `total_image_pixel_contour_count` elements, and host/device block-segment lengths
    /// must match. Checked size conversions and native size queries can also fail.
    ///
    /// # Safety
    ///
    /// Every image, geometry list, count, offset, and range must correspond to the same
    /// completed native contour generation. Supply matching query-sized block-segment
    /// storage; equal host/device lengths alone do not establish adequate capacity.
    /// The host offset sentinel must be ready for immediate reading. Source and metadata
    /// storage passed through mutable native pointers must be exclusively available.
    /// Keep all native-writable regions disjoint from other participating allocations,
    /// initialized wherever the operation reads or preserves values, and valid through
    /// completion. Host allocations must satisfy native host-memory requirements.
    pub unsafe fn contours_image_marching_squares_interpolation_32f(
        stream_context: &StreamContext,
        contours_image_dev: &ImageView<'_, u8, C1>,
        contours_interpolated_image_dev: &mut ImageViewMut<'_, Point32f, C1>,
        contours_direction_image_dev: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_geometry_lists_dev: &mut DeviceMemory<ContourPixelGeometryInfo>,
        contours_pixel_geometry_lists_host: &mut [ContourPixelGeometryInfo],
        contours_interpolated_geometry_lists_dev: &mut DeviceMemory<Point32f>,
        contours_pixels_found_list_host: &mut [u32],
        contours_pixels_starting_offset_dev: &mut DeviceMemory<u32>,
        contours_pixels_starting_offset_host: &mut [u32],
        total_image_pixel_contour_count: u32,
        max_marker_label_id: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
        contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
        contours_block_segment_list_host: &mut [ContourBlockSegment],
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(contours_image_dev)?;
        stream_context.prepare_region(contours_interpolated_image_dev)?;
        stream_context.prepare_region(contours_direction_image_dev)?;
        stream_context.prepare_region(contours_pixel_geometry_lists_dev)?;
        stream_context.prepare_region(contours_interpolated_geometry_lists_dev)?;
        stream_context.prepare_region(contours_pixels_starting_offset_dev)?;
        stream_context.prepare_region(contours_block_segment_list_dev)?;
        // SAFETY: the caller upholds this function's safety contract.
        stream_context.sync_scope(|| unsafe {
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
        })
    }

    /// Interpolates matching contour geometry with marching squares using `Point64f` coordinates.
    ///
    /// Updates the supplied image and device geometry outputs for the selected contour
    /// ID range. All three images must have matching dimensions. Uses the corresponding
    /// host/device geometry, count, offset, and block-segment lists from completed native
    /// contour generation. Sparse output locations outside native writes retain their
    /// previous values. Prepares device regions and waits for their producers on other
    /// streams. Rejects capture and completes native use of host and device buffers before
    /// returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native interpolation, or synchronization fail. Image dimensions
    /// must match and IDs must satisfy `first < last <= max_marker_label_id`. The host
    /// offset sentinel at `max_marker_label_id` must exist. Host/device geometry lengths
    /// must match and cover the native geometry-size query for that sentinel. Found-count
    /// and starting-offset lists must cover all labels; the host offsets also need the
    /// sentinel. The interpolated device geometry list must have exactly
    /// `total_image_pixel_contour_count` elements, and host/device block-segment lengths
    /// must match. Checked size conversions and native size queries can also fail.
    ///
    /// # Safety
    ///
    /// Every image, geometry list, count, offset, and range must correspond to the same
    /// completed native contour generation. Supply matching query-sized block-segment
    /// storage; equal host/device lengths alone do not establish adequate capacity.
    /// The host offset sentinel must be ready for immediate reading. Source and metadata
    /// storage passed through mutable native pointers must be exclusively available.
    /// Keep all native-writable regions disjoint from other participating allocations,
    /// initialized wherever the operation reads or preserves values, and valid through
    /// completion. Host allocations must satisfy native host-memory requirements.
    pub unsafe fn contours_image_marching_squares_interpolation_64f(
        stream_context: &StreamContext,
        contours_image_dev: &ImageView<'_, u8, C1>,
        contours_interpolated_image_dev: &mut ImageViewMut<'_, Point64f, C1>,
        contours_direction_image_dev: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_geometry_lists_dev: &mut DeviceMemory<ContourPixelGeometryInfo>,
        contours_pixel_geometry_lists_host: &mut [ContourPixelGeometryInfo],
        contours_interpolated_geometry_lists_dev: &mut DeviceMemory<Point64f>,
        contours_pixels_found_list_host: &mut [u32],
        contours_pixels_starting_offset_dev: &mut DeviceMemory<u32>,
        contours_pixels_starting_offset_host: &mut [u32],
        total_image_pixel_contour_count: u32,
        max_marker_label_id: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
        contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
        contours_block_segment_list_host: &mut [ContourBlockSegment],
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(contours_image_dev)?;
        stream_context.prepare_region(contours_interpolated_image_dev)?;
        stream_context.prepare_region(contours_direction_image_dev)?;
        stream_context.prepare_region(contours_pixel_geometry_lists_dev)?;
        stream_context.prepare_region(contours_interpolated_geometry_lists_dev)?;
        stream_context.prepare_region(contours_pixels_starting_offset_dev)?;
        stream_context.prepare_region(contours_block_segment_list_dev)?;
        // SAFETY: the caller upholds this function's safety contract.
        stream_context.sync_scope(|| unsafe {
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
        })
    }
}
