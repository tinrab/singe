use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{C1, ImageView, ImageViewMut},
    try_ffi,
    types::{
        CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelDirectionInfo,
        ContourPixelGeometryInfo, ContourTotalsInfo, DataTypeLike, ImageNormalization, Point32f,
        Point64f, Size,
    },
    utility::to_usize,
};

use super::filtering_validation::*;

pub(crate) fn label_markers_uf_buffer_size_u32_c1(roi: Size) -> Result<usize> {
    let mut bytes = 0;
    unsafe {
        try_ffi!(sys::nppiLabelMarkersUFGetBufferSize_32u_C1R(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn label_markers_uf_u8_to_u32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    let required_bytes = label_markers_uf_buffer_size_u32_c1(source.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiLabelMarkersUF_8u32u_C1R_Ctx(
            source.as_ptr().cast_mut().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            norm.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn label_markers_uf_u16_to_u32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageViewMut<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    let required_bytes = label_markers_uf_buffer_size_u32_c1(source.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiLabelMarkersUF_16u32u_C1R_Ctx(
            source.as_ptr().cast_mut().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            norm.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn label_markers_uf_u32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u32, C1>,
    destination: &mut ImageViewMut<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    let required_bytes = label_markers_uf_buffer_size_u32_c1(source.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiLabelMarkersUF_32u_C1R_Ctx(
            source.as_ptr().cast_mut().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            norm.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

macro_rules! impl_label_markers_uf_batch {
    ($name:ident, $source_ty:ty, $ffi_name:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $source_ty, C1>],
            destinations: &mut [ImageViewMut<'_, u32, C1>],
            norm: ImageNormalization,
        ) -> Result<()> {
            let (batch_size, roi, source_descriptors, destination_descriptors) =
                label_markers_uf_batch_descriptors(sources, destinations)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_ptr().cast_mut().cast(),
                    batch_size,
                    roi.into(),
                    norm.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_label_markers_uf_batch_advanced {
    ($name:ident, $source_ty:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $source_ty, C1>],
            destinations: &mut [ImageViewMut<'_, u32, C1>],
            norm: ImageNormalization,
        ) -> Result<()> {
            let (batch_size, max_roi, source_descriptors, destination_descriptors) =
                label_markers_uf_batch_advanced_descriptors(sources, destinations)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_ptr().cast_mut().cast(),
                    batch_size,
                    max_roi.into(),
                    norm.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_label_markers_uf_batch!(
    label_markers_uf_batch_u8_to_u32_c1,
    u8,
    nppiLabelMarkersUFBatch_8u32u_C1R_Ctx
);
impl_label_markers_uf_batch!(
    label_markers_uf_batch_u16_to_u32_c1,
    u16,
    nppiLabelMarkersUFBatch_16u32u_C1R_Ctx
);
impl_label_markers_uf_batch!(
    label_markers_uf_batch_u32_c1,
    u32,
    nppiLabelMarkersUFBatch_32u_C1R_Ctx
);

impl_label_markers_uf_batch_advanced!(
    label_markers_uf_batch_u8_to_u32_c1_advanced,
    u8,
    nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx
);
impl_label_markers_uf_batch_advanced!(
    label_markers_uf_batch_u16_to_u32_c1_advanced,
    u16,
    nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx
);
impl_label_markers_uf_batch_advanced!(
    label_markers_uf_batch_u32_c1_advanced,
    u32,
    nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx
);

pub trait LabelMarkersUfC1: DataTypeLike + Sized {
    fn buffer_size(roi: Size) -> Result<usize> {
        label_markers_uf_buffer_size_u32_c1(roi)
    }

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

impl LabelMarkersUfC1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_u8_to_u32_c1(stream_context, source, destination, norm)
    }
}

impl LabelMarkersUfC1 for u16 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_u16_to_u32_c1(stream_context, source, destination, norm)
    }
}

impl LabelMarkersUfC1 for u32 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_u32_c1(stream_context, source, destination, norm)
    }
}

pub fn label_markers_uf_buffer_size_for<T: LabelMarkersUfC1>(roi: Size) -> Result<usize> {
    T::buffer_size(roi)
}

pub fn label_markers_uf<T: LabelMarkersUfC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    T::dispatch(stream_context, source, destination, norm)
}

pub trait LabelMarkersUfToC1<Destination>: DataTypeLike + Sized
where
    Destination: DataTypeLike,
{
    fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Destination, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

impl<T> LabelMarkersUfToC1<u32> for T
where
    T: LabelMarkersUfC1,
{
    fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        T::dispatch(stream_context, source, destination, norm)
    }
}

pub fn label_markers_uf_to<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, Destination, C1>,
    norm: ImageNormalization,
) -> Result<()>
where
    T: LabelMarkersUfToC1<Destination>,
    Destination: DataTypeLike,
{
    T::dispatch_to(stream_context, source, destination, norm)
}

pub trait LabelMarkersUfBatchC1: DataTypeLike + Sized {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;
}

impl LabelMarkersUfBatchC1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_batch_u8_to_u32_c1(stream_context, sources, destinations, norm)
    }
}

impl LabelMarkersUfBatchC1 for u16 {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_batch_u16_to_u32_c1(stream_context, sources, destinations, norm)
    }
}

impl LabelMarkersUfBatchC1 for u32 {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_batch_u32_c1(stream_context, sources, destinations, norm)
    }
}

pub fn label_markers_uf_batch<T: LabelMarkersUfBatchC1>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageViewMut<'_, u32, C1>],
    norm: ImageNormalization,
) -> Result<()> {
    T::dispatch(stream_context, sources, destinations, norm)
}

pub trait LabelMarkersUfBatchAdvancedC1: DataTypeLike + Sized {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;
}

impl LabelMarkersUfBatchAdvancedC1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_batch_u8_to_u32_c1_advanced(stream_context, sources, destinations, norm)
    }
}

impl LabelMarkersUfBatchAdvancedC1 for u16 {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_batch_u16_to_u32_c1_advanced(stream_context, sources, destinations, norm)
    }
}

impl LabelMarkersUfBatchAdvancedC1 for u32 {
    fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        label_markers_uf_batch_u32_c1_advanced(stream_context, sources, destinations, norm)
    }
}

pub fn label_markers_uf_batch_advanced<T: LabelMarkersUfBatchAdvancedC1>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageViewMut<'_, u32, C1>],
    norm: ImageNormalization,
) -> Result<()> {
    T::dispatch(stream_context, sources, destinations, norm)
}

pub(crate) fn compress_marker_labels_uf_batch_u32_c1_in_place(
    stream_context: &StreamContext,
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    new_max_label_ids: &mut DeviceMemory<u32>,
) -> Result<()> {
    let (batch_size, roi, descriptors) =
        compress_marker_labels_uf_batch_descriptors(source_destinations)?;
    validate_batch_label_output(new_max_label_ids, batch_size as usize)?;
    let starting_number = roi_area(roi)?;
    let per_image_bytes = compress_marker_labels_uf_buffer_size_u32_c1(starting_number)?;
    let (_buffers, mut buffer_descriptors) =
        compress_marker_labels_uf_batch_buffers(batch_size as usize, per_image_bytes)?;
    let per_image_bytes = i32::try_from(per_image_bytes).map_err(|_| Error::OutOfRange {
        name: "buffer size".into(),
    })?;

    unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(
            descriptors.as_ptr().cast_mut().cast(),
            buffer_descriptors.as_mut_ptr().cast(),
            new_max_label_ids.as_mut_ptr().cast(),
            batch_size,
            roi.into(),
            per_image_bytes,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
    stream_context: &StreamContext,
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    new_max_label_ids: &mut DeviceMemory<u32>,
) -> Result<()> {
    let (batch_size, max_roi, descriptors) =
        compress_marker_labels_uf_batch_advanced_descriptors(source_destinations)?;
    validate_batch_label_output(new_max_label_ids, batch_size as usize)?;
    let (largest_per_image_bytes, _buffers, mut buffer_descriptors) =
        compress_marker_labels_uf_batch_advanced_buffers(source_destinations)?;
    let largest_per_image_bytes =
        i32::try_from(largest_per_image_bytes).map_err(|_| Error::OutOfRange {
            name: "buffer size".into(),
        })?;

    unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(
            descriptors.as_ptr().cast_mut().cast(),
            buffer_descriptors.as_mut_ptr().cast(),
            new_max_label_ids.as_mut_ptr().cast(),
            batch_size,
            max_roi.into(),
            largest_per_image_bytes,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn compress_marker_labels_uf_buffer_size_u32_c1(starting_number: i32) -> Result<usize> {
    let mut bytes = 0;
    unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            starting_number,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn compressed_marker_labels_uf_info_list_size_u32_c1(
    max_marker_label_id: u32,
) -> Result<usize> {
    let mut bytes = 0_u32;
    unsafe {
        try_ffi!(sys::nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(
            max_marker_label_id,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn compressed_marker_labels_uf_contours_block_segment_list_size_c1(
    contours_pixel_counts_list_host: &mut [u32],
    total_image_pixel_contour_count: u32,
    compressed_label_count: u32,
    first_contour_geometry_list_id: u32,
    last_contour_geometry_list_id: u32,
) -> Result<usize> {
    let mut bytes = 0_u32;
    unsafe {
        try_ffi!(
            sys::nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
                contours_pixel_counts_list_host.as_mut_ptr().cast(),
                total_image_pixel_contour_count,
                compressed_label_count,
                first_contour_geometry_list_id,
                last_contour_geometry_list_id,
                &raw mut bytes,
            )
        )?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn compressed_marker_labels_uf_geometry_lists_size_c1(
    max_contour_pixel_geometry_info_count: u32,
) -> Result<usize> {
    let mut bytes = 0_u32;
    unsafe {
        try_ffi!(sys::nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(
            max_contour_pixel_geometry_info_count,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn compressed_marker_labels_uf_info_u32_c1(
    stream_context: &StreamContext,
    compressed_marker_labels: &ImageView<'_, u32, C1>,
    max_marker_label_id: u32,
    marker_labels_info_list: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
    contours_image: &mut ImageViewMut<'_, u8, C1>,
    contours_direction_image: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
    contours_totals_info_host: &mut ContourTotalsInfo,
    contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
    contours_pixel_counts_list_host: &mut [u32],
    contours_pixel_starting_offset_dev: &mut DeviceMemory<u32>,
    contours_pixel_starting_offset_host: &mut [u32],
) -> Result<()> {
    validate_same_size(compressed_marker_labels.size(), contours_image.size())?;
    validate_same_size(
        compressed_marker_labels.size(),
        contours_direction_image.size(),
    )?;
    validate_u32_label_marker_step(
        compressed_marker_labels.size(),
        compressed_marker_labels.step(),
    )?;
    validate_marker_label_info_output(marker_labels_info_list, max_marker_label_id)?;
    validate_u32_list_output(
        contours_pixel_counts_list_dev,
        contours_pixel_counts_list_host,
        max_marker_label_id,
        "contours pixel counts list",
    )?;
    validate_u32_list_output(
        contours_pixel_starting_offset_dev,
        contours_pixel_starting_offset_host,
        max_marker_label_id,
        "contours pixel starting offset list",
    )?;

    unsafe {
        try_ffi!(sys::nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
            compressed_marker_labels.as_ptr().cast_mut().cast(),
            compressed_marker_labels.step(),
            compressed_marker_labels.size().into(),
            max_marker_label_id,
            marker_labels_info_list.as_mut_ptr().cast(),
            contours_image.as_mut_ptr().cast(),
            contours_image.step(),
            contours_direction_image.as_mut_ptr().cast(),
            contours_direction_image.step(),
            contours_totals_info_host as *mut _ as *mut _,
            contours_pixel_counts_list_dev.as_mut_ptr().cast(),
            contours_pixel_counts_list_host.as_mut_ptr().cast(),
            contours_pixel_starting_offset_dev.as_mut_ptr().cast(),
            contours_pixel_starting_offset_host.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compressed_marker_labels_uf_contours_generate_geometry_lists_c1(
    stream_context: &StreamContext,
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
    max_marker_label_id: u32,
    first_contour_geometry_list_id: u32,
    last_contour_geometry_list_id: u32,
    contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
    contours_block_segment_list_host: &mut [ContourBlockSegment],
    output_in_counterclockwise_order: bool,
) -> Result<()> {
    validate_marker_label_info_host_output(marker_labels_info_list_host, max_marker_label_id)?;
    validate_marker_label_info_output(marker_labels_info_list_dev, max_marker_label_id)?;
    validate_u32_list_dev_output(
        contours_pixel_counts_list_dev,
        max_marker_label_id,
        "contours pixel counts list",
    )?;
    validate_u32_list_output(
        contours_pixels_found_list_dev,
        contours_pixels_found_list_host,
        max_marker_label_id,
        "contours pixels found list",
    )?;
    validate_u32_list_output(
        contours_pixels_starting_offset_dev,
        contours_pixels_starting_offset_host,
        max_marker_label_id,
        "contours pixel starting offset list",
    )?;
    validate_geometry_list_id_range(
        first_contour_geometry_list_id,
        last_contour_geometry_list_id,
        max_marker_label_id,
    )?;
    validate_contours_geometry_image_host(
        contours_geometry_image_host,
        contours_direction_image_dev.size(),
        contours_geometry_image_step,
    )?;
    validate_contours_block_segment_output(
        contours_block_segment_list_dev,
        contours_block_segment_list_host,
        contours_pixels_found_list_host,
        total_image_pixel_contour_count,
        max_marker_label_id,
        first_contour_geometry_list_id,
        last_contour_geometry_list_id,
    )?;

    unsafe {
        try_ffi!(
            sys::nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
                marker_labels_info_list_dev.as_mut_ptr().cast(),
                marker_labels_info_list_host.as_mut_ptr().cast(),
                contours_direction_image_dev.as_mut_ptr().cast(),
                contours_direction_image_dev.step(),
                contours_pixel_geometry_lists_dev.as_mut_ptr().cast(),
                contours_pixel_geometry_lists_host.as_mut_ptr().cast(),
                contours_geometry_image_host.as_mut_ptr().cast(),
                contours_geometry_image_step,
                contours_pixel_counts_list_dev.as_mut_ptr().cast(),
                contours_pixels_found_list_dev.as_mut_ptr().cast(),
                contours_pixels_found_list_host.as_mut_ptr().cast(),
                contours_pixels_starting_offset_dev.as_mut_ptr().cast(),
                contours_pixels_starting_offset_host.as_mut_ptr().cast(),
                total_image_pixel_contour_count,
                max_marker_label_id,
                first_contour_geometry_list_id,
                last_contour_geometry_list_id,
                contours_block_segment_list_dev.as_mut_ptr().cast(),
                contours_block_segment_list_host.as_mut_ptr().cast(),
                u32::from(output_in_counterclockwise_order),
                contours_direction_image_dev.size().into(),
                stream_context.as_raw(),
            )
        )?;
    }
    Ok(())
}

macro_rules! impl_contours_image_marching_squares_interpolation {
    ($name:ident, $point_ty:ty, $ffi_name:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            stream_context: &StreamContext,
            contours_image_dev: &ImageView<'_, u8, C1>,
            contours_interpolated_image_dev: &mut ImageViewMut<'_, $point_ty, C1>,
            contours_direction_image_dev: &ImageView<'_, ContourPixelDirectionInfo, C1>,
            contours_pixel_geometry_lists_dev: &DeviceMemory<ContourPixelGeometryInfo>,
            contours_pixel_geometry_lists_host: &[ContourPixelGeometryInfo],
            contours_interpolated_geometry_lists_dev: &mut DeviceMemory<$point_ty>,
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
            validate_same_size(
                contours_image_dev.size(),
                contours_interpolated_image_dev.size(),
            )?;
            validate_same_size(
                contours_image_dev.size(),
                contours_direction_image_dev.size(),
            )?;
            validate_geometry_list_id_range(
                first_contour_geometry_list_id,
                last_contour_geometry_list_id,
                max_marker_label_id,
            )?;
            validate_geometry_list_buffers(
                contours_pixel_geometry_lists_dev,
                contours_pixel_geometry_lists_host,
            )?;
            validate_u32_list_host_output(
                contours_pixels_found_list_host,
                max_marker_label_id,
                "contours pixels found list",
            )?;
            validate_u32_list_device_and_host_output(
                contours_pixels_starting_offset_dev,
                contours_pixels_starting_offset_host,
                max_marker_label_id,
                "contours pixel starting offset list",
            )?;
            validate_typed_output_len(
                contours_interpolated_geometry_lists_dev.len(),
                total_image_pixel_contour_count,
                "contours interpolated geometry lists",
            )?;
            validate_block_segment_inputs(
                contours_block_segment_list_dev,
                contours_block_segment_list_host,
            )?;

            unsafe {
                try_ffi!(sys::$ffi_name(
                    contours_image_dev.as_ptr().cast_mut().cast(),
                    contours_image_dev.step(),
                    contours_interpolated_image_dev.as_mut_ptr().cast(),
                    contours_interpolated_image_dev.step(),
                    contours_direction_image_dev.as_ptr().cast_mut().cast(),
                    contours_direction_image_dev.step(),
                    contours_pixel_geometry_lists_dev.as_ptr().cast_mut().cast(),
                    contours_pixel_geometry_lists_host
                        .as_ptr()
                        .cast_mut()
                        .cast(),
                    contours_interpolated_geometry_lists_dev.as_mut_ptr().cast(),
                    contours_pixels_found_list_host.as_ptr().cast_mut().cast(),
                    contours_pixels_starting_offset_dev
                        .as_ptr()
                        .cast_mut()
                        .cast(),
                    contours_pixels_starting_offset_host
                        .as_ptr()
                        .cast_mut()
                        .cast(),
                    total_image_pixel_contour_count,
                    max_marker_label_id,
                    first_contour_geometry_list_id,
                    last_contour_geometry_list_id,
                    contours_block_segment_list_dev.as_ptr().cast_mut().cast(),
                    contours_block_segment_list_host.as_ptr().cast_mut().cast(),
                    contours_image_dev.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_contours_image_marching_squares_interpolation!(
    contours_image_marching_squares_interpolation_32f_c1,
    Point32f,
    nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx
);
impl_contours_image_marching_squares_interpolation!(
    contours_image_marching_squares_interpolation_64f_c1,
    Point64f,
    nppiContoursImageMarchingSquaresInterpolation_64f_C1R_Ctx
);
