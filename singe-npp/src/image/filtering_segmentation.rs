use super::*;

pub(crate) fn segment_watershed_buffer_size_u8_c1(roi: Size) -> Result<usize> {
    let mut bytes = 0_u64;
    unsafe {
        try_ffi!(sys::nppiSegmentWatershedGetBufferSize_8u_C1R(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn segment_watershed_buffer_size_u16_c1(roi: Size) -> Result<usize> {
    let mut bytes = 0_u64;
    unsafe {
        try_ffi!(sys::nppiSegmentWatershedGetBufferSize_16u_C1R(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn segment_watershed_u8_c1_in_place(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, u8, C1>,
    marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
    norm: ImageNormalization,
    boundary_type: WatershedSegmentBoundaryType,
) -> Result<()> {
    if let Some(marker_labels) = marker_labels.as_ref() {
        validate_same_size(source_destination.size(), marker_labels.size())?;
    }
    let required_bytes = segment_watershed_buffer_size_u8_c1(source_destination.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    let (marker_labels_ptr, marker_labels_step) = match marker_labels {
        Some(marker_labels) => (marker_labels.as_mut_ptr().cast(), marker_labels.step()),
        None => (std::ptr::null_mut(), 0),
    };

    unsafe {
        try_ffi!(sys::nppiSegmentWatershed_8u_C1IR_Ctx(
            source_destination.as_mut_ptr().cast(),
            source_destination.step(),
            marker_labels_ptr,
            marker_labels_step,
            norm.into(),
            boundary_type.into(),
            source_destination.size().into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn segment_watershed_u16_c1_in_place(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, u16, C1>,
    marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
    norm: ImageNormalization,
    boundary_type: WatershedSegmentBoundaryType,
) -> Result<()> {
    if let Some(marker_labels) = marker_labels.as_ref() {
        validate_same_size(source_destination.size(), marker_labels.size())?;
    }
    let required_bytes = segment_watershed_buffer_size_u16_c1(source_destination.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    let (marker_labels_ptr, marker_labels_step) = match marker_labels {
        Some(marker_labels) => (marker_labels.as_mut_ptr().cast(), marker_labels.step()),
        None => (std::ptr::null_mut(), 0),
    };

    unsafe {
        try_ffi!(sys::nppiSegmentWatershed_16u_C1IR_Ctx(
            source_destination.as_mut_ptr().cast(),
            source_destination.step(),
            marker_labels_ptr,
            marker_labels_step,
            norm.into(),
            boundary_type.into(),
            source_destination.size().into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub trait SegmentWatershedC1InPlace: DataTypeLike + Sized {
    fn buffer_size(roi: Size) -> Result<usize>;

    fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()>;
}

impl SegmentWatershedC1InPlace for u8 {
    fn buffer_size(roi: Size) -> Result<usize> {
        segment_watershed_buffer_size_u8_c1(roi)
    }

    fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()> {
        segment_watershed_u8_c1_in_place(
            stream_context,
            source_destination,
            marker_labels,
            norm,
            boundary_type,
        )
    }
}

impl SegmentWatershedC1InPlace for u16 {
    fn buffer_size(roi: Size) -> Result<usize> {
        segment_watershed_buffer_size_u16_c1(roi)
    }

    fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()> {
        segment_watershed_u16_c1_in_place(
            stream_context,
            source_destination,
            marker_labels,
            norm,
            boundary_type,
        )
    }
}

pub fn segment_watershed_buffer_size_for<T: SegmentWatershedC1InPlace>(roi: Size) -> Result<usize> {
    T::buffer_size(roi)
}

pub fn segment_watershed_in_place<T: SegmentWatershedC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
    norm: ImageNormalization,
    boundary_type: WatershedSegmentBoundaryType,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source_destination,
        marker_labels,
        norm,
        boundary_type,
    )
}

pub fn compress_marker_labels_uf_u32_c1_in_place(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, u32, C1>,
    starting_number: i32,
) -> Result<u32> {
    validate_label_marker_step(source_destination)?;
    let required_bytes = compress_marker_labels_uf_buffer_size_u32_c1(starting_number)?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    let mut new_number = 0;
    unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(
            source_destination.as_mut_ptr().cast(),
            source_destination.step(),
            source_destination.size().into(),
            starting_number,
            &raw mut new_number,
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    u32::try_from(new_number).map_err(|_| Error::OutOfRange {
        name: "new number".into(),
    })
}

pub trait CompressMarkerLabelsUfC1InPlace: DataTypeLike + Sized {
    fn buffer_size(starting_number: i32) -> Result<usize>;
    fn info_list_size(max_marker_label_id: u32) -> Result<usize>;

    fn info_dispatch(
        stream_context: &StreamContext,
        compressed_marker_labels: &ImageView<'_, Self, C1>,
        max_marker_label_id: u32,
        marker_labels_info_list: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
        contours_image: &mut ImageViewMut<'_, u8, C1>,
        contours_direction_image: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
        contours_totals_info_host: &mut ContourTotalsInfo,
        contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
        contours_pixel_counts_list_host: &mut [u32],
        contours_pixel_starting_offset_dev: &mut DeviceMemory<u32>,
        contours_pixel_starting_offset_host: &mut [u32],
    ) -> Result<()>;

    fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        starting_number: i32,
    ) -> Result<u32>;

    fn batch_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()>;

    fn batch_advanced_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()>;
}

impl CompressMarkerLabelsUfC1InPlace for u32 {
    fn buffer_size(starting_number: i32) -> Result<usize> {
        compress_marker_labels_uf_buffer_size_u32_c1(starting_number)
    }

    fn info_list_size(max_marker_label_id: u32) -> Result<usize> {
        compressed_marker_labels_uf_info_list_size_u32_c1(max_marker_label_id)
    }

    fn info_dispatch(
        stream_context: &StreamContext,
        compressed_marker_labels: &ImageView<'_, Self, C1>,
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
        compressed_marker_labels_uf_info_u32_c1(
            stream_context,
            compressed_marker_labels,
            max_marker_label_id,
            marker_labels_info_list,
            contours_image,
            contours_direction_image,
            contours_totals_info_host,
            contours_pixel_counts_list_dev,
            contours_pixel_counts_list_host,
            contours_pixel_starting_offset_dev,
            contours_pixel_starting_offset_host,
        )
    }

    fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        starting_number: i32,
    ) -> Result<u32> {
        compress_marker_labels_uf_u32_c1_in_place(
            stream_context,
            source_destination,
            starting_number,
        )
    }

    fn batch_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()> {
        compress_marker_labels_uf_batch_u32_c1_in_place(
            stream_context,
            source_destinations,
            new_max_label_ids,
        )
    }

    fn batch_advanced_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()> {
        compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
            stream_context,
            source_destinations,
            new_max_label_ids,
        )
    }
}
