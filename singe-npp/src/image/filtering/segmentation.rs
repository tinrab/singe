use std::ptr;

use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_labels::{
            compress_marker_labels_uf_batch_u32_c1_in_place,
            compress_marker_labels_uf_batch_u32_c1_in_place_advanced,
            compress_marker_labels_uf_buffer_size_u32_c1,
            compressed_marker_labels_uf_info_list_size_u32_c1,
            compressed_marker_labels_uf_info_u32_c1,
        },
        filtering_validation::{validate_label_marker_step, validate_same_size},
        view::{C1, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{
        CompressedMarkerLabelsInfo, ContourPixelDirectionInfo, ContourTotalsInfo, DataTypeLike,
        ImageNormalization, Size, WatershedSegmentBoundaryType,
    },
    utility::to_usize,
};

/// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
pub fn segment_watershed_buffer_size_u8_c1(roi: Size) -> Result<usize> {
    let mut bytes = 0_u64;
    // SAFETY: `nppiSegmentWatershedGetBufferSize_8u_C1R` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiSegmentWatershedGetBufferSize_8u_C1R(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
pub fn segment_watershed_buffer_size_u16_c1(roi: Size) -> Result<usize> {
    let mut bytes = 0_u64;
    // SAFETY: `nppiSegmentWatershedGetBufferSize_16u_C1R` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiSegmentWatershedGetBufferSize_16u_C1R(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Segments the image in place with the native watershed algorithm.
///
/// `marker_labels` optionally receives region labels; `None` omits that output.
/// `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity. `boundary_type` selects how native segment boundaries are rendered.
/// The provided implementations own device scratch, reject stream capture, and return after successful stream completion. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the segmentation parameters.
/// A supplied marker-label image must have the same size as `source_destination`.
/// Workspace queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// The in-place image must be initialized and ready for native execution. Keep its storage exclusively accessible during native use.
/// Any marker-label output must be exclusively writable and disjoint from the in-place image.
/// Retain allocations until this call completes its native work. Native layouts, strides, alignment, and allocation bounds must be valid.
pub unsafe fn segment_watershed_u8_c1_in_place(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, u8, C1>,
    marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
    norm: ImageNormalization,
    boundary_type: WatershedSegmentBoundaryType,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source_destination)?;
    if let Some(image) = marker_labels.as_ref() {
        stream_context.validate_region(*image)?;
    }
    stream_context.stream().ensure_not_capturing()?;

    if let Some(marker_labels) = marker_labels.as_ref() {
        validate_same_size(source_destination.size(), marker_labels.size())?;
    }
    let required_bytes = segment_watershed_buffer_size_u8_c1(source_destination.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
    let (marker_labels_ptr, marker_labels_step) = marker_labels
        .map_or((ptr::null_mut(), 0), |marker_labels| {
            (marker_labels.as_mut_ptr().cast(), marker_labels.step())
        });

    // SAFETY: `nppiSegmentWatershed_8u_C1IR_Ctx` receives live handles owned by the wrappers and valid output locations.
    stream_context.sync_scope(|| unsafe {
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
        ))
    })?;

    Ok(())
}

/// Segments the image in place with the native watershed algorithm.
///
/// `marker_labels` optionally receives region labels; `None` omits that output.
/// `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity. `boundary_type` selects how native segment boundaries are rendered.
/// The provided implementations own device scratch, reject stream capture, and return after successful stream completion. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the segmentation parameters.
/// A supplied marker-label image must have the same size as `source_destination`.
/// Workspace queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// The in-place image must be initialized and ready for native execution. Keep its storage exclusively accessible during native use.
/// Any marker-label output must be exclusively writable and disjoint from the in-place image.
/// Retain allocations until this call completes its native work. Native layouts, strides, alignment, and allocation bounds must be valid.
pub unsafe fn segment_watershed_u16_c1_in_place(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, u16, C1>,
    marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
    norm: ImageNormalization,
    boundary_type: WatershedSegmentBoundaryType,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source_destination)?;
    if let Some(image) = marker_labels.as_ref() {
        stream_context.validate_region(*image)?;
    }
    stream_context.stream().ensure_not_capturing()?;

    if let Some(marker_labels) = marker_labels.as_ref() {
        validate_same_size(source_destination.size(), marker_labels.size())?;
    }
    let required_bytes = segment_watershed_buffer_size_u16_c1(source_destination.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
    let (marker_labels_ptr, marker_labels_step) = marker_labels
        .map_or((ptr::null_mut(), 0), |marker_labels| {
            (marker_labels.as_mut_ptr().cast(), marker_labels.step())
        });

    // SAFETY: `nppiSegmentWatershed_16u_C1IR_Ctx` receives live handles owned by the wrappers and valid output locations.
    stream_context.sync_scope(|| unsafe {
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
        ))
    })?;

    Ok(())
}

/// Dispatches native watershed segmentation.
///
/// # Safety
///
/// Implementations must preserve native element representations, output semantics, and the documented execution contract.
/// Scratch and list-size queries must not underestimate native capacities. Preserve UF allocation identity and queued lifetimes where required.
pub unsafe trait SegmentWatershedC1InPlace: DataTypeLike + Sized {
    /// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
    fn buffer_size(roi: Size) -> Result<usize>;

    /// Segments the image in place with the native watershed algorithm.
    ///
    /// `marker_labels` optionally receives region labels; `None` omits that output.
    /// `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity. `boundary_type` selects how native segment boundaries are rendered.
    /// The provided implementations own device scratch, reject stream capture, and return after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the segmentation parameters.
    /// A supplied marker-label image must have the same size as `source_destination`.
    /// Workspace queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// The in-place image must be initialized and ready for native execution. Keep its storage exclusively accessible during native use.
    /// Any marker-label output must be exclusively writable and disjoint from the in-place image.
    /// Retain allocations until this call completes its native work. Native layouts, strides, alignment, and allocation bounds must be valid.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()>;
}

unsafe impl SegmentWatershedC1InPlace for u8 {
    fn buffer_size(roi: Size) -> Result<usize> {
        segment_watershed_buffer_size_u8_c1(roi)
    }

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            segment_watershed_u8_c1_in_place(
                stream_context,
                source_destination,
                marker_labels,
                norm,
                boundary_type,
            )
        }
    }
}

unsafe impl SegmentWatershedC1InPlace for u16 {
    fn buffer_size(roi: Size) -> Result<usize> {
        segment_watershed_buffer_size_u16_c1(roi)
    }

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            segment_watershed_u16_c1_in_place(
                stream_context,
                source_destination,
                marker_labels,
                norm,
                boundary_type,
            )
        }
    }
}

/// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
pub fn segment_watershed_buffer_size_for<T: SegmentWatershedC1InPlace>(roi: Size) -> Result<usize> {
    T::buffer_size(roi)
}

/// Segments the image in place with the native watershed algorithm.
///
/// `marker_labels` optionally receives region labels; `None` omits that output.
/// `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity. `boundary_type` selects how native segment boundaries are rendered.
/// The provided implementations own device scratch, reject stream capture, and return after successful stream completion. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the segmentation parameters.
/// A supplied marker-label image must have the same size as `source_destination`.
/// Workspace queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// The in-place image must be initialized and ready for native execution. Keep its storage exclusively accessible during native use.
/// Any marker-label output must be exclusively writable and disjoint from the in-place image.
/// Retain allocations until this call completes its native work. Native layouts, strides, alignment, and allocation bounds must be valid.
pub unsafe fn segment_watershed_in_place<T: SegmentWatershedC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
    norm: ImageNormalization,
    boundary_type: WatershedSegmentBoundaryType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source_destination,
            marker_labels,
            norm,
            boundary_type,
        )
    }
}

/// Renumbers UF marker labels in place to eliminate gaps in label IDs.
///
/// Returns the native compression result as `u32` after successful stream completion.
/// The provided implementation owns temporary device scratch and rejects stream capture.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, or its byte stride differs from its width times `size_of::<u32>()`.
/// Workspace queries, device allocation, capture checks, native compression, stream synchronization, or conversion of the native result to `u32` can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// The image must contain completed output from the matching UF labeling call, with its original base pointer, CUDA context, and ROI. Use a linear CUDA allocation rather than a pitched allocation.
/// `starting_number` must equal that original ROI's width times height; this wrapper does not validate that relationship.
/// Keep the image initialized, ready, and exclusively accessible until native work completes. Native layout, allocation bounds, and label contents must be valid.
pub unsafe fn compress_marker_labels_uf_u32_c1_in_place(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, u32, C1>,
    starting_number: i32,
) -> Result<u32> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source_destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_label_marker_step(source_destination)?;
    let required_bytes = compress_marker_labels_uf_buffer_size_u32_c1(starting_number)?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
    let mut new_number = 0;

    // SAFETY: `nppiCompressMarkerLabelsUF_32u_C1IR_Ctx` receives live handles owned by the wrappers and valid output locations.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(
            source_destination.as_mut_ptr().cast(),
            source_destination.step(),
            source_destination.size().into(),
            starting_number,
            &raw mut new_number,
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    u32::try_from(new_number).map_err(|_| Error::out_of_range("new number"))
}

/// Dispatches UF label compression and corresponding information queries.
///
/// # Safety
///
/// Implementations must preserve native element representations, output semantics, and the documented execution contract.
/// Scratch and list-size queries must not underestimate native capacities. Preserve UF allocation identity and queued lifetimes where required.
pub unsafe trait CompressMarkerLabelsUfC1InPlace: DataTypeLike + Sized {
    /// Returns the scratch capacity in bytes for compressing UF marker labels.
    ///
    /// `starting_number` is the ROI width times height used during marker generation.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    fn buffer_size(starting_number: i32) -> Result<usize>;

    /// Returns the marker-info list capacity in bytes for `max_marker_label_id`.
    ///
    /// Use the maximum label ID returned by the corresponding UF label-compression call.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    fn info_list_size(max_marker_label_id: u32) -> Result<usize>;

    /// Extracts compressed-label statistics, contour images, directions, counts, and starting offsets.
    ///
    /// All host outputs correspond to the same UF compressed-label image and `max_marker_label_id`.
    /// Submits native work without an explicit wrapper synchronization.
    /// Host and device buffers remain borrowed by native work until completion; returning successfully does not release those lifetime obligations.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Contour and direction images must match the label-image size. The label-image stride must equal its width times `size_of::<u32>()`.
    /// The marker-info allocation must cover the queried byte count. Both count lists require at least `max_marker_label_id` entries; both offset lists require at least `max_marker_label_id + 1`, with checked sentinel arithmetic.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// The label image must be a linear CUDA allocation produced by the UF generation and compression sequence, using its original base pointer and ROI. NPP receives a mutable label-image pointer; provide exclusive access during native use.
    /// Keep host outputs and device buffers valid and exclusively accessible through queued completion, including submission errors and every captured replay. Host allocations must meet native host-memory requirements.
    unsafe fn info_dispatch(
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

    /// Renumbers UF marker labels in place to eliminate gaps in label IDs.
    ///
    /// Returns the native compression result as `u32` after successful stream completion.
    /// The provided implementation owns temporary device scratch and rejects stream capture.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, or its byte stride differs from its width times `size_of::<u32>()`.
    /// Workspace queries, device allocation, capture checks, native compression, stream synchronization, or conversion of the native result to `u32` can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// The image must contain completed output from the matching UF labeling call, with its original base pointer, CUDA context, and ROI. Use a linear CUDA allocation rather than a pitched allocation.
    /// `starting_number` must equal that original ROI's width times height; this wrapper does not validate that relationship.
    /// Keep the image initialized, ready, and exclusively accessible until native work completes. Native layout, allocation bounds, and label contents must be valid.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        starting_number: i32,
    ) -> Result<u32>;

    /// Renumbers UF marker IDs in place to remove numbering gaps.
    ///
    /// The device `new_max_label_ids` output receives one new maximum label ID per image.
    /// The provided implementations reject stream capture and return after successful stream completion.
    /// Temporary device scratch or descriptor allocations remain alive through native use.
    /// All images use one common size and per-image scratch capacity.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// The batch must contain at least two images and fit in `i32`. Each image stride must equal its width times `size_of::<u32>()`.
    /// `new_max_label_ids` must contain exactly one element per image. ROI-area arithmetic, scratch queries and allocations, descriptor uploads, capture checks, and synchronization can also fail.
    /// All image sizes must match.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// Each image must contain output from a matching UF label-generation call, with the same base pointer and ROI. Use linear CUDA allocations rather than pitched allocations.
    /// Retain images and the count output until native work completes.
    unsafe fn batch_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()>;

    /// Renumbers UF marker IDs in place to remove numbering gaps.
    ///
    /// The device `new_max_label_ids` output receives one new maximum label ID per image.
    /// The provided implementations reject stream capture and return after successful stream completion.
    /// Temporary device scratch or descriptor allocations remain alive through native use.
    /// Uses each image's own dimensions and scratch size. The largest width, height, and per-image scratch size provide native batch bounds.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// The batch must contain at least two images and fit in `i32`. Each image stride must equal its width times `size_of::<u32>()`.
    /// `new_max_label_ids` must contain exactly one element per image. ROI-area arithmetic, scratch queries and allocations, descriptor uploads, capture checks, and synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// Each image must contain output from a matching UF label-generation call, with the same base pointer and ROI. Use linear CUDA allocations rather than pitched allocations.
    /// Retain images and the count output until native work completes.
    unsafe fn batch_advanced_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()>;
}

unsafe impl CompressMarkerLabelsUfC1InPlace for u32 {
    fn buffer_size(starting_number: i32) -> Result<usize> {
        compress_marker_labels_uf_buffer_size_u32_c1(starting_number)
    }

    fn info_list_size(max_marker_label_id: u32) -> Result<usize> {
        compressed_marker_labels_uf_info_list_size_u32_c1(max_marker_label_id)
    }

    unsafe fn info_dispatch(
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
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
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
    }

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        starting_number: i32,
    ) -> Result<u32> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            compress_marker_labels_uf_u32_c1_in_place(
                stream_context,
                source_destination,
                starting_number,
            )
        }
    }

    unsafe fn batch_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            compress_marker_labels_uf_batch_u32_c1_in_place(
                stream_context,
                source_destinations,
                new_max_label_ids,
            )
        }
    }

    unsafe fn batch_advanced_dispatch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, Self, C1>],
        new_max_label_ids: &mut DeviceMemory<u32>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
                stream_context,
                source_destinations,
                new_max_label_ids,
            )
        }
    }
}
