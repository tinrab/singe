use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceOutput, DeviceOutputView},
};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{
            validate_border_roi, validate_canny_thresholds,
            validate_harris_corners_border_parameters, validate_hough_line_destination_roi,
            validate_hough_line_io, validate_hough_line_parameters, validate_positive_size,
        },
        view::{C1, C3, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{
        BorderType, DataTypeLike, DifferentialKernel, ImageNormalization, MaskSize, Point,
        PointPolar, Rectangle, Size,
    },
    utility::to_usize,
};

/// Returns the scratch capacity in bytes for Canny edge detection.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
pub fn filter_canny_border_buffer_size(roi: Size) -> Result<usize> {
    let mut bytes = 0_i32;
    // SAFETY: `nppiFilterCannyBorderGetBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiFilterCannyBorderGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
///
/// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
/// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_canny_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // NPP border offsets describe the position already addressed by pSrc within the full image.
    let source_pointer = source
        .window(Rectangle {
            x: source_offset.x,
            y: source_offset.y,
            width: 1,
            height: 1,
        })?
        .view()
        .as_ptr();

    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_canny_thresholds(low_threshold, high_threshold)?;
    let required_bytes = filter_canny_border_buffer_size(destination.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiFilterCannyBorder_8u_C1R_Ctx` receives live handles owned by the wrappers and valid output locations.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiFilterCannyBorder_8u_C1R_Ctx(
            source_pointer.cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            filter_type.into(),
            mask_size.into(),
            low_threshold,
            high_threshold,
            norm.into(),
            border_type.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
///
/// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
/// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_canny_border_u8_c3_to_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // NPP border offsets describe the position already addressed by pSrc within the full image.
    let source_pointer = source
        .window(Rectangle {
            x: source_offset.x,
            y: source_offset.y,
            width: 1,
            height: 1,
        })?
        .view()
        .as_ptr();

    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_canny_thresholds(low_threshold, high_threshold)?;
    let required_bytes = filter_canny_border_buffer_size(destination.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiFilterCannyBorder_8u_C3C1R_Ctx` receives live handles owned by the wrappers and valid output locations.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiFilterCannyBorder_8u_C3C1R_Ctx(
            source_pointer.cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            filter_type.into(),
            mask_size.into(),
            low_threshold,
            high_threshold,
            norm.into(),
            border_type.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
///
/// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
/// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_canny_border_c3_to_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        filter_canny_border_u8_c3_to_c1(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

/// Returns the scratch capacity in bytes for Harris corner responses.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
pub fn filter_harris_corners_border_buffer_size(roi: Size) -> Result<usize> {
    let mut bytes = 0_i32;
    // SAFETY: `nppiFilterHarrisCornersBorderGetBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiFilterHarrisCornersBorderGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Computes Harris corner responses into the destination image.
///
/// `filter_type` selects the Sobel or Scharr differential filter. `mask_size` selects the differential mask, and `average_window_size` selects the averaging window.
/// `k` is the Harris constant, commonly 0.04, and `scale` scales the output response. The wrapper forwards these two values without validating them.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// `mask_size` and `average_window_size` must each be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`. Only `BorderType::Replicate` is accepted.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_harris_corners_border_u8_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, f32, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    average_window_size: MaskSize,
    k: f32,
    scale: f32,
    border_type: BorderType,
) -> Result<()> {
    // NPP border offsets describe the position already addressed by pSrc within the full image.
    let source_pointer = source
        .window(Rectangle {
            x: source_offset.x,
            y: source_offset.y,
            width: 1,
            height: 1,
        })?
        .view()
        .as_ptr();

    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_harris_corners_border_parameters(mask_size, average_window_size, border_type)?;
    let required_bytes = filter_harris_corners_border_buffer_size(destination.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx` receives live handles owned by the wrappers and valid output locations.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(
            source_pointer.cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            filter_type.into(),
            mask_size.into(),
            average_window_size.into(),
            k,
            scale,
            border_type.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Returns the scratch capacity in bytes for Hough line detection.
/// `delta` supplies the native radial and angular discretization increments; `max_line_count` is the output capacity.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
/// ROI dimensions and `max_line_count` must be positive. `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
pub fn filter_hough_line_buffer_size(
    roi: Size,
    delta: PointPolar,
    max_line_count: i32,
) -> Result<usize> {
    validate_positive_size(roi)?;
    validate_hough_line_parameters(delta, max_line_count)?;
    let mut bytes = 0_i32;
    // SAFETY: `nppiFilterHoughLineGetBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiFilterHoughLineGetBufferSize(
            roi.into(),
            delta.into(),
            max_line_count,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Detects lines and writes their polar parameters to device memory.
///
/// `delta` supplies the native radial and angular discretization increments.
/// `threshold` is the minimum number of points required to accept a line.
/// `lines.len()` limits the number of returned lines; the single device element in `line_count` receives the count. Only that many entries of `lines` contain results.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
///
/// # Errors
///
/// Source dimensions and `threshold` must be positive. `lines` must have a nonzero capacity representable as `i32`, and `line_count` must contain exactly one element.
/// `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
/// Both `lines` and `line_count` must be device-accessible allocations in the selected CUDA context.
pub unsafe fn filter_hough_line_u8_to_polar_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
    line_count: &mut DeviceOutputView<'_, i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(line_count)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_hough_line_io(
        stream_context,
        source.size(),
        delta,
        threshold,
        lines,
        line_count,
    )?;
    let max_line_count =
        i32::try_from(lines.len()).map_err(|_| Error::out_of_range("max line count"))?;
    let required_bytes = filter_hough_line_buffer_size(source.size(), delta, max_line_count)?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiFilterHoughLine_8u32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiFilterHoughLine_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            delta.into(),
            threshold,
            lines.as_mut_ptr().cast(),
            max_line_count,
            line_count.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Detects lines and writes their polar parameters to device memory.
///
/// `delta` supplies the native radial and angular discretization increments.
/// `threshold` is the minimum number of points required to accept a line.
/// `lines.len()` limits the number of returned lines; the single device element in `line_count` receives the count. Only that many entries of `lines` contain results.
/// `destination_roi` contains the lower and upper polar bounds of the search region.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
///
/// # Errors
///
/// Source dimensions and `threshold` must be positive. `lines` must have a nonzero capacity representable as `i32`, and `line_count` must contain exactly one element.
/// `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
/// Each component of the first polar bound must not exceed the corresponding component of the second bound.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
/// Both `lines` and `line_count` must be device-accessible allocations in the selected CUDA context.
pub unsafe fn filter_hough_line_region_u8_to_polar_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
    destination_roi: [PointPolar; 2],
    line_count: &mut DeviceOutputView<'_, i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(line_count)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_hough_line_io(
        stream_context,
        source.size(),
        delta,
        threshold,
        lines,
        line_count,
    )?;
    validate_hough_line_destination_roi(destination_roi)?;
    let max_line_count =
        i32::try_from(lines.len()).map_err(|_| Error::out_of_range("max line count"))?;
    let required_bytes = filter_hough_line_buffer_size(source.size(), delta, max_line_count)?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
    let mut destination_roi = destination_roi;

    // SAFETY: `nppiFilterHoughLineRegion_8u32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiFilterHoughLineRegion_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            delta.into(),
            threshold,
            lines.as_mut_ptr().cast(),
            destination_roi.as_mut_ptr().cast(),
            max_line_count,
            line_count.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Dispatches Canny edge detection for a supported source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, output extents, and detection semantics.
/// Scratch-size queries must not underestimate native workspace requirements.
/// Uphold the documented raw execution contract and retain temporary resources until their native use completes.
pub unsafe trait FilterCannyBorderC1: DataTypeLike + Sized {
    /// Returns the scratch capacity in bytes for Canny edge detection.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
    fn buffer_size(roi: Size) -> Result<usize>;

    /// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
    ///
    /// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
    /// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
    /// `source_offset` locates the destination-sized processing ROI within the full source image.
    /// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
    /// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
    /// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
    /// Retain allocations until this call completes its native work.
    /// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

unsafe impl FilterCannyBorderC1 for u8 {
    fn buffer_size(roi: Size) -> Result<usize> {
        filter_canny_border_buffer_size(roi)
    }

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            filter_canny_border_u8_c1(
                stream_context,
                source,
                source_offset,
                destination,
                filter_type,
                mask_size,
                low_threshold,
                high_threshold,
                norm,
                border_type,
            )
        }
    }
}

/// Returns the scratch capacity in bytes for Canny edge detection.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
pub fn filter_canny_border_typed_buffer_size<T: FilterCannyBorderC1>(roi: Size) -> Result<usize> {
    T::buffer_size(roi)
}

/// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
///
/// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
/// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_canny_border<T: FilterCannyBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

/// Dispatches Canny edge detection for a supported source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, output extents, and detection semantics.
/// Scratch-size queries must not underestimate native workspace requirements.
/// Uphold the documented raw execution contract and retain temporary resources until their native use completes.
pub unsafe trait FilterCannyBorderC3ToC1: DataTypeLike + Sized {
    /// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
    ///
    /// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
    /// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
    /// `source_offset` locates the destination-sized processing ROI within the full source image.
    /// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
    /// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
    /// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
    /// Retain allocations until this call completes its native work.
    /// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

unsafe impl FilterCannyBorderC3ToC1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            filter_canny_border_u8_c3_to_c1(
                stream_context,
                source,
                source_offset,
                destination,
                filter_type,
                mask_size,
                low_threshold,
                high_threshold,
                norm,
                border_type,
            )
        }
    }
}

/// Detects Canny edges and writes a black (`0x00`) and white (`0xff`) destination mask.
///
/// `filter_type` selects the Sobel or Scharr differential filter, and `mask_size` selects its native mask.
/// `low_threshold` and `high_threshold` are the hysteresis thresholds. `norm` selects the gradient-distance method.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Both thresholds must be nonnegative, and `low_threshold` must not exceed `high_threshold`.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_canny_border_to_mask<T: FilterCannyBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

/// Dispatches Harris corner responses for a supported source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, output extents, and detection semantics.
/// Scratch-size queries must not underestimate native workspace requirements.
/// Uphold the documented raw execution contract and retain temporary resources until their native use completes.
pub unsafe trait FilterHarrisCornersBorderToF32C1: DataTypeLike + Sized {
    /// Returns the scratch capacity in bytes for Harris corner responses.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
    fn buffer_size(roi: Size) -> Result<usize>;

    /// Computes Harris corner responses into the destination image.
    ///
    /// `filter_type` selects the Sobel or Scharr differential filter. `mask_size` selects the differential mask, and `average_window_size` selects the averaging window.
    /// `k` is the Harris constant, commonly 0.04, and `scale` scales the output response. The wrapper forwards these two values without validating them.
    /// `source_offset` locates the destination-sized processing ROI within the full source image.
    /// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// `mask_size` and `average_window_size` must each be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`. Only `BorderType::Replicate` is accepted.
    /// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
    /// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
    /// Retain allocations until this call completes its native work.
    /// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

unsafe impl FilterHarrisCornersBorderToF32C1 for u8 {
    fn buffer_size(roi: Size) -> Result<usize> {
        filter_harris_corners_border_buffer_size(roi)
    }

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            filter_harris_corners_border_u8_to_f32_c1(
                stream_context,
                source,
                source_offset,
                destination,
                filter_type,
                mask_size,
                average_window_size,
                k,
                scale,
                border_type,
            )
        }
    }
}

/// Returns the scratch capacity in bytes for Harris corner responses.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
pub fn filter_harris_corners_border_to_f32_buffer_size<T: FilterHarrisCornersBorderToF32C1>(
    roi: Size,
) -> Result<usize> {
    T::buffer_size(roi)
}

/// Computes Harris corner responses into the destination image.
///
/// `filter_type` selects the Sobel or Scharr differential filter. `mask_size` selects the differential mask, and `average_window_size` selects the averaging window.
/// `k` is the Harris constant, commonly 0.04, and `scale` scales the output response. The wrapper forwards these two values without validating them.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// `mask_size` and `average_window_size` must each be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`. Only `BorderType::Replicate` is accepted.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_harris_corners_border_to_f32<T: FilterHarrisCornersBorderToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, f32, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    average_window_size: MaskSize,
    k: f32,
    scale: f32,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            average_window_size,
            k,
            scale,
            border_type,
        )
    }
}

/// Dispatches Harris corner responses for a supported source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, output extents, and detection semantics.
/// Scratch-size queries must not underestimate native workspace requirements.
/// Uphold the documented raw execution contract and retain temporary resources until their native use completes.
pub unsafe trait FilterHarrisCornersBorderToC1<Destination>: DataTypeLike + Sized
where
    Destination: DataTypeLike,
{
    /// Returns the scratch capacity in bytes for Harris corner responses.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
    fn buffer_size(roi: Size) -> Result<usize>;

    /// Computes Harris corner responses into the destination image.
    ///
    /// `filter_type` selects the Sobel or Scharr differential filter. `mask_size` selects the differential mask, and `average_window_size` selects the averaging window.
    /// `k` is the Harris constant, commonly 0.04, and `scale` scales the output response. The wrapper forwards these two values without validating them.
    /// `source_offset` locates the destination-sized processing ROI within the full source image.
    /// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// `mask_size` and `average_window_size` must each be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`. Only `BorderType::Replicate` is accepted.
    /// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
    /// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
    /// Retain allocations until this call completes its native work.
    /// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Destination, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

unsafe impl<T> FilterHarrisCornersBorderToC1<f32> for T
where
    T: FilterHarrisCornersBorderToF32C1,
{
    fn buffer_size(roi: Size) -> Result<usize> {
        T::buffer_size(roi)
    }

    unsafe fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::dispatch(
                stream_context,
                source,
                source_offset,
                destination,
                filter_type,
                mask_size,
                average_window_size,
                k,
                scale,
                border_type,
            )
        }
    }
}

/// Returns the scratch capacity in bytes for Harris corner responses.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
pub fn filter_harris_corners_border_typed_buffer_size<T, Destination>(roi: Size) -> Result<usize>
where
    T: FilterHarrisCornersBorderToC1<Destination>,
    Destination: DataTypeLike,
{
    T::buffer_size(roi)
}

/// Computes Harris corner responses into the destination image.
///
/// `filter_type` selects the Sobel or Scharr differential filter. `mask_size` selects the differential mask, and `average_window_size` selects the averaging window.
/// `k` is the Harris constant, commonly 0.04, and `scale` scales the output response. The wrapper forwards these two values without validating them.
/// `source_offset` locates the destination-sized processing ROI within the full source image.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// `mask_size` and `average_window_size` must each be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`. Only `BorderType::Replicate` is accepted.
/// The source offset must address a source pixel, ROI endpoint calculations must not overflow, and the destination-sized ROI must fit within the full source image.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn filter_harris_corners_border<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, Destination, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    average_window_size: MaskSize,
    k: f32,
    scale: f32,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterHarrisCornersBorderToC1<Destination>,
    Destination: DataTypeLike,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch_to(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            average_window_size,
            k,
            scale,
            border_type,
        )
    }
}

/// Dispatches Hough line detection for a supported source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, output extents, and detection semantics.
/// Scratch-size queries must not underestimate native workspace requirements.
/// Uphold the documented raw execution contract and retain temporary resources until their native use completes.
pub unsafe trait FilterHoughLineToPolarC1: DataTypeLike + Sized {
    /// Detects lines and writes their polar parameters to device memory.
    ///
    /// `delta` supplies the native radial and angular discretization increments.
    /// `threshold` is the minimum number of points required to accept a line.
    /// `lines.len()` limits the number of returned lines; the single device element in `line_count` receives the count. Only that many entries of `lines` contain results.
    /// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
    ///
    /// # Errors
    ///
    /// Source dimensions and `threshold` must be positive. `lines` must have a nonzero capacity representable as `i32`, and `line_count` must contain exactly one element.
    /// `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
    /// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
    /// Retain allocations until this call completes its native work.
    /// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
    /// Both `lines` and `line_count` must be device-accessible allocations in the selected CUDA context.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
        line_count: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    /// Detects lines and writes their polar parameters to device memory.
    ///
    /// `delta` supplies the native radial and angular discretization increments.
    /// `threshold` is the minimum number of points required to accept a line.
    /// `lines.len()` limits the number of returned lines; the single device element in `line_count` receives the count. Only that many entries of `lines` contain results.
    /// `destination_roi` contains the lower and upper polar bounds of the search region.
    /// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
    ///
    /// # Errors
    ///
    /// Source dimensions and `threshold` must be positive. `lines` must have a nonzero capacity representable as `i32`, and `line_count` must contain exactly one element.
    /// `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
    /// Each component of the first polar bound must not exceed the corresponding component of the second bound.
    /// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
    /// Retain allocations until this call completes its native work.
    /// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
    /// Both `lines` and `line_count` must be device-accessible allocations in the selected CUDA context.
    unsafe fn region_dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
        destination_roi: [PointPolar; 2],
        line_count: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()>;
}

unsafe impl FilterHoughLineToPolarC1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
        line_count: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            filter_hough_line_u8_to_polar_c1(
                stream_context,
                source,
                delta,
                threshold,
                lines,
                line_count,
            )
        }
    }

    unsafe fn region_dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
        destination_roi: [PointPolar; 2],
        line_count: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            filter_hough_line_region_u8_to_polar_c1(
                stream_context,
                source,
                delta,
                threshold,
                lines,
                destination_roi,
                line_count,
            )
        }
    }
}

/// Detects lines and writes their polar parameters to device memory.
///
/// `delta` supplies the native radial and angular discretization increments.
/// `threshold` is the minimum number of points required to accept a line.
/// `lines.len()` limits the number of returned lines; the single device element in `line_count` receives the count. Only that many entries of `lines` contain results.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
///
/// # Errors
///
/// Source dimensions and `threshold` must be positive. `lines` must have a nonzero capacity representable as `i32`, and `line_count` must contain exactly one element.
/// `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
/// Both `lines` and `line_count` must be device-accessible allocations in the selected CUDA context.
pub unsafe fn filter_hough_line_to_polar<T: FilterHoughLineToPolarC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
    line_count: &mut DeviceOutputView<'_, i32>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, source, delta, threshold, lines, line_count) }
}

/// Detects lines and writes their polar parameters to device memory.
///
/// `delta` supplies the native radial and angular discretization increments.
/// `threshold` is the minimum number of points required to accept a line.
/// `lines.len()` limits the number of returned lines; the single device element in `line_count` receives the count. Only that many entries of `lines` contain results.
/// `destination_roi` contains the lower and upper polar bounds of the search region.
/// The provided implementations allocate temporary device scratch, reject stream capture, and return after successful stream completion.
///
/// # Errors
///
/// Source dimensions and `threshold` must be positive. `lines` must have a nonzero capacity representable as `i32`, and `line_count` must contain exactly one element.
/// `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
/// Each component of the first polar bound must not exceed the corresponding component of the second bound.
/// CUDA context binding, region-context validation, scratch-size queries, device allocation, NPP execution, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep the source read-only and all output storage exclusively writable and disjoint from the source and other outputs during native use.
/// Retain allocations until this call completes its native work.
/// Native pixel layouts, byte strides, alignment, and allocation bounds must be valid.
/// Both `lines` and `line_count` must be device-accessible allocations in the selected CUDA context.
pub unsafe fn filter_hough_line_region_to_polar<T: FilterHoughLineToPolarC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut (impl DeviceOutput<PointPolar> + ?Sized),
    destination_roi: [PointPolar; 2],
    line_count: &mut DeviceOutputView<'_, i32>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::region_dispatch(
            stream_context,
            source,
            delta,
            threshold,
            lines,
            destination_roi,
            line_count,
        )
    }
}
