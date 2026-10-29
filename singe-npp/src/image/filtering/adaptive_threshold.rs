use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_positive_size},
        view::{C1, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Rectangle, Size},
};

/// Thresholds each source pixel against its local box mean minus `delta`.
///
/// Writes `value_greater_than` when the source pixel is greater than that adjusted mean;
/// otherwise writes `value_less_or_equal`. NPP requires a square mask with odd side lengths
/// and supports only [`BorderType::Replicate`]. The wrapper checks positive mask dimensions
/// and forwards the remaining native restrictions and `delta` without validating them.
/// The destination-sized ROI starts at `source_offset` within the full source image.
/// Queues work on the selected stream without synchronizing. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if the source offset is invalid, ROI endpoints overflow or exceed the
/// source image, mask dimensions are nonpositive, CUDA context binding or region-context
/// validation fails, or NPP rejects the threshold parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Source pixels must be
/// initialized and ready for the selected stream. Keep the source read-only and destination
/// exclusively writable and disjoint during native use. Retain both allocations and the
/// stream context through queued completion, including submission errors and captured
/// replay. Uphold native layout, mask, sampling, byte-stride, alignment, and bounds requirements.
pub unsafe fn filter_threshold_adaptive_box_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, u8, C1>,
    mask_size: Size,
    delta: f32,
    value_greater_than: u8,
    value_less_or_equal: u8,
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

    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_positive_size(mask_size)?;
    // SAFETY: `nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(
            source_pointer.cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            mask_size.into(),
            delta,
            value_greater_than,
            value_less_or_equal,
            border_type.into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Dispatches adaptive box thresholding for a native single-channel element type.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and mean-minus-delta threshold
/// semantics. Uphold the method's geometry validation, memory-access rules, and queued lifetimes.
pub unsafe trait FilterThresholdAdaptiveBoxBorderC1: DataTypeLike {
    /// Thresholds each source pixel against its local box mean minus `delta`.
    ///
    /// Writes `value_greater_than` when the source pixel is greater than that adjusted mean;
    /// otherwise writes `value_less_or_equal`. NPP requires a square mask with odd side lengths
    /// and supports only [`BorderType::Replicate`]. The wrapper checks positive mask dimensions
    /// and forwards the remaining native restrictions and `delta` without validating them.
    /// The destination-sized ROI starts at `source_offset` within the full source image.
    /// Queues work on the selected stream without synchronizing. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if the source offset is invalid, ROI endpoints overflow or exceed the
    /// source image, mask dimensions are nonpositive, CUDA context binding or region-context
    /// validation fails, or NPP rejects the threshold parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw). Source pixels must be
    /// initialized and ready for the selected stream. Keep the source read-only and destination
    /// exclusively writable and disjoint during native use. Retain both allocations and the
    /// stream context through queued completion, including submission errors and captured
    /// replay. Uphold native layout, mask, sampling, byte-stride, alignment, and bounds requirements.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, C1>,
        mask_size: Size,
        delta: f32,
        value_greater_than: Self,
        value_less_or_equal: Self,
        border_type: BorderType,
    ) -> Result<()>;
}

unsafe impl FilterThresholdAdaptiveBoxBorderC1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, C1>,
        mask_size: Size,
        delta: f32,
        value_greater_than: Self,
        value_less_or_equal: Self,
        border_type: BorderType,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            filter_threshold_adaptive_box_border_u8_c1(
                stream_context,
                source,
                source_offset,
                destination,
                mask_size,
                delta,
                value_greater_than,
                value_less_or_equal,
                border_type,
            )
        }
    }
}

/// Thresholds each source pixel against its local box mean minus `delta`.
///
/// Writes `value_greater_than` when the source pixel is greater than that adjusted mean;
/// otherwise writes `value_less_or_equal`. NPP requires a square mask with odd side lengths
/// and supports only [`BorderType::Replicate`]. The wrapper checks positive mask dimensions
/// and forwards the remaining native restrictions and `delta` without validating them.
/// The destination-sized ROI starts at `source_offset` within the full source image.
/// Queues work on the selected stream without synchronizing. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if the source offset is invalid, ROI endpoints overflow or exceed the
/// source image, mask dimensions are nonpositive, CUDA context binding or region-context
/// validation fails, or NPP rejects the threshold parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Source pixels must be
/// initialized and ready for the selected stream. Keep the source read-only and destination
/// exclusively writable and disjoint during native use. Retain both allocations and the
/// stream context through queued completion, including submission errors and captured
/// replay. Uphold native layout, mask, sampling, byte-stride, alignment, and bounds requirements.
pub unsafe fn filter_threshold_adaptive_box_border<T: FilterThresholdAdaptiveBoxBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, C1>,
    mask_size: Size,
    delta: f32,
    value_greater_than: T,
    value_less_or_equal: T,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            delta,
            value_greater_than,
            value_less_or_equal,
            border_type,
        )
    }
}
