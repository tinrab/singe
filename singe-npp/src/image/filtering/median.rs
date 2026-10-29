use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_filter_mask, validate_same_size},
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
    utility::to_usize,
};

impl_filter_median!(
    filter_median_u8_c1_buffer_size,
    filter_median_u8_c1,
    u8,
    C1,
    nppiFilterMedianGetBufferSize_8u_C1R_Ctx,
    nppiFilterMedian_8u_C1R_Ctx
);

impl_filter_median!(
    filter_median_u8_c3_buffer_size,
    filter_median_u8_c3,
    u8,
    C3,
    nppiFilterMedianGetBufferSize_8u_C3R_Ctx,
    nppiFilterMedian_8u_C3R_Ctx
);

impl_filter_median!(
    filter_median_u8_c4_buffer_size,
    filter_median_u8_c4,
    u8,
    C4,
    nppiFilterMedianGetBufferSize_8u_C4R_Ctx,
    nppiFilterMedian_8u_C4R_Ctx
);

impl_filter_median!(
    filter_median_u8_ac4_buffer_size,
    filter_median_u8_ac4,
    u8,
    AC4,
    nppiFilterMedianGetBufferSize_8u_AC4R_Ctx,
    nppiFilterMedian_8u_AC4R_Ctx
);

impl_filter_median!(
    filter_median_u16_c1_buffer_size,
    filter_median_u16_c1,
    u16,
    C1,
    nppiFilterMedianGetBufferSize_16u_C1R_Ctx,
    nppiFilterMedian_16u_C1R_Ctx
);

impl_filter_median!(
    filter_median_u16_c3_buffer_size,
    filter_median_u16_c3,
    u16,
    C3,
    nppiFilterMedianGetBufferSize_16u_C3R_Ctx,
    nppiFilterMedian_16u_C3R_Ctx
);

impl_filter_median!(
    filter_median_u16_c4_buffer_size,
    filter_median_u16_c4,
    u16,
    C4,
    nppiFilterMedianGetBufferSize_16u_C4R_Ctx,
    nppiFilterMedian_16u_C4R_Ctx
);

impl_filter_median!(
    filter_median_u16_ac4_buffer_size,
    filter_median_u16_ac4,
    u16,
    AC4,
    nppiFilterMedianGetBufferSize_16u_AC4R_Ctx,
    nppiFilterMedian_16u_AC4R_Ctx
);

impl_filter_median!(
    filter_median_i16_c1_buffer_size,
    filter_median_i16_c1,
    i16,
    C1,
    nppiFilterMedianGetBufferSize_16s_C1R_Ctx,
    nppiFilterMedian_16s_C1R_Ctx
);

impl_filter_median!(
    filter_median_i16_c3_buffer_size,
    filter_median_i16_c3,
    i16,
    C3,
    nppiFilterMedianGetBufferSize_16s_C3R_Ctx,
    nppiFilterMedian_16s_C3R_Ctx
);

impl_filter_median!(
    filter_median_i16_c4_buffer_size,
    filter_median_i16_c4,
    i16,
    C4,
    nppiFilterMedianGetBufferSize_16s_C4R_Ctx,
    nppiFilterMedian_16s_C4R_Ctx
);

impl_filter_median!(
    filter_median_i16_ac4_buffer_size,
    filter_median_i16_ac4,
    i16,
    AC4,
    nppiFilterMedianGetBufferSize_16s_AC4R_Ctx,
    nppiFilterMedian_16s_AC4R_Ctx
);

impl_filter_median!(
    filter_median_f32_c1_buffer_size,
    filter_median_f32_c1,
    f32,
    C1,
    nppiFilterMedianGetBufferSize_32f_C1R_Ctx,
    nppiFilterMedian_32f_C1R_Ctx
);

impl_filter_median!(
    filter_median_f32_c3_buffer_size,
    filter_median_f32_c3,
    f32,
    C3,
    nppiFilterMedianGetBufferSize_32f_C3R_Ctx,
    nppiFilterMedian_32f_C3R_Ctx
);

impl_filter_median!(
    filter_median_f32_c4_buffer_size,
    filter_median_f32_c4,
    f32,
    C4,
    nppiFilterMedianGetBufferSize_32f_C4R_Ctx,
    nppiFilterMedian_32f_C4R_Ctx
);

impl_filter_median!(
    filter_median_f32_ac4_buffer_size,
    filter_median_f32_ac4,
    f32,
    AC4,
    nppiFilterMedianGetBufferSize_32f_AC4R_Ctx,
    nppiFilterMedian_32f_AC4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u8_c1_buffer_size,
    filter_median_border_u8_c1,
    u8,
    C1,
    nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx,
    nppiFilterMedianBorder_8u_C1R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u8_c3_buffer_size,
    filter_median_border_u8_c3,
    u8,
    C3,
    nppiFilterMedianBorderGetBufferSize_8u_C3R_Ctx,
    nppiFilterMedianBorder_8u_C3R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u8_c4_buffer_size,
    filter_median_border_u8_c4,
    u8,
    C4,
    nppiFilterMedianBorderGetBufferSize_8u_C4R_Ctx,
    nppiFilterMedianBorder_8u_C4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u8_ac4_buffer_size,
    filter_median_border_u8_ac4,
    u8,
    AC4,
    nppiFilterMedianBorderGetBufferSize_8u_AC4R_Ctx,
    nppiFilterMedianBorder_8u_AC4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u16_c1_buffer_size,
    filter_median_border_u16_c1,
    u16,
    C1,
    nppiFilterMedianBorderGetBufferSize_16u_C1R_Ctx,
    nppiFilterMedianBorder_16u_C1R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u16_c3_buffer_size,
    filter_median_border_u16_c3,
    u16,
    C3,
    nppiFilterMedianBorderGetBufferSize_16u_C3R_Ctx,
    nppiFilterMedianBorder_16u_C3R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u16_c4_buffer_size,
    filter_median_border_u16_c4,
    u16,
    C4,
    nppiFilterMedianBorderGetBufferSize_16u_C4R_Ctx,
    nppiFilterMedianBorder_16u_C4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u16_ac4_buffer_size,
    filter_median_border_u16_ac4,
    u16,
    AC4,
    nppiFilterMedianBorderGetBufferSize_16u_AC4R_Ctx,
    nppiFilterMedianBorder_16u_AC4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_i16_c1_buffer_size,
    filter_median_border_i16_c1,
    i16,
    C1,
    nppiFilterMedianBorderGetBufferSize_16s_C1R_Ctx,
    nppiFilterMedianBorder_16s_C1R_Ctx
);

impl_filter_median_border!(
    filter_median_border_i16_c3_buffer_size,
    filter_median_border_i16_c3,
    i16,
    C3,
    nppiFilterMedianBorderGetBufferSize_16s_C3R_Ctx,
    nppiFilterMedianBorder_16s_C3R_Ctx
);

impl_filter_median_border!(
    filter_median_border_i16_c4_buffer_size,
    filter_median_border_i16_c4,
    i16,
    C4,
    nppiFilterMedianBorderGetBufferSize_16s_C4R_Ctx,
    nppiFilterMedianBorder_16s_C4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_i16_ac4_buffer_size,
    filter_median_border_i16_ac4,
    i16,
    AC4,
    nppiFilterMedianBorderGetBufferSize_16s_AC4R_Ctx,
    nppiFilterMedianBorder_16s_AC4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_f32_c1_buffer_size,
    filter_median_border_f32_c1,
    f32,
    C1,
    nppiFilterMedianBorderGetBufferSize_32f_C1R_Ctx,
    nppiFilterMedianBorder_32f_C1R_Ctx
);

impl_filter_median_border!(
    filter_median_border_f32_c3_buffer_size,
    filter_median_border_f32_c3,
    f32,
    C3,
    nppiFilterMedianBorderGetBufferSize_32f_C3R_Ctx,
    nppiFilterMedianBorder_32f_C3R_Ctx
);

impl_filter_median_border!(
    filter_median_border_f32_c4_buffer_size,
    filter_median_border_f32_c4,
    f32,
    C4,
    nppiFilterMedianBorderGetBufferSize_32f_C4R_Ctx,
    nppiFilterMedianBorder_32f_C4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_f32_ac4_buffer_size,
    filter_median_border_f32_ac4,
    f32,
    AC4,
    nppiFilterMedianBorderGetBufferSize_32f_AC4R_Ctx,
    nppiFilterMedianBorder_32f_AC4R_Ctx
);

/// Dispatches median filtering.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations, neighborhood
/// and anchor semantics, and scratch-size requirements. Uphold the method's validation,
/// source halo requirements where applicable, and completed native execution contract.
pub unsafe trait FilterMedian<L: ChannelLayout>: DataTypeLike {
    /// Returns the scratch size in bytes for median filtering of `roi` using `mask_size`.
    ///
    /// # Errors
    ///
    /// Returns an error if mask dimensions are nonpositive, NPP rejects the ROI or mask, or the returned size cannot be represented as `usize`.
    fn filter_median_buffer_size(
        stream_context: &StreamContext,
        roi: Size,
        mask_size: Size,
    ) -> Result<usize>;

    /// Applies a median filter over the neighborhood defined by `mask_size` and `anchor`.
    ///
    /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
    /// For AC4 layouts, the destination alpha channel is not processed.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// Source and destination image sizes must match.
    /// Mask dimensions must be positive, and `anchor` must lie within the mask.
    /// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_median(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

macro_rules! impl_filter_median_dispatch {
    ($pixel_ty:ty, $layout:ty, $buffer_size_name:ident, $name:ident) => {
        unsafe impl FilterMedian<$layout> for $pixel_ty {
            fn filter_median_buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                mask_size: Size,
            ) -> Result<usize> {
                $buffer_size_name(stream_context, roi, mask_size)
            }

            unsafe fn filter_median(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination, mask_size, anchor) }
            }
        }
    };
}

impl_filter_median_dispatch!(u8, C1, filter_median_u8_c1_buffer_size, filter_median_u8_c1);

impl_filter_median_dispatch!(u8, C3, filter_median_u8_c3_buffer_size, filter_median_u8_c3);

impl_filter_median_dispatch!(u8, C4, filter_median_u8_c4_buffer_size, filter_median_u8_c4);

impl_filter_median_dispatch!(
    u8,
    AC4,
    filter_median_u8_ac4_buffer_size,
    filter_median_u8_ac4
);

impl_filter_median_dispatch!(
    u16,
    C1,
    filter_median_u16_c1_buffer_size,
    filter_median_u16_c1
);

impl_filter_median_dispatch!(
    u16,
    C3,
    filter_median_u16_c3_buffer_size,
    filter_median_u16_c3
);

impl_filter_median_dispatch!(
    u16,
    C4,
    filter_median_u16_c4_buffer_size,
    filter_median_u16_c4
);

impl_filter_median_dispatch!(
    u16,
    AC4,
    filter_median_u16_ac4_buffer_size,
    filter_median_u16_ac4
);

impl_filter_median_dispatch!(
    i16,
    C1,
    filter_median_i16_c1_buffer_size,
    filter_median_i16_c1
);

impl_filter_median_dispatch!(
    i16,
    C3,
    filter_median_i16_c3_buffer_size,
    filter_median_i16_c3
);

impl_filter_median_dispatch!(
    i16,
    C4,
    filter_median_i16_c4_buffer_size,
    filter_median_i16_c4
);

impl_filter_median_dispatch!(
    i16,
    AC4,
    filter_median_i16_ac4_buffer_size,
    filter_median_i16_ac4
);

impl_filter_median_dispatch!(
    f32,
    C1,
    filter_median_f32_c1_buffer_size,
    filter_median_f32_c1
);

impl_filter_median_dispatch!(
    f32,
    C3,
    filter_median_f32_c3_buffer_size,
    filter_median_f32_c3
);

impl_filter_median_dispatch!(
    f32,
    C4,
    filter_median_f32_c4_buffer_size,
    filter_median_f32_c4
);

impl_filter_median_dispatch!(
    f32,
    AC4,
    filter_median_f32_ac4_buffer_size,
    filter_median_f32_ac4
);

/// Returns the scratch size in bytes for median filtering of `roi` using `mask_size`.
///
/// # Errors
///
/// Returns an error if mask dimensions are nonpositive, NPP rejects the ROI or mask, or the returned size cannot be represented as `usize`.
pub fn filter_median_buffer_size<T, L>(
    stream_context: &StreamContext,
    roi: Size,
    mask_size: Size,
) -> Result<usize>
where
    T: FilterMedian<L>,
    L: ChannelLayout,
{
    T::filter_median_buffer_size(stream_context, roi, mask_size)
}

/// Applies a median filter over the neighborhood defined by `mask_size` and `anchor`.
///
/// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
/// For AC4 layouts, the destination alpha channel is not processed.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// Source and destination image sizes must match.
/// Mask dimensions must be positive, and `anchor` must lie within the mask.
/// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Keep image allocations valid until this call completes its native work.
/// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_median<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: Size,
    anchor: Point,
) -> Result<()>
where
    T: FilterMedian<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_median(stream_context, source, destination, mask_size, anchor) }
}

/// Dispatches median filtering with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations, neighborhood
/// and anchor semantics, and scratch-size requirements. Uphold the method's validation,
/// source halo requirements where applicable, and completed native execution contract.
pub unsafe trait FilterMedianBorder<L: ChannelLayout>: DataTypeLike {
    /// Returns the scratch size in bytes for median filtering of `roi` using `mask_size`.
    ///
    /// # Errors
    ///
    /// Returns an error if mask dimensions are nonpositive, NPP rejects the ROI or mask or border mode, or the returned size cannot be represented as `usize`.
    fn filter_median_border_buffer_size(
        stream_context: &StreamContext,
        roi: Size,
        mask_size: Size,
        border_type: BorderType,
    ) -> Result<usize>;

    /// Applies a median filter over the neighborhood defined by `mask_size` and `anchor` with border control.
    ///
    /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
    /// For AC4 layouts, the destination alpha channel is not processed.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
    /// Mask dimensions must be positive, and `anchor` must lie within the mask.
    /// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_median_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_median_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $buffer_size_name:ident, $name:ident) => {
        unsafe impl FilterMedianBorder<$layout> for $pixel_ty {
            fn filter_median_border_buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                mask_size: Size,
                border_type: BorderType,
            ) -> Result<usize> {
                $buffer_size_name(stream_context, roi, mask_size, border_type)
            }

            unsafe fn filter_median_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        anchor,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_median_border_dispatch!(
    u8,
    C1,
    filter_median_border_u8_c1_buffer_size,
    filter_median_border_u8_c1
);

impl_filter_median_border_dispatch!(
    u8,
    C3,
    filter_median_border_u8_c3_buffer_size,
    filter_median_border_u8_c3
);

impl_filter_median_border_dispatch!(
    u8,
    C4,
    filter_median_border_u8_c4_buffer_size,
    filter_median_border_u8_c4
);

impl_filter_median_border_dispatch!(
    u8,
    AC4,
    filter_median_border_u8_ac4_buffer_size,
    filter_median_border_u8_ac4
);

impl_filter_median_border_dispatch!(
    u16,
    C1,
    filter_median_border_u16_c1_buffer_size,
    filter_median_border_u16_c1
);

impl_filter_median_border_dispatch!(
    u16,
    C3,
    filter_median_border_u16_c3_buffer_size,
    filter_median_border_u16_c3
);

impl_filter_median_border_dispatch!(
    u16,
    C4,
    filter_median_border_u16_c4_buffer_size,
    filter_median_border_u16_c4
);

impl_filter_median_border_dispatch!(
    u16,
    AC4,
    filter_median_border_u16_ac4_buffer_size,
    filter_median_border_u16_ac4
);

impl_filter_median_border_dispatch!(
    i16,
    C1,
    filter_median_border_i16_c1_buffer_size,
    filter_median_border_i16_c1
);

impl_filter_median_border_dispatch!(
    i16,
    C3,
    filter_median_border_i16_c3_buffer_size,
    filter_median_border_i16_c3
);

impl_filter_median_border_dispatch!(
    i16,
    C4,
    filter_median_border_i16_c4_buffer_size,
    filter_median_border_i16_c4
);

impl_filter_median_border_dispatch!(
    i16,
    AC4,
    filter_median_border_i16_ac4_buffer_size,
    filter_median_border_i16_ac4
);

impl_filter_median_border_dispatch!(
    f32,
    C1,
    filter_median_border_f32_c1_buffer_size,
    filter_median_border_f32_c1
);

impl_filter_median_border_dispatch!(
    f32,
    C3,
    filter_median_border_f32_c3_buffer_size,
    filter_median_border_f32_c3
);

impl_filter_median_border_dispatch!(
    f32,
    C4,
    filter_median_border_f32_c4_buffer_size,
    filter_median_border_f32_c4
);

impl_filter_median_border_dispatch!(
    f32,
    AC4,
    filter_median_border_f32_ac4_buffer_size,
    filter_median_border_f32_ac4
);

/// Returns the scratch size in bytes for median filtering of `roi` using `mask_size`.
///
/// # Errors
///
/// Returns an error if mask dimensions are nonpositive, NPP rejects the ROI or mask or border mode, or the returned size cannot be represented as `usize`.
pub fn filter_median_border_buffer_size<T, L>(
    stream_context: &StreamContext,
    roi: Size,
    mask_size: Size,
    border_type: BorderType,
) -> Result<usize>
where
    T: FilterMedianBorder<L>,
    L: ChannelLayout,
{
    T::filter_median_border_buffer_size(stream_context, roi, mask_size, border_type)
}

/// Applies a median filter over the neighborhood defined by `mask_size` and `anchor` with border control.
///
/// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
/// For AC4 layouts, the destination alpha channel is not processed.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
/// Mask dimensions must be positive, and `anchor` must lie within the mask.
/// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Keep image allocations valid until this call completes its native work.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_median_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: Size,
    anchor: Point,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterMedianBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_median_border(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }
}
