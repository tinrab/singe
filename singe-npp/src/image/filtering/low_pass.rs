use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_same_size},
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, MaskSize, Point},
};

impl_filter_masked_kernel!(
    "Single channel 8-bit unsigned low-pass filter.",
    filter_low_pass_u8_c1,
    u8,
    C1,
    nppiFilterLowPass_8u_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 8-bit unsigned low-pass filter.",
    filter_low_pass_u8_c3,
    u8,
    C3,
    nppiFilterLowPass_8u_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 8-bit unsigned low-pass filter.",
    filter_low_pass_u8_c4,
    u8,
    C4,
    nppiFilterLowPass_8u_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 8-bit unsigned low-pass filter, ignoring alpha channel.",
    filter_low_pass_u8_ac4,
    u8,
    AC4,
    nppiFilterLowPass_8u_AC4R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 16-bit unsigned low-pass filter.",
    filter_low_pass_u16_c1,
    u16,
    C1,
    nppiFilterLowPass_16u_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 16-bit unsigned low-pass filter.",
    filter_low_pass_u16_c3,
    u16,
    C3,
    nppiFilterLowPass_16u_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 16-bit unsigned low-pass filter.",
    filter_low_pass_u16_c4,
    u16,
    C4,
    nppiFilterLowPass_16u_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 16-bit unsigned low-pass filter, ignoring alpha channel.",
    filter_low_pass_u16_ac4,
    u16,
    AC4,
    nppiFilterLowPass_16u_AC4R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 16-bit signed low-pass filter.",
    filter_low_pass_i16_c1,
    i16,
    C1,
    nppiFilterLowPass_16s_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 16-bit signed low-pass filter.",
    filter_low_pass_i16_c3,
    i16,
    C3,
    nppiFilterLowPass_16s_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 16-bit signed low-pass filter.",
    filter_low_pass_i16_c4,
    i16,
    C4,
    nppiFilterLowPass_16s_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 16-bit signed low-pass filter, ignoring alpha channel.",
    filter_low_pass_i16_ac4,
    i16,
    AC4,
    nppiFilterLowPass_16s_AC4R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point low-pass filter.",
    filter_low_pass_f32_c1,
    f32,
    C1,
    nppiFilterLowPass_32f_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 32-bit floating-point low-pass filter.",
    filter_low_pass_f32_c3,
    f32,
    C3,
    nppiFilterLowPass_32f_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 32-bit floating-point low-pass filter.",
    filter_low_pass_f32_c4,
    f32,
    C4,
    nppiFilterLowPass_32f_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.",
    filter_low_pass_f32_ac4,
    f32,
    AC4,
    nppiFilterLowPass_32f_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 8-bit unsigned high-pass filter.",
    filter_low_pass_border_u8_c1,
    u8,
    C1,
    nppiFilterLowPassBorder_8u_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 8-bit unsigned high-pass filter.",
    filter_low_pass_border_u8_c3,
    u8,
    C3,
    nppiFilterLowPassBorder_8u_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 8-bit unsigned high-pass filter.",
    filter_low_pass_border_u8_c4,
    u8,
    C4,
    nppiFilterLowPassBorder_8u_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.",
    filter_low_pass_border_u8_ac4,
    u8,
    AC4,
    nppiFilterLowPassBorder_8u_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 16-bit unsigned high-pass filter.",
    filter_low_pass_border_u16_c1,
    u16,
    C1,
    nppiFilterLowPassBorder_16u_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 16-bit unsigned high-pass filter.",
    filter_low_pass_border_u16_c3,
    u16,
    C3,
    nppiFilterLowPassBorder_16u_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 16-bit unsigned high-pass filter.",
    filter_low_pass_border_u16_c4,
    u16,
    C4,
    nppiFilterLowPassBorder_16u_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.",
    filter_low_pass_border_u16_ac4,
    u16,
    AC4,
    nppiFilterLowPassBorder_16u_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 16-bit signed high-pass filter.",
    filter_low_pass_border_i16_c1,
    i16,
    C1,
    nppiFilterLowPassBorder_16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 16-bit signed high-pass filter.",
    filter_low_pass_border_i16_c3,
    i16,
    C3,
    nppiFilterLowPassBorder_16s_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 16-bit signed high-pass filter.",
    filter_low_pass_border_i16_c4,
    i16,
    C4,
    nppiFilterLowPassBorder_16s_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 16-bit signed high-pass filter, ignoring alpha channel.",
    filter_low_pass_border_i16_ac4,
    i16,
    AC4,
    nppiFilterLowPassBorder_16s_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point high-pass filter.",
    filter_low_pass_border_f32_c1,
    f32,
    C1,
    nppiFilterLowPassBorder_32f_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 32-bit floating-point high-pass filter.",
    filter_low_pass_border_f32_c3,
    f32,
    C3,
    nppiFilterLowPassBorder_32f_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 32-bit floating-point high-pass filter.",
    filter_low_pass_border_f32_c4,
    f32,
    C4,
    nppiFilterLowPassBorder_32f_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.",
    filter_low_pass_border_f32_ac4,
    f32,
    AC4,
    nppiFilterLowPassBorder_32f_AC4R_Ctx
);

/// Dispatches the low-pass filter.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations, filter semantics,
/// and sampling requirements. Uphold the method's validation, source-halo requirements where
/// applicable, coefficient address-space requirements, and queued allocation lifetimes.
pub unsafe trait FilterLowPass<L: ChannelLayout>: DataTypeLike {
    /// Applies the low-pass filter.
    ///
    /// Applies the selected `mask_size` and enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
    /// Source and destination image sizes must match.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_low_pass(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

macro_rules! impl_filter_low_pass_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterLowPass<$layout> for $pixel_ty {
            unsafe fn filter_low_pass(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination, mask_size) }
            }
        }
    };
}

impl_filter_low_pass_dispatch!(u8, C1, filter_low_pass_u8_c1);

impl_filter_low_pass_dispatch!(u8, C3, filter_low_pass_u8_c3);

impl_filter_low_pass_dispatch!(u8, C4, filter_low_pass_u8_c4);

impl_filter_low_pass_dispatch!(u8, AC4, filter_low_pass_u8_ac4);

impl_filter_low_pass_dispatch!(u16, C1, filter_low_pass_u16_c1);

impl_filter_low_pass_dispatch!(u16, C3, filter_low_pass_u16_c3);

impl_filter_low_pass_dispatch!(u16, C4, filter_low_pass_u16_c4);

impl_filter_low_pass_dispatch!(u16, AC4, filter_low_pass_u16_ac4);

impl_filter_low_pass_dispatch!(i16, C1, filter_low_pass_i16_c1);

impl_filter_low_pass_dispatch!(i16, C3, filter_low_pass_i16_c3);

impl_filter_low_pass_dispatch!(i16, C4, filter_low_pass_i16_c4);

impl_filter_low_pass_dispatch!(i16, AC4, filter_low_pass_i16_ac4);

impl_filter_low_pass_dispatch!(f32, C1, filter_low_pass_f32_c1);

impl_filter_low_pass_dispatch!(f32, C3, filter_low_pass_f32_c3);

impl_filter_low_pass_dispatch!(f32, C4, filter_low_pass_f32_c4);

impl_filter_low_pass_dispatch!(f32, AC4, filter_low_pass_f32_ac4);

/// Applies the low-pass filter.
///
/// Applies the selected `mask_size` and enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_low_pass<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: MaskSize,
) -> Result<()>
where
    T: FilterLowPass<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_low_pass(stream_context, source, destination, mask_size) }
}

/// Dispatches the low-pass filter with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations, filter semantics,
/// and sampling requirements. Uphold the method's validation, source-halo requirements where
/// applicable, coefficient address-space requirements, and queued allocation lifetimes.
pub unsafe trait FilterLowPassBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the low-pass filter with border control.
    ///
    /// Applies the selected `mask_size` and enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
    /// `source_offset` must address a source pixel, its ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
    /// NPP can also reject the selected `border_type`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_low_pass_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_low_pass_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterLowPassBorder<$layout> for $pixel_ty {
            unsafe fn filter_low_pass_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                mask_size: MaskSize,
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
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_low_pass_border_dispatch!(u8, C1, filter_low_pass_border_u8_c1);

impl_filter_low_pass_border_dispatch!(u8, C3, filter_low_pass_border_u8_c3);

impl_filter_low_pass_border_dispatch!(u8, C4, filter_low_pass_border_u8_c4);

impl_filter_low_pass_border_dispatch!(u8, AC4, filter_low_pass_border_u8_ac4);

impl_filter_low_pass_border_dispatch!(u16, C1, filter_low_pass_border_u16_c1);

impl_filter_low_pass_border_dispatch!(u16, C3, filter_low_pass_border_u16_c3);

impl_filter_low_pass_border_dispatch!(u16, C4, filter_low_pass_border_u16_c4);

impl_filter_low_pass_border_dispatch!(u16, AC4, filter_low_pass_border_u16_ac4);

impl_filter_low_pass_border_dispatch!(i16, C1, filter_low_pass_border_i16_c1);

impl_filter_low_pass_border_dispatch!(i16, C3, filter_low_pass_border_i16_c3);

impl_filter_low_pass_border_dispatch!(i16, C4, filter_low_pass_border_i16_c4);

impl_filter_low_pass_border_dispatch!(i16, AC4, filter_low_pass_border_i16_ac4);

impl_filter_low_pass_border_dispatch!(f32, C1, filter_low_pass_border_f32_c1);

impl_filter_low_pass_border_dispatch!(f32, C3, filter_low_pass_border_f32_c3);

impl_filter_low_pass_border_dispatch!(f32, C4, filter_low_pass_border_f32_c4);

impl_filter_low_pass_border_dispatch!(f32, AC4, filter_low_pass_border_f32_ac4);

/// Applies the low-pass filter with border control.
///
/// Applies the selected `mask_size` and enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
/// `source_offset` must address a source pixel, its ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
/// NPP can also reject the selected `border_type`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_low_pass_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterLowPassBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_low_pass_border(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
