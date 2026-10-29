use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{
            validate_border_roi, validate_same_size, validate_unsharp_parameters,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
    utility::to_usize,
};

impl_filter_directional!(
    "Single channel 8-bit unsigned sharpening filter.",
    filter_sharpen_u8_c1,
    u8,
    C1,
    nppiFilterSharpen_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned sharpening filter.",
    filter_sharpen_u8_c3,
    u8,
    C3,
    nppiFilterSharpen_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned sharpening filter.",
    filter_sharpen_u8_c4,
    u8,
    C4,
    nppiFilterSharpen_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned sharpening filter, ignoring alpha channel.",
    filter_sharpen_u8_ac4,
    u8,
    AC4,
    nppiFilterSharpen_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit unsigned sharpening filter.",
    filter_sharpen_u16_c1,
    u16,
    C1,
    nppiFilterSharpen_16u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit unsigned sharpening filter.",
    filter_sharpen_u16_c3,
    u16,
    C3,
    nppiFilterSharpen_16u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit unsigned sharpening filter.",
    filter_sharpen_u16_c4,
    u16,
    C4,
    nppiFilterSharpen_16u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit unsigned sharpening filter, ignoring alpha channel.",
    filter_sharpen_u16_ac4,
    u16,
    AC4,
    nppiFilterSharpen_16u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed sharpening filter.",
    filter_sharpen_i16_c1,
    i16,
    C1,
    nppiFilterSharpen_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed sharpening filter.",
    filter_sharpen_i16_c3,
    i16,
    C3,
    nppiFilterSharpen_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed sharpening filter.",
    filter_sharpen_i16_c4,
    i16,
    C4,
    nppiFilterSharpen_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed sharpening filter, ignoring alpha channel.",
    filter_sharpen_i16_ac4,
    i16,
    AC4,
    nppiFilterSharpen_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point sharpening filter.",
    filter_sharpen_f32_c1,
    f32,
    C1,
    nppiFilterSharpen_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point sharpening filter.",
    filter_sharpen_f32_c3,
    f32,
    C3,
    nppiFilterSharpen_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point sharpening filter.",
    filter_sharpen_f32_c4,
    f32,
    C4,
    nppiFilterSharpen_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point sharpening filter, ignoring alpha channel.",
    filter_sharpen_f32_ac4,
    f32,
    AC4,
    nppiFilterSharpen_32f_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned sharpening filter with border control.",
    filter_sharpen_border_u8_c1,
    u8,
    C1,
    nppiFilterSharpenBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned sharpening filter with border control.",
    filter_sharpen_border_u8_c3,
    u8,
    C3,
    nppiFilterSharpenBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned sharpening filter with border control.",
    filter_sharpen_border_u8_c4,
    u8,
    C4,
    nppiFilterSharpenBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned sharpening filter with border control, ignoring alpha channel.",
    filter_sharpen_border_u8_ac4,
    u8,
    AC4,
    nppiFilterSharpenBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit unsigned sharpening filter with border control.",
    filter_sharpen_border_u16_c1,
    u16,
    C1,
    nppiFilterSharpenBorder_16u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit unsigned sharpening filter with border control.",
    filter_sharpen_border_u16_c3,
    u16,
    C3,
    nppiFilterSharpenBorder_16u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit unsigned sharpening filter with border control.",
    filter_sharpen_border_u16_c4,
    u16,
    C4,
    nppiFilterSharpenBorder_16u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit unsigned sharpening filter with border control, ignoring alpha channel.",
    filter_sharpen_border_u16_ac4,
    u16,
    AC4,
    nppiFilterSharpenBorder_16u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed sharpening filter with border control.",
    filter_sharpen_border_i16_c1,
    i16,
    C1,
    nppiFilterSharpenBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed sharpening filter with border control.",
    filter_sharpen_border_i16_c3,
    i16,
    C3,
    nppiFilterSharpenBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed sharpening filter with border control.",
    filter_sharpen_border_i16_c4,
    i16,
    C4,
    nppiFilterSharpenBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed sharpening filter with border control, ignoring alpha channel.",
    filter_sharpen_border_i16_ac4,
    i16,
    AC4,
    nppiFilterSharpenBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point sharpening filter with border control.",
    filter_sharpen_border_f32_c1,
    f32,
    C1,
    nppiFilterSharpenBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point sharpening filter with border control.",
    filter_sharpen_border_f32_c3,
    f32,
    C3,
    nppiFilterSharpenBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point sharpening filter with border control.",
    filter_sharpen_border_f32_c4,
    f32,
    C4,
    nppiFilterSharpenBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point sharpening filter with border control, ignoring alpha channel.",
    filter_sharpen_border_f32_ac4,
    f32,
    AC4,
    nppiFilterSharpenBorder_32f_AC4R_Ctx
);

/// Dispatches sharpening.
///
/// # Safety
///
/// Implementations must preserve the native pixel and layout ABI and the selected filter semantics.
/// Uphold the method's geometry validation, memory-access, and execution-lifetime contracts.
pub unsafe trait FilterSharpen<L: ChannelLayout>: DataTypeLike {
    /// Applies the sharpening filter.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the image geometry or filter parameters.
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
    unsafe fn filter_sharpen(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_filter_sharpen_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterSharpen<$layout> for $pixel_ty {
            unsafe fn filter_sharpen(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination) }
            }
        }
    };
}

impl_filter_sharpen_dispatch!(u8, C1, filter_sharpen_u8_c1);

impl_filter_sharpen_dispatch!(u8, C3, filter_sharpen_u8_c3);

impl_filter_sharpen_dispatch!(u8, C4, filter_sharpen_u8_c4);

impl_filter_sharpen_dispatch!(u8, AC4, filter_sharpen_u8_ac4);

impl_filter_sharpen_dispatch!(u16, C1, filter_sharpen_u16_c1);

impl_filter_sharpen_dispatch!(u16, C3, filter_sharpen_u16_c3);

impl_filter_sharpen_dispatch!(u16, C4, filter_sharpen_u16_c4);

impl_filter_sharpen_dispatch!(u16, AC4, filter_sharpen_u16_ac4);

impl_filter_sharpen_dispatch!(i16, C1, filter_sharpen_i16_c1);

impl_filter_sharpen_dispatch!(i16, C3, filter_sharpen_i16_c3);

impl_filter_sharpen_dispatch!(i16, C4, filter_sharpen_i16_c4);

impl_filter_sharpen_dispatch!(i16, AC4, filter_sharpen_i16_ac4);

impl_filter_sharpen_dispatch!(f32, C1, filter_sharpen_f32_c1);

impl_filter_sharpen_dispatch!(f32, C3, filter_sharpen_f32_c3);

impl_filter_sharpen_dispatch!(f32, C4, filter_sharpen_f32_c4);

impl_filter_sharpen_dispatch!(f32, AC4, filter_sharpen_f32_ac4);

/// Applies the sharpening filter.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the image geometry or filter parameters.
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
pub unsafe fn filter_sharpen<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterSharpen<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sharpen(stream_context, source, destination) }
}

/// Dispatches sharpening with border control.
///
/// # Safety
///
/// Implementations must preserve the native pixel and layout ABI and the selected filter semantics.
/// Uphold the method's geometry validation, memory-access, and execution-lifetime contracts.
pub unsafe trait FilterSharpenBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the sharpening filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the image geometry or filter parameters.
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
    unsafe fn filter_sharpen_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_sharpen_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterSharpenBorder<$layout> for $pixel_ty {
            unsafe fn filter_sharpen_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_sharpen_border_dispatch!(u8, C1, filter_sharpen_border_u8_c1);

impl_filter_sharpen_border_dispatch!(u8, C3, filter_sharpen_border_u8_c3);

impl_filter_sharpen_border_dispatch!(u8, C4, filter_sharpen_border_u8_c4);

impl_filter_sharpen_border_dispatch!(u8, AC4, filter_sharpen_border_u8_ac4);

impl_filter_sharpen_border_dispatch!(u16, C1, filter_sharpen_border_u16_c1);

impl_filter_sharpen_border_dispatch!(u16, C3, filter_sharpen_border_u16_c3);

impl_filter_sharpen_border_dispatch!(u16, C4, filter_sharpen_border_u16_c4);

impl_filter_sharpen_border_dispatch!(u16, AC4, filter_sharpen_border_u16_ac4);

impl_filter_sharpen_border_dispatch!(i16, C1, filter_sharpen_border_i16_c1);

impl_filter_sharpen_border_dispatch!(i16, C3, filter_sharpen_border_i16_c3);

impl_filter_sharpen_border_dispatch!(i16, C4, filter_sharpen_border_i16_c4);

impl_filter_sharpen_border_dispatch!(i16, AC4, filter_sharpen_border_i16_ac4);

impl_filter_sharpen_border_dispatch!(f32, C1, filter_sharpen_border_f32_c1);

impl_filter_sharpen_border_dispatch!(f32, C3, filter_sharpen_border_f32_c3);

impl_filter_sharpen_border_dispatch!(f32, C4, filter_sharpen_border_f32_c4);

impl_filter_sharpen_border_dispatch!(f32, AC4, filter_sharpen_border_f32_ac4);

/// Applies the sharpening filter with border control.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the image geometry or filter parameters.
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
pub unsafe fn filter_sharpen_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterSharpenBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sharpen_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

impl_filter_unsharp_border!(
    filter_unsharp_border_u8_c1_buffer_size,
    filter_unsharp_border_u8_c1,
    u8,
    C1,
    nppiFilterUnsharpGetBufferSize_8u_C1R,
    nppiFilterUnsharpBorder_8u_C1R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u8_c3_buffer_size,
    filter_unsharp_border_u8_c3,
    u8,
    C3,
    nppiFilterUnsharpGetBufferSize_8u_C3R,
    nppiFilterUnsharpBorder_8u_C3R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u8_c4_buffer_size,
    filter_unsharp_border_u8_c4,
    u8,
    C4,
    nppiFilterUnsharpGetBufferSize_8u_C4R,
    nppiFilterUnsharpBorder_8u_C4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u8_ac4_buffer_size,
    filter_unsharp_border_u8_ac4,
    u8,
    AC4,
    nppiFilterUnsharpGetBufferSize_8u_AC4R,
    nppiFilterUnsharpBorder_8u_AC4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u16_c1_buffer_size,
    filter_unsharp_border_u16_c1,
    u16,
    C1,
    nppiFilterUnsharpGetBufferSize_16u_C1R,
    nppiFilterUnsharpBorder_16u_C1R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u16_c3_buffer_size,
    filter_unsharp_border_u16_c3,
    u16,
    C3,
    nppiFilterUnsharpGetBufferSize_16u_C3R,
    nppiFilterUnsharpBorder_16u_C3R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u16_c4_buffer_size,
    filter_unsharp_border_u16_c4,
    u16,
    C4,
    nppiFilterUnsharpGetBufferSize_16u_C4R,
    nppiFilterUnsharpBorder_16u_C4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_u16_ac4_buffer_size,
    filter_unsharp_border_u16_ac4,
    u16,
    AC4,
    nppiFilterUnsharpGetBufferSize_16u_AC4R,
    nppiFilterUnsharpBorder_16u_AC4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_i16_c1_buffer_size,
    filter_unsharp_border_i16_c1,
    i16,
    C1,
    nppiFilterUnsharpGetBufferSize_16s_C1R,
    nppiFilterUnsharpBorder_16s_C1R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_i16_c3_buffer_size,
    filter_unsharp_border_i16_c3,
    i16,
    C3,
    nppiFilterUnsharpGetBufferSize_16s_C3R,
    nppiFilterUnsharpBorder_16s_C3R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_i16_c4_buffer_size,
    filter_unsharp_border_i16_c4,
    i16,
    C4,
    nppiFilterUnsharpGetBufferSize_16s_C4R,
    nppiFilterUnsharpBorder_16s_C4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_i16_ac4_buffer_size,
    filter_unsharp_border_i16_ac4,
    i16,
    AC4,
    nppiFilterUnsharpGetBufferSize_16s_AC4R,
    nppiFilterUnsharpBorder_16s_AC4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_f32_c1_buffer_size,
    filter_unsharp_border_f32_c1,
    f32,
    C1,
    nppiFilterUnsharpGetBufferSize_32f_C1R,
    nppiFilterUnsharpBorder_32f_C1R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_f32_c3_buffer_size,
    filter_unsharp_border_f32_c3,
    f32,
    C3,
    nppiFilterUnsharpGetBufferSize_32f_C3R,
    nppiFilterUnsharpBorder_32f_C3R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_f32_c4_buffer_size,
    filter_unsharp_border_f32_c4,
    f32,
    C4,
    nppiFilterUnsharpGetBufferSize_32f_C4R,
    nppiFilterUnsharpBorder_32f_C4R_Ctx
);

impl_filter_unsharp_border!(
    filter_unsharp_border_f32_ac4_buffer_size,
    filter_unsharp_border_f32_ac4,
    f32,
    AC4,
    nppiFilterUnsharpGetBufferSize_32f_AC4R,
    nppiFilterUnsharpBorder_32f_AC4R_Ctx
);

/// Dispatches unsharp-mask sharpening with border control.
///
/// # Safety
///
/// Implementations must preserve the native pixel and layout ABI and the selected filter semantics.
/// Uphold the method's geometry validation, memory-access, and execution-lifetime contracts.
pub unsafe trait FilterUnsharpBorder<L: ChannelLayout>: DataTypeLike {
    /// Returns the scratch size in bytes for unsharp-mask sharpening with the given `radius` and `sigma`.
    ///
    /// # Errors
    ///
    /// Returns an error if `radius` or `sigma` does not compare greater than zero, NPP rejects the query, or the returned size cannot be represented as `usize`.
    fn filter_unsharp_border_buffer_size(radius: f32, sigma: f32) -> Result<usize>;

    /// Sharpens the source using an unsharp mask with border control.
    ///
    /// `radius` is the Gaussian radius in pixels excluding the center pixel; `sigma` is its standard deviation in pixels.
    /// `weight` scales the high-pass contribution, and `threshold` selects the minimum absolute high-pass value to add back.
    /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
    /// For AC4 layouts, the destination alpha channel is not processed.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
    /// `radius` and `sigma` must compare greater than zero; the wrapper does not validate `weight` or `threshold`.
    /// The native unsharp filter supports `BorderType::Replicate`; the wrapper forwards `border_type` without validating it.
    /// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_unsharp_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_unsharp_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $buffer_size_name:ident, $name:ident) => {
        unsafe impl FilterUnsharpBorder<$layout> for $pixel_ty {
            fn filter_unsharp_border_buffer_size(radius: f32, sigma: f32) -> Result<usize> {
                $buffer_size_name(radius, sigma)
            }

            unsafe fn filter_unsharp_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                radius: f32,
                sigma: f32,
                weight: f32,
                threshold: f32,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        radius,
                        sigma,
                        weight,
                        threshold,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_unsharp_border_dispatch!(
    u8,
    C1,
    filter_unsharp_border_u8_c1_buffer_size,
    filter_unsharp_border_u8_c1
);

impl_filter_unsharp_border_dispatch!(
    u8,
    C3,
    filter_unsharp_border_u8_c3_buffer_size,
    filter_unsharp_border_u8_c3
);

impl_filter_unsharp_border_dispatch!(
    u8,
    C4,
    filter_unsharp_border_u8_c4_buffer_size,
    filter_unsharp_border_u8_c4
);

impl_filter_unsharp_border_dispatch!(
    u8,
    AC4,
    filter_unsharp_border_u8_ac4_buffer_size,
    filter_unsharp_border_u8_ac4
);

impl_filter_unsharp_border_dispatch!(
    u16,
    C1,
    filter_unsharp_border_u16_c1_buffer_size,
    filter_unsharp_border_u16_c1
);

impl_filter_unsharp_border_dispatch!(
    u16,
    C3,
    filter_unsharp_border_u16_c3_buffer_size,
    filter_unsharp_border_u16_c3
);

impl_filter_unsharp_border_dispatch!(
    u16,
    C4,
    filter_unsharp_border_u16_c4_buffer_size,
    filter_unsharp_border_u16_c4
);

impl_filter_unsharp_border_dispatch!(
    u16,
    AC4,
    filter_unsharp_border_u16_ac4_buffer_size,
    filter_unsharp_border_u16_ac4
);

impl_filter_unsharp_border_dispatch!(
    i16,
    C1,
    filter_unsharp_border_i16_c1_buffer_size,
    filter_unsharp_border_i16_c1
);

impl_filter_unsharp_border_dispatch!(
    i16,
    C3,
    filter_unsharp_border_i16_c3_buffer_size,
    filter_unsharp_border_i16_c3
);

impl_filter_unsharp_border_dispatch!(
    i16,
    C4,
    filter_unsharp_border_i16_c4_buffer_size,
    filter_unsharp_border_i16_c4
);

impl_filter_unsharp_border_dispatch!(
    i16,
    AC4,
    filter_unsharp_border_i16_ac4_buffer_size,
    filter_unsharp_border_i16_ac4
);

impl_filter_unsharp_border_dispatch!(
    f32,
    C1,
    filter_unsharp_border_f32_c1_buffer_size,
    filter_unsharp_border_f32_c1
);

impl_filter_unsharp_border_dispatch!(
    f32,
    C3,
    filter_unsharp_border_f32_c3_buffer_size,
    filter_unsharp_border_f32_c3
);

impl_filter_unsharp_border_dispatch!(
    f32,
    C4,
    filter_unsharp_border_f32_c4_buffer_size,
    filter_unsharp_border_f32_c4
);

impl_filter_unsharp_border_dispatch!(
    f32,
    AC4,
    filter_unsharp_border_f32_ac4_buffer_size,
    filter_unsharp_border_f32_ac4
);

/// Returns the scratch size in bytes for unsharp-mask sharpening with the given `radius` and `sigma`.
///
/// # Errors
///
/// Returns an error if `radius` or `sigma` does not compare greater than zero, NPP rejects the query, or the returned size cannot be represented as `usize`.
pub fn filter_unsharp_border_buffer_size<T, L>(radius: f32, sigma: f32) -> Result<usize>
where
    T: FilterUnsharpBorder<L>,
    L: ChannelLayout,
{
    T::filter_unsharp_border_buffer_size(radius, sigma)
}

/// Sharpens the source using an unsharp mask with border control.
///
/// `radius` is the Gaussian radius in pixels excluding the center pixel; `sigma` is its standard deviation in pixels.
/// `weight` scales the high-pass contribution, and `threshold` selects the minimum absolute high-pass value to add back.
/// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
/// For AC4 layouts, the destination alpha channel is not processed.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
/// `radius` and `sigma` must compare greater than zero; the wrapper does not validate `weight` or `threshold`.
/// The native unsharp filter supports `BorderType::Replicate`; the wrapper forwards `border_type` without validating it.
/// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Keep image allocations valid until this call completes its native work.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_unsharp_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    radius: f32,
    sigma: f32,
    weight: f32,
    threshold: f32,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterUnsharpBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_unsharp_border(
            stream_context,
            source,
            source_offset,
            destination,
            radius,
            sigma,
            weight,
            threshold,
            border_type,
        )
    }
}
