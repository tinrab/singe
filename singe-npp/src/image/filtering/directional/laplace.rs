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
    "Single channel 8-bit unsigned Laplace filter.",
    filter_laplace_u8_c1,
    u8,
    C1,
    nppiFilterLaplace_8u_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 8-bit unsigned Laplace filter.",
    filter_laplace_u8_c3,
    u8,
    C3,
    nppiFilterLaplace_8u_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 8-bit unsigned Laplace filter.",
    filter_laplace_u8_c4,
    u8,
    C4,
    nppiFilterLaplace_8u_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 8-bit unsigned Laplace filter, ignoring alpha channel.",
    filter_laplace_u8_ac4,
    u8,
    AC4,
    nppiFilterLaplace_8u_AC4R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 16-bit signed Laplace filter.",
    filter_laplace_i16_c1,
    i16,
    C1,
    nppiFilterLaplace_16s_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 16-bit signed Laplace filter.",
    filter_laplace_i16_c3,
    i16,
    C3,
    nppiFilterLaplace_16s_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 16-bit signed Laplace filter.",
    filter_laplace_i16_c4,
    i16,
    C4,
    nppiFilterLaplace_16s_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 16-bit signed Laplace filter, ignoring alpha channel.",
    filter_laplace_i16_ac4,
    i16,
    AC4,
    nppiFilterLaplace_16s_AC4R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point Laplace filter.",
    filter_laplace_f32_c1,
    f32,
    C1,
    nppiFilterLaplace_32f_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Three channel 32-bit floating-point Laplace filter.",
    filter_laplace_f32_c3,
    f32,
    C3,
    nppiFilterLaplace_32f_C3R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 32-bit floating-point Laplace filter.",
    filter_laplace_f32_c4,
    f32,
    C4,
    nppiFilterLaplace_32f_C4R_Ctx
);

impl_filter_masked_kernel!(
    "Four channel 32-bit floating-point Laplace filter, ignoring alpha channel.",
    filter_laplace_f32_ac4,
    f32,
    AC4,
    nppiFilterLaplace_32f_AC4R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit unsigned to 16-bit signed Laplace filter.",
    filter_laplace_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterLaplace_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit signed to 16-bit signed Laplace filter.",
    filter_laplace_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterLaplace_8s16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 8-bit unsigned Laplace filter with border control.",
    filter_laplace_border_u8_c1,
    u8,
    C1,
    nppiFilterLaplaceBorder_8u_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 8-bit unsigned Laplace filter with border control.",
    filter_laplace_border_u8_c3,
    u8,
    C3,
    nppiFilterLaplaceBorder_8u_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 8-bit unsigned Laplace filter with border control.",
    filter_laplace_border_u8_c4,
    u8,
    C4,
    nppiFilterLaplaceBorder_8u_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 8-bit unsigned Laplace filter with border control, ignoring alpha channel.",
    filter_laplace_border_u8_ac4,
    u8,
    AC4,
    nppiFilterLaplaceBorder_8u_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 16-bit signed Laplace filter with border control.",
    filter_laplace_border_i16_c1,
    i16,
    C1,
    nppiFilterLaplaceBorder_16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 16-bit signed Laplace filter with border control.",
    filter_laplace_border_i16_c3,
    i16,
    C3,
    nppiFilterLaplaceBorder_16s_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 16-bit signed Laplace filter with border control.",
    filter_laplace_border_i16_c4,
    i16,
    C4,
    nppiFilterLaplaceBorder_16s_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 16-bit signed Laplace filter with border control, ignoring alpha channel.",
    filter_laplace_border_i16_ac4,
    i16,
    AC4,
    nppiFilterLaplaceBorder_16s_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point Laplace filter with border control.",
    filter_laplace_border_f32_c1,
    f32,
    C1,
    nppiFilterLaplaceBorder_32f_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Three channel 32-bit floating-point Laplace filter with border control.",
    filter_laplace_border_f32_c3,
    f32,
    C3,
    nppiFilterLaplaceBorder_32f_C3R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 32-bit floating-point Laplace filter with border control.",
    filter_laplace_border_f32_c4,
    f32,
    C4,
    nppiFilterLaplaceBorder_32f_C4R_Ctx
);

impl_filter_masked_kernel_border!(
    "Four channel 32-bit floating-point Laplace filter with border control, ignoring alpha channel.",
    filter_laplace_border_f32_ac4,
    f32,
    AC4,
    nppiFilterLaplaceBorder_32f_AC4R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed Laplace filter with border control.",
    filter_laplace_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterLaplaceBorder_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit signed to 16-bit signed Laplace filter with border control.",
    filter_laplace_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterLaplaceBorder_8s16s_C1R_Ctx
);

/// Dispatches the Laplace filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterLaplace<L: ChannelLayout>: DataTypeLike {
    /// Applies the Laplace filter.
    /// For AC4 layouts, destination alpha is not processed.
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
    unsafe fn filter_laplace(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

macro_rules! impl_filter_laplace_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterLaplace<$layout> for $pixel_ty {
            unsafe fn filter_laplace(
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

impl_filter_laplace_dispatch!(u8, C1, filter_laplace_u8_c1);

impl_filter_laplace_dispatch!(u8, C3, filter_laplace_u8_c3);

impl_filter_laplace_dispatch!(u8, C4, filter_laplace_u8_c4);

impl_filter_laplace_dispatch!(u8, AC4, filter_laplace_u8_ac4);

impl_filter_laplace_dispatch!(i16, C1, filter_laplace_i16_c1);

impl_filter_laplace_dispatch!(i16, C3, filter_laplace_i16_c3);

impl_filter_laplace_dispatch!(i16, C4, filter_laplace_i16_c4);

impl_filter_laplace_dispatch!(i16, AC4, filter_laplace_i16_ac4);

impl_filter_laplace_dispatch!(f32, C1, filter_laplace_f32_c1);

impl_filter_laplace_dispatch!(f32, C3, filter_laplace_f32_c3);

impl_filter_laplace_dispatch!(f32, C4, filter_laplace_f32_c4);

impl_filter_laplace_dispatch!(f32, AC4, filter_laplace_f32_ac4);

/// Applies the Laplace filter.
/// For AC4 layouts, destination alpha is not processed.
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
pub unsafe fn filter_laplace<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: MaskSize,
) -> Result<()>
where
    T: FilterLaplace<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_laplace(stream_context, source, destination, mask_size) }
}

/// Dispatches the Laplace filter.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterLaplaceC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the Laplace filter.
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
    unsafe fn filter_laplace_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterLaplaceC1,
    filter_laplace_typed,
    u8,
    i16,
    filter_laplace_u8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterLaplaceC1,
    filter_laplace_typed,
    i8,
    i16,
    filter_laplace_i8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterLaplaceC1,
    filter_laplace_typed,
    f32,
    f32,
    filter_laplace_f32_c1
);

/// Applies the Laplace filter.
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
pub unsafe fn filter_laplace_typed<T: FilterLaplaceC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_laplace_typed(stream_context, source, destination, mask_size) }
}

/// Dispatches the Laplace filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterLaplaceBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the Laplace filter with border control.
    /// For AC4 layouts, destination alpha is not processed.
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
    unsafe fn filter_laplace_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_laplace_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterLaplaceBorder<$layout> for $pixel_ty {
            unsafe fn filter_laplace_border(
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

impl_filter_laplace_border_dispatch!(u8, C1, filter_laplace_border_u8_c1);

impl_filter_laplace_border_dispatch!(u8, C3, filter_laplace_border_u8_c3);

impl_filter_laplace_border_dispatch!(u8, C4, filter_laplace_border_u8_c4);

impl_filter_laplace_border_dispatch!(u8, AC4, filter_laplace_border_u8_ac4);

impl_filter_laplace_border_dispatch!(i16, C1, filter_laplace_border_i16_c1);

impl_filter_laplace_border_dispatch!(i16, C3, filter_laplace_border_i16_c3);

impl_filter_laplace_border_dispatch!(i16, C4, filter_laplace_border_i16_c4);

impl_filter_laplace_border_dispatch!(i16, AC4, filter_laplace_border_i16_ac4);

impl_filter_laplace_border_dispatch!(f32, C1, filter_laplace_border_f32_c1);

impl_filter_laplace_border_dispatch!(f32, C3, filter_laplace_border_f32_c3);

impl_filter_laplace_border_dispatch!(f32, C4, filter_laplace_border_f32_c4);

impl_filter_laplace_border_dispatch!(f32, AC4, filter_laplace_border_f32_ac4);

/// Applies the Laplace filter with border control.
/// For AC4 layouts, destination alpha is not processed.
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
pub unsafe fn filter_laplace_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterLaplaceBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_laplace_border(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}

/// Dispatches the Laplace filter with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterLaplaceBorderC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the Laplace filter with border control.
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
    unsafe fn filter_laplace_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterLaplaceBorderC1,
    filter_laplace_border_typed,
    u8,
    i16,
    filter_laplace_border_u8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterLaplaceBorderC1,
    filter_laplace_border_typed,
    i8,
    i16,
    filter_laplace_border_i8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterLaplaceBorderC1,
    filter_laplace_border_typed,
    f32,
    f32,
    filter_laplace_border_f32_c1
);

/// Applies the Laplace filter with border control.
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
pub unsafe fn filter_laplace_border_typed<T: FilterLaplaceBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_laplace_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
