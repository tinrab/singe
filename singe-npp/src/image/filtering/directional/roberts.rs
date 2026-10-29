use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_same_size},
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
};

impl_filter_directional!(
    "Single channel 8-bit unsigned horizontal Roberts filter.",
    filter_roberts_down_u8_c1,
    u8,
    C1,
    nppiFilterRobertsDown_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned horizontal Roberts filter.",
    filter_roberts_down_u8_c3,
    u8,
    C3,
    nppiFilterRobertsDown_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned horizontal Roberts filter.",
    filter_roberts_down_u8_c4,
    u8,
    C4,
    nppiFilterRobertsDown_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned horizontal Roberts filter, ignoring alpha-channel.",
    filter_roberts_down_u8_ac4,
    u8,
    AC4,
    nppiFilterRobertsDown_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed horizontal Roberts filter.",
    filter_roberts_down_i16_c1,
    i16,
    C1,
    nppiFilterRobertsDown_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed horizontal Roberts filter.",
    filter_roberts_down_i16_c3,
    i16,
    C3,
    nppiFilterRobertsDown_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed horizontal Roberts filter.",
    filter_roberts_down_i16_c4,
    i16,
    C4,
    nppiFilterRobertsDown_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed horizontal Roberts filter, ignoring alpha-channel.",
    filter_roberts_down_i16_ac4,
    i16,
    AC4,
    nppiFilterRobertsDown_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point horizontal Roberts filter.",
    filter_roberts_down_f32_c1,
    f32,
    C1,
    nppiFilterRobertsDown_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point horizontal Roberts filter.",
    filter_roberts_down_f32_c3,
    f32,
    C3,
    nppiFilterRobertsDown_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point horizontal Roberts filter.",
    filter_roberts_down_f32_c4,
    f32,
    C4,
    nppiFilterRobertsDown_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point horizontal Roberts filter, ignoring alpha-channel.",
    filter_roberts_down_f32_ac4,
    f32,
    AC4,
    nppiFilterRobertsDown_32f_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 8-bit unsigned vertical Roberts filter.",
    filter_roberts_up_u8_c1,
    u8,
    C1,
    nppiFilterRobertsUp_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned vertical Roberts filter.",
    filter_roberts_up_u8_c3,
    u8,
    C3,
    nppiFilterRobertsUp_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned vertical Roberts filter.",
    filter_roberts_up_u8_c4,
    u8,
    C4,
    nppiFilterRobertsUp_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned vertical Roberts filter, ignoring alpha-channel.",
    filter_roberts_up_u8_ac4,
    u8,
    AC4,
    nppiFilterRobertsUp_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed vertical Roberts filter.",
    filter_roberts_up_i16_c1,
    i16,
    C1,
    nppiFilterRobertsUp_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed vertical Roberts filter.",
    filter_roberts_up_i16_c3,
    i16,
    C3,
    nppiFilterRobertsUp_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed vertical Roberts filter.",
    filter_roberts_up_i16_c4,
    i16,
    C4,
    nppiFilterRobertsUp_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed vertical Roberts filter, ignoring alpha-channel.",
    filter_roberts_up_i16_ac4,
    i16,
    AC4,
    nppiFilterRobertsUp_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point vertical Roberts filter.",
    filter_roberts_up_f32_c1,
    f32,
    C1,
    nppiFilterRobertsUp_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point vertical Roberts filter.",
    filter_roberts_up_f32_c3,
    f32,
    C3,
    nppiFilterRobertsUp_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point vertical Roberts filter.",
    filter_roberts_up_f32_c4,
    f32,
    C4,
    nppiFilterRobertsUp_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point vertical Roberts filter, ignoring alpha-channel.",
    filter_roberts_up_f32_ac4,
    f32,
    AC4,
    nppiFilterRobertsUp_32f_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned horizontal Roberts filter with border control.",
    filter_roberts_down_border_u8_c1,
    u8,
    C1,
    nppiFilterRobertsDownBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned horizontal Roberts filter with border control.",
    filter_roberts_down_border_u8_c3,
    u8,
    C3,
    nppiFilterRobertsDownBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned horizontal Roberts filter with border control.",
    filter_roberts_down_border_u8_c4,
    u8,
    C4,
    nppiFilterRobertsDownBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned horizontal Roberts filter with border control, ignoring alpha-channel.",
    filter_roberts_down_border_u8_ac4,
    u8,
    AC4,
    nppiFilterRobertsDownBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed horizontal Roberts filter with border control.",
    filter_roberts_down_border_i16_c1,
    i16,
    C1,
    nppiFilterRobertsDownBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed horizontal Roberts filter with border control.",
    filter_roberts_down_border_i16_c3,
    i16,
    C3,
    nppiFilterRobertsDownBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed horizontal Roberts filter with border control.",
    filter_roberts_down_border_i16_c4,
    i16,
    C4,
    nppiFilterRobertsDownBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed horizontal Roberts filter with border control, ignoring alpha-channel.",
    filter_roberts_down_border_i16_ac4,
    i16,
    AC4,
    nppiFilterRobertsDownBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point horizontal Roberts filter with border control.",
    filter_roberts_down_border_f32_c1,
    f32,
    C1,
    nppiFilterRobertsDownBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point horizontal Roberts filter with border control.",
    filter_roberts_down_border_f32_c3,
    f32,
    C3,
    nppiFilterRobertsDownBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point horizontal Roberts filter with border control.",
    filter_roberts_down_border_f32_c4,
    f32,
    C4,
    nppiFilterRobertsDownBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point horizontal Roberts filter with border control, ignoring alpha-channel.",
    filter_roberts_down_border_f32_ac4,
    f32,
    AC4,
    nppiFilterRobertsDownBorder_32f_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned vertical Roberts filter with border control.",
    filter_roberts_up_border_u8_c1,
    u8,
    C1,
    nppiFilterRobertsUpBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned vertical Roberts filter with border control.",
    filter_roberts_up_border_u8_c3,
    u8,
    C3,
    nppiFilterRobertsUpBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned vertical Roberts filter with border control.",
    filter_roberts_up_border_u8_c4,
    u8,
    C4,
    nppiFilterRobertsUpBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned vertical Roberts filter with border control, ignoring alpha-channel.",
    filter_roberts_up_border_u8_ac4,
    u8,
    AC4,
    nppiFilterRobertsUpBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed vertical Roberts filter with border control.",
    filter_roberts_up_border_i16_c1,
    i16,
    C1,
    nppiFilterRobertsUpBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed vertical Roberts filter with border control.",
    filter_roberts_up_border_i16_c3,
    i16,
    C3,
    nppiFilterRobertsUpBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed vertical Roberts filter with border control.",
    filter_roberts_up_border_i16_c4,
    i16,
    C4,
    nppiFilterRobertsUpBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed vertical Roberts filter with border control, ignoring alpha-channel.",
    filter_roberts_up_border_i16_ac4,
    i16,
    AC4,
    nppiFilterRobertsUpBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point vertical Roberts filter with border control.",
    filter_roberts_up_border_f32_c1,
    f32,
    C1,
    nppiFilterRobertsUpBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point vertical Roberts filter with border control.",
    filter_roberts_up_border_f32_c3,
    f32,
    C3,
    nppiFilterRobertsUpBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point vertical Roberts filter with border control.",
    filter_roberts_up_border_f32_c4,
    f32,
    C4,
    nppiFilterRobertsUpBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point vertical Roberts filter with border control, ignoring alpha-channel.",
    filter_roberts_up_border_f32_ac4,
    f32,
    AC4,
    nppiFilterRobertsUpBorder_32f_AC4R_Ctx
);

/// Dispatches the downward Roberts filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterRobertsDown<L: ChannelLayout>: DataTypeLike {
    /// Applies the downward Roberts filter.
    /// For AC4 layouts, destination alpha is not processed.
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
    unsafe fn filter_roberts_down(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    u8,
    C1,
    filter_roberts_down_u8_c1
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    u8,
    C3,
    filter_roberts_down_u8_c3
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    u8,
    C4,
    filter_roberts_down_u8_c4
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    u8,
    AC4,
    filter_roberts_down_u8_ac4
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    i16,
    C1,
    filter_roberts_down_i16_c1
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    i16,
    C3,
    filter_roberts_down_i16_c3
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    i16,
    C4,
    filter_roberts_down_i16_c4
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    i16,
    AC4,
    filter_roberts_down_i16_ac4
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    f32,
    C1,
    filter_roberts_down_f32_c1
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    f32,
    C3,
    filter_roberts_down_f32_c3
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    f32,
    C4,
    filter_roberts_down_f32_c4
);

impl_directional_filter_dispatch!(
    FilterRobertsDown,
    filter_roberts_down,
    f32,
    AC4,
    filter_roberts_down_f32_ac4
);

/// Applies the downward Roberts filter.
/// For AC4 layouts, destination alpha is not processed.
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
pub unsafe fn filter_roberts_down<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterRobertsDown<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_roberts_down(stream_context, source, destination) }
}

/// Dispatches the upward Roberts filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterRobertsUp<L: ChannelLayout>: DataTypeLike {
    /// Applies the upward Roberts filter.
    /// For AC4 layouts, destination alpha is not processed.
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
    unsafe fn filter_roberts_up(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    u8,
    C1,
    filter_roberts_up_u8_c1
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    u8,
    C3,
    filter_roberts_up_u8_c3
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    u8,
    C4,
    filter_roberts_up_u8_c4
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    u8,
    AC4,
    filter_roberts_up_u8_ac4
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    i16,
    C1,
    filter_roberts_up_i16_c1
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    i16,
    C3,
    filter_roberts_up_i16_c3
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    i16,
    C4,
    filter_roberts_up_i16_c4
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    i16,
    AC4,
    filter_roberts_up_i16_ac4
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    f32,
    C1,
    filter_roberts_up_f32_c1
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    f32,
    C3,
    filter_roberts_up_f32_c3
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    f32,
    C4,
    filter_roberts_up_f32_c4
);

impl_directional_filter_dispatch!(
    FilterRobertsUp,
    filter_roberts_up,
    f32,
    AC4,
    filter_roberts_up_f32_ac4
);

/// Applies the upward Roberts filter.
/// For AC4 layouts, destination alpha is not processed.
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
pub unsafe fn filter_roberts_up<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterRobertsUp<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_roberts_up(stream_context, source, destination) }
}

/// Dispatches the downward Roberts filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterRobertsDownBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the downward Roberts filter with border control.
    /// For AC4 layouts, destination alpha is not processed.
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
    unsafe fn filter_roberts_down_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    u8,
    C1,
    filter_roberts_down_border_u8_c1
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    u8,
    C3,
    filter_roberts_down_border_u8_c3
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    u8,
    C4,
    filter_roberts_down_border_u8_c4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    u8,
    AC4,
    filter_roberts_down_border_u8_ac4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    i16,
    C1,
    filter_roberts_down_border_i16_c1
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    i16,
    C3,
    filter_roberts_down_border_i16_c3
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    i16,
    C4,
    filter_roberts_down_border_i16_c4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    i16,
    AC4,
    filter_roberts_down_border_i16_ac4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    f32,
    C1,
    filter_roberts_down_border_f32_c1
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    f32,
    C3,
    filter_roberts_down_border_f32_c3
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    f32,
    C4,
    filter_roberts_down_border_f32_c4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsDownBorder,
    filter_roberts_down_border,
    f32,
    AC4,
    filter_roberts_down_border_f32_ac4
);

/// Applies the downward Roberts filter with border control.
/// For AC4 layouts, destination alpha is not processed.
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
pub unsafe fn filter_roberts_down_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterRobertsDownBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_roberts_down_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

/// Dispatches the upward Roberts filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterRobertsUpBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the upward Roberts filter with border control.
    /// For AC4 layouts, destination alpha is not processed.
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
    unsafe fn filter_roberts_up_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    u8,
    C1,
    filter_roberts_up_border_u8_c1
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    u8,
    C3,
    filter_roberts_up_border_u8_c3
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    u8,
    C4,
    filter_roberts_up_border_u8_c4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    u8,
    AC4,
    filter_roberts_up_border_u8_ac4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    i16,
    C1,
    filter_roberts_up_border_i16_c1
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    i16,
    C3,
    filter_roberts_up_border_i16_c3
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    i16,
    C4,
    filter_roberts_up_border_i16_c4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    i16,
    AC4,
    filter_roberts_up_border_i16_ac4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    f32,
    C1,
    filter_roberts_up_border_f32_c1
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    f32,
    C3,
    filter_roberts_up_border_f32_c3
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    f32,
    C4,
    filter_roberts_up_border_f32_c4
);

impl_directional_filter_border_dispatch!(
    FilterRobertsUpBorder,
    filter_roberts_up_border,
    f32,
    AC4,
    filter_roberts_up_border_f32_ac4
);

/// Applies the upward Roberts filter with border control.
/// For AC4 layouts, destination alpha is not processed.
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
pub unsafe fn filter_roberts_up_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterRobertsUpBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_roberts_up_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
