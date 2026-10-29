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
    "Single channel 8-bit unsigned horizontal Prewitt filter.",
    filter_prewitt_horizontal_u8_c1,
    u8,
    C1,
    nppiFilterPrewittHoriz_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned horizontal Prewitt filter.",
    filter_prewitt_horizontal_u8_c3,
    u8,
    C3,
    nppiFilterPrewittHoriz_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned horizontal Prewitt filter.",
    filter_prewitt_horizontal_u8_c4,
    u8,
    C4,
    nppiFilterPrewittHoriz_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned horizontal Prewitt filter, ignoring alpha channel.",
    filter_prewitt_horizontal_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittHoriz_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed horizontal Prewitt filter.",
    filter_prewitt_horizontal_i16_c1,
    i16,
    C1,
    nppiFilterPrewittHoriz_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed horizontal Prewitt filter.",
    filter_prewitt_horizontal_i16_c3,
    i16,
    C3,
    nppiFilterPrewittHoriz_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed horizontal Prewitt filter.",
    filter_prewitt_horizontal_i16_c4,
    i16,
    C4,
    nppiFilterPrewittHoriz_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed horizontal Prewitt filter, ignoring alpha channel.",
    filter_prewitt_horizontal_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittHoriz_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point horizontal Prewitt filter.",
    filter_prewitt_horizontal_f32_c1,
    f32,
    C1,
    nppiFilterPrewittHoriz_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point horizontal Prewitt filter.",
    filter_prewitt_horizontal_f32_c3,
    f32,
    C3,
    nppiFilterPrewittHoriz_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point horizontal Prewitt filter.",
    filter_prewitt_horizontal_f32_c4,
    f32,
    C4,
    nppiFilterPrewittHoriz_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point horizontal Prewitt filter, ignoring alpha channel.",
    filter_prewitt_horizontal_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittHoriz_32f_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 8-bit unsigned vertical Prewitt filter.",
    filter_prewitt_vertical_u8_c1,
    u8,
    C1,
    nppiFilterPrewittVert_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned vertical Prewitt filter.",
    filter_prewitt_vertical_u8_c3,
    u8,
    C3,
    nppiFilterPrewittVert_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned vertical Prewitt filter.",
    filter_prewitt_vertical_u8_c4,
    u8,
    C4,
    nppiFilterPrewittVert_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned vertical Prewitt filter, ignoring alpha channel.",
    filter_prewitt_vertical_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittVert_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed vertical Prewitt filter.",
    filter_prewitt_vertical_i16_c1,
    i16,
    C1,
    nppiFilterPrewittVert_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed vertical Prewitt filter.",
    filter_prewitt_vertical_i16_c3,
    i16,
    C3,
    nppiFilterPrewittVert_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed vertical Prewitt filter.",
    filter_prewitt_vertical_i16_c4,
    i16,
    C4,
    nppiFilterPrewittVert_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed vertical Prewitt filter, ignoring alpha channel.",
    filter_prewitt_vertical_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittVert_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point vertical Prewitt filter.",
    filter_prewitt_vertical_f32_c1,
    f32,
    C1,
    nppiFilterPrewittVert_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point vertical Prewitt filter.",
    filter_prewitt_vertical_f32_c3,
    f32,
    C3,
    nppiFilterPrewittVert_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point vertical Prewitt filter.",
    filter_prewitt_vertical_f32_c4,
    f32,
    C4,
    nppiFilterPrewittVert_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point vertical Prewitt filter, ignoring alpha channel.",
    filter_prewitt_vertical_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittVert_32f_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_u8_c1,
    u8,
    C1,
    nppiFilterPrewittHorizBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_u8_c3,
    u8,
    C3,
    nppiFilterPrewittHorizBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_u8_c4,
    u8,
    C4,
    nppiFilterPrewittHorizBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned horizontal Prewitt filter with border control, ignoring alpha channel.",
    filter_prewitt_horizontal_border_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittHorizBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_i16_c1,
    i16,
    C1,
    nppiFilterPrewittHorizBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_i16_c3,
    i16,
    C3,
    nppiFilterPrewittHorizBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_i16_c4,
    i16,
    C4,
    nppiFilterPrewittHorizBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed horizontal Prewitt filter with border control, ignoring alpha channel.",
    filter_prewitt_horizontal_border_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittHorizBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_f32_c1,
    f32,
    C1,
    nppiFilterPrewittHorizBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_f32_c3,
    f32,
    C3,
    nppiFilterPrewittHorizBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point horizontal Prewitt filter with border control.",
    filter_prewitt_horizontal_border_f32_c4,
    f32,
    C4,
    nppiFilterPrewittHorizBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point horizontal Prewitt filter with border control, ignoring alpha channel.",
    filter_prewitt_horizontal_border_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittHorizBorder_32f_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_u8_c1,
    u8,
    C1,
    nppiFilterPrewittVertBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_u8_c3,
    u8,
    C3,
    nppiFilterPrewittVertBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_u8_c4,
    u8,
    C4,
    nppiFilterPrewittVertBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned vertical Prewitt filter with border control, ignoring alpha channel.",
    filter_prewitt_vertical_border_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittVertBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_i16_c1,
    i16,
    C1,
    nppiFilterPrewittVertBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_i16_c3,
    i16,
    C3,
    nppiFilterPrewittVertBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_i16_c4,
    i16,
    C4,
    nppiFilterPrewittVertBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed vertical Prewitt filter with border control, ignoring alpha channel.",
    filter_prewitt_vertical_border_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittVertBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_f32_c1,
    f32,
    C1,
    nppiFilterPrewittVertBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_f32_c3,
    f32,
    C3,
    nppiFilterPrewittVertBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point vertical Prewitt filter with border control.",
    filter_prewitt_vertical_border_f32_c4,
    f32,
    C4,
    nppiFilterPrewittVertBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point vertical Prewitt filter with border control, ignoring alpha channel.",
    filter_prewitt_vertical_border_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittVertBorder_32f_AC4R_Ctx
);

/// Dispatches the horizontal Prewitt filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterPrewittHorizontal<L: ChannelLayout>: DataTypeLike {
    /// Applies the horizontal Prewitt filter.
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
    unsafe fn filter_prewitt_horizontal(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    C1,
    filter_prewitt_horizontal_u8_c1
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    C3,
    filter_prewitt_horizontal_u8_c3
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    C4,
    filter_prewitt_horizontal_u8_c4
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    AC4,
    filter_prewitt_horizontal_u8_ac4
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    C1,
    filter_prewitt_horizontal_i16_c1
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    C3,
    filter_prewitt_horizontal_i16_c3
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    C4,
    filter_prewitt_horizontal_i16_c4
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    AC4,
    filter_prewitt_horizontal_i16_ac4
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    C1,
    filter_prewitt_horizontal_f32_c1
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    C3,
    filter_prewitt_horizontal_f32_c3
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    C4,
    filter_prewitt_horizontal_f32_c4
);

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    AC4,
    filter_prewitt_horizontal_f32_ac4
);

/// Applies the horizontal Prewitt filter.
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
pub unsafe fn filter_prewitt_horizontal<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterPrewittHorizontal<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_prewitt_horizontal(stream_context, source, destination) }
}

/// Dispatches the vertical Prewitt filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterPrewittVertical<L: ChannelLayout>: DataTypeLike {
    /// Applies the vertical Prewitt filter.
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
    unsafe fn filter_prewitt_vertical(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    C1,
    filter_prewitt_vertical_u8_c1
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    C3,
    filter_prewitt_vertical_u8_c3
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    C4,
    filter_prewitt_vertical_u8_c4
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    AC4,
    filter_prewitt_vertical_u8_ac4
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    C1,
    filter_prewitt_vertical_i16_c1
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    C3,
    filter_prewitt_vertical_i16_c3
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    C4,
    filter_prewitt_vertical_i16_c4
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    AC4,
    filter_prewitt_vertical_i16_ac4
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    C1,
    filter_prewitt_vertical_f32_c1
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    C3,
    filter_prewitt_vertical_f32_c3
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    C4,
    filter_prewitt_vertical_f32_c4
);

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    AC4,
    filter_prewitt_vertical_f32_ac4
);

/// Applies the vertical Prewitt filter.
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
pub unsafe fn filter_prewitt_vertical<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterPrewittVertical<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_prewitt_vertical(stream_context, source, destination) }
}

/// Dispatches the horizontal Prewitt filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterPrewittHorizontalBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the horizontal Prewitt filter with border control.
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
    unsafe fn filter_prewitt_horizontal_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    C1,
    filter_prewitt_horizontal_border_u8_c1
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    C3,
    filter_prewitt_horizontal_border_u8_c3
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    C4,
    filter_prewitt_horizontal_border_u8_c4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    AC4,
    filter_prewitt_horizontal_border_u8_ac4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    C1,
    filter_prewitt_horizontal_border_i16_c1
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    C3,
    filter_prewitt_horizontal_border_i16_c3
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    C4,
    filter_prewitt_horizontal_border_i16_c4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    AC4,
    filter_prewitt_horizontal_border_i16_ac4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    C1,
    filter_prewitt_horizontal_border_f32_c1
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    C3,
    filter_prewitt_horizontal_border_f32_c3
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    C4,
    filter_prewitt_horizontal_border_f32_c4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    AC4,
    filter_prewitt_horizontal_border_f32_ac4
);

/// Applies the horizontal Prewitt filter with border control.
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
pub unsafe fn filter_prewitt_horizontal_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterPrewittHorizontalBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_prewitt_horizontal_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

/// Dispatches the vertical Prewitt filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterPrewittVerticalBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the vertical Prewitt filter with border control.
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
    unsafe fn filter_prewitt_vertical_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    C1,
    filter_prewitt_vertical_border_u8_c1
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    C3,
    filter_prewitt_vertical_border_u8_c3
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    C4,
    filter_prewitt_vertical_border_u8_c4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    AC4,
    filter_prewitt_vertical_border_u8_ac4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    C1,
    filter_prewitt_vertical_border_i16_c1
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    C3,
    filter_prewitt_vertical_border_i16_c3
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    C4,
    filter_prewitt_vertical_border_i16_c4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    AC4,
    filter_prewitt_vertical_border_i16_ac4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    C1,
    filter_prewitt_vertical_border_f32_c1
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    C3,
    filter_prewitt_vertical_border_f32_c3
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    C4,
    filter_prewitt_vertical_border_f32_c4
);

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    AC4,
    filter_prewitt_vertical_border_f32_ac4
);

/// Applies the vertical Prewitt filter with border control.
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
pub unsafe fn filter_prewitt_vertical_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterPrewittVerticalBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_prewitt_vertical_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
