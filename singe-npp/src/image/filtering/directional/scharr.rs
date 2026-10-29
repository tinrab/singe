use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_same_size},
        view::{C1, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
};

impl_filter_directional_typed!(
    "Single channel 8-bit unsigned to 16-bit signed horizontal Scharr filter.",
    filter_scharr_horizontal_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrHoriz_8u16s_C1R_Ctx
);

impl_filter_directional_typed!(
    "Single channel 8-bit signed to 16-bit signed horizontal Scharr filter.",
    filter_scharr_horizontal_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrHoriz_8s16s_C1R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point horizontal Scharr filter.",
    filter_scharr_horizontal_f32_c1,
    f32,
    C1,
    nppiFilterScharrHoriz_32f_C1R_Ctx
);

impl_filter_directional_typed!(
    "Single channel 8-bit unsigned to 16-bit signed vertical Scharr filter.",
    filter_scharr_vertical_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrVert_8u16s_C1R_Ctx
);

impl_filter_directional_typed!(
    "Single channel 8-bit signed to 16-bit signed vertical Scharr filter.",
    filter_scharr_vertical_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrVert_8s16s_C1R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point vertical Scharr filter.",
    filter_scharr_vertical_f32_c1,
    f32,
    C1,
    nppiFilterScharrVert_32f_C1R_Ctx
);

impl_filter_directional_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed horizontal Scharr filter kernel with border control.",
    filter_scharr_horizontal_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrHorizBorder_8u16s_C1R_Ctx
);

impl_filter_directional_typed_border!(
    "Single channel 8-bit signed to 16-bit signed horizontal Scharr filter kernel with border control.",
    filter_scharr_horizontal_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrHorizBorder_8s16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point horizontal Scharr filter kernel with border control.",
    filter_scharr_horizontal_border_f32_c1,
    f32,
    C1,
    nppiFilterScharrHorizBorder_32f_C1R_Ctx
);

impl_filter_directional_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed vertical Scharr filter kernel with border control.",
    filter_scharr_vertical_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrVertBorder_8u16s_C1R_Ctx
);

impl_filter_directional_typed_border!(
    "Single channel 8-bit signed to 16-bit signed vertical Scharr filter kernel with border control.",
    filter_scharr_vertical_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrVertBorder_8s16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point vertical Scharr filter kernel with border control.",
    filter_scharr_vertical_border_f32_c1,
    f32,
    C1,
    nppiFilterScharrVertBorder_32f_C1R_Ctx
);

/// Dispatches the horizontal Scharr filter.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterScharrHorizontalC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the horizontal Scharr filter.
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
    unsafe fn filter_scharr_horizontal_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
    ) -> Result<()>;
}

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrHorizontalC1,
    filter_scharr_horizontal_typed,
    u8,
    i16,
    filter_scharr_horizontal_u8_to_i16_c1
);

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrHorizontalC1,
    filter_scharr_horizontal_typed,
    i8,
    i16,
    filter_scharr_horizontal_i8_to_i16_c1
);

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrHorizontalC1,
    filter_scharr_horizontal_typed,
    f32,
    f32,
    filter_scharr_horizontal_f32_c1
);

/// Applies the horizontal Scharr filter.
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
pub unsafe fn filter_scharr_horizontal_typed<T: FilterScharrHorizontalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_scharr_horizontal_typed(stream_context, source, destination) }
}

/// Dispatches the vertical Scharr filter.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterScharrVerticalC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the vertical Scharr filter.
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
    unsafe fn filter_scharr_vertical_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
    ) -> Result<()>;
}

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrVerticalC1,
    filter_scharr_vertical_typed,
    u8,
    i16,
    filter_scharr_vertical_u8_to_i16_c1
);

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrVerticalC1,
    filter_scharr_vertical_typed,
    i8,
    i16,
    filter_scharr_vertical_i8_to_i16_c1
);

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrVerticalC1,
    filter_scharr_vertical_typed,
    f32,
    f32,
    filter_scharr_vertical_f32_c1
);

/// Applies the vertical Scharr filter.
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
pub unsafe fn filter_scharr_vertical_typed<T: FilterScharrVerticalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_scharr_vertical_typed(stream_context, source, destination) }
}

/// Dispatches the horizontal Scharr filter with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterScharrHorizontalBorderC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the horizontal Scharr filter with border control.
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
    unsafe fn filter_scharr_horizontal_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrHorizontalBorderC1,
    filter_scharr_horizontal_border_typed,
    u8,
    i16,
    filter_scharr_horizontal_border_u8_to_i16_c1
);

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrHorizontalBorderC1,
    filter_scharr_horizontal_border_typed,
    i8,
    i16,
    filter_scharr_horizontal_border_i8_to_i16_c1
);

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrHorizontalBorderC1,
    filter_scharr_horizontal_border_typed,
    f32,
    f32,
    filter_scharr_horizontal_border_f32_c1
);

/// Applies the horizontal Scharr filter with border control.
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
pub unsafe fn filter_scharr_horizontal_border_typed<T: FilterScharrHorizontalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_scharr_horizontal_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

/// Dispatches the vertical Scharr filter with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterScharrVerticalBorderC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the vertical Scharr filter with border control.
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
    unsafe fn filter_scharr_vertical_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrVerticalBorderC1,
    filter_scharr_vertical_border_typed,
    u8,
    i16,
    filter_scharr_vertical_border_u8_to_i16_c1
);

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrVerticalBorderC1,
    filter_scharr_vertical_border_typed,
    i8,
    i16,
    filter_scharr_vertical_border_i8_to_i16_c1
);

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrVerticalBorderC1,
    filter_scharr_vertical_border_typed,
    f32,
    f32,
    filter_scharr_vertical_border_f32_c1
);

/// Applies the vertical Scharr filter with border control.
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
pub unsafe fn filter_scharr_vertical_border_typed<T: FilterScharrVerticalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_scharr_vertical_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
