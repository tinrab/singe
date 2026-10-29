use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering::directional::sobel::primary::raw::{
            filter_sobel_horizontal_border_f32_ac4, filter_sobel_horizontal_border_f32_c1,
            filter_sobel_horizontal_border_f32_c3, filter_sobel_horizontal_border_f32_c4,
            filter_sobel_horizontal_border_i16_ac4, filter_sobel_horizontal_border_i16_c1,
            filter_sobel_horizontal_border_i16_c3, filter_sobel_horizontal_border_i16_c4,
            filter_sobel_horizontal_border_u8_ac4, filter_sobel_horizontal_border_u8_c1,
            filter_sobel_horizontal_border_u8_c3, filter_sobel_horizontal_border_u8_c4,
            filter_sobel_vertical_border_f32_ac4, filter_sobel_vertical_border_f32_c1,
            filter_sobel_vertical_border_f32_c3, filter_sobel_vertical_border_f32_c4,
            filter_sobel_vertical_border_i16_ac4, filter_sobel_vertical_border_i16_c1,
            filter_sobel_vertical_border_i16_c3, filter_sobel_vertical_border_i16_c4,
            filter_sobel_vertical_border_u8_ac4, filter_sobel_vertical_border_u8_c1,
            filter_sobel_vertical_border_u8_c3, filter_sobel_vertical_border_u8_c4,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    types::{BorderType, DataTypeLike, Point},
};

/// Dispatches the horizontal Sobel filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelHorizontalBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the horizontal Sobel filter with border control.
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
    unsafe fn filter_sobel_horizontal_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    C1,
    filter_sobel_horizontal_border_u8_c1
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    C3,
    filter_sobel_horizontal_border_u8_c3
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    C4,
    filter_sobel_horizontal_border_u8_c4
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    AC4,
    filter_sobel_horizontal_border_u8_ac4
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    C1,
    filter_sobel_horizontal_border_i16_c1
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    C3,
    filter_sobel_horizontal_border_i16_c3
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    C4,
    filter_sobel_horizontal_border_i16_c4
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    AC4,
    filter_sobel_horizontal_border_i16_ac4
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    C1,
    filter_sobel_horizontal_border_f32_c1
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    C3,
    filter_sobel_horizontal_border_f32_c3
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    C4,
    filter_sobel_horizontal_border_f32_c4
);

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    AC4,
    filter_sobel_horizontal_border_f32_ac4
);

/// Applies the horizontal Sobel filter with border control.
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
pub unsafe fn filter_sobel_horizontal_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterSobelHorizontalBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_horizontal_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

/// Dispatches the vertical Sobel filter with border control.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelVerticalBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies the vertical Sobel filter with border control.
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
    unsafe fn filter_sobel_vertical_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    C1,
    filter_sobel_vertical_border_u8_c1
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    C3,
    filter_sobel_vertical_border_u8_c3
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    C4,
    filter_sobel_vertical_border_u8_c4
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    AC4,
    filter_sobel_vertical_border_u8_ac4
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    C1,
    filter_sobel_vertical_border_i16_c1
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    C3,
    filter_sobel_vertical_border_i16_c3
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    C4,
    filter_sobel_vertical_border_i16_c4
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    AC4,
    filter_sobel_vertical_border_i16_ac4
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    C1,
    filter_sobel_vertical_border_f32_c1
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    C3,
    filter_sobel_vertical_border_f32_c3
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    C4,
    filter_sobel_vertical_border_f32_c4
);

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    AC4,
    filter_sobel_vertical_border_f32_ac4
);

/// Applies the vertical Sobel filter with border control.
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
pub unsafe fn filter_sobel_vertical_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterSobelVerticalBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_vertical_border(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
