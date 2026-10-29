use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering::directional::sobel::primary::raw::{
            filter_sobel_horizontal_f32_ac4, filter_sobel_horizontal_f32_c1,
            filter_sobel_horizontal_f32_c3, filter_sobel_horizontal_f32_c4,
            filter_sobel_horizontal_i16_ac4, filter_sobel_horizontal_i16_c1,
            filter_sobel_horizontal_i16_c3, filter_sobel_horizontal_i16_c4,
            filter_sobel_horizontal_u8_ac4, filter_sobel_horizontal_u8_c1,
            filter_sobel_horizontal_u8_c3, filter_sobel_horizontal_u8_c4,
            filter_sobel_vertical_f32_ac4, filter_sobel_vertical_f32_c1,
            filter_sobel_vertical_f32_c3, filter_sobel_vertical_f32_c4,
            filter_sobel_vertical_i16_ac4, filter_sobel_vertical_i16_c1,
            filter_sobel_vertical_i16_c3, filter_sobel_vertical_i16_c4,
            filter_sobel_vertical_u8_ac4, filter_sobel_vertical_u8_c1, filter_sobel_vertical_u8_c3,
            filter_sobel_vertical_u8_c4,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    types::DataTypeLike,
};

/// Dispatches the horizontal Sobel filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelHorizontal<L: ChannelLayout>: DataTypeLike {
    /// Applies the horizontal Sobel filter.
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
    unsafe fn filter_sobel_horizontal(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    C1,
    filter_sobel_horizontal_u8_c1
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    C3,
    filter_sobel_horizontal_u8_c3
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    C4,
    filter_sobel_horizontal_u8_c4
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    AC4,
    filter_sobel_horizontal_u8_ac4
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    C1,
    filter_sobel_horizontal_i16_c1
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    C3,
    filter_sobel_horizontal_i16_c3
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    C4,
    filter_sobel_horizontal_i16_c4
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    AC4,
    filter_sobel_horizontal_i16_ac4
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    C1,
    filter_sobel_horizontal_f32_c1
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    C3,
    filter_sobel_horizontal_f32_c3
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    C4,
    filter_sobel_horizontal_f32_c4
);

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    AC4,
    filter_sobel_horizontal_f32_ac4
);

/// Applies the horizontal Sobel filter.
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
pub unsafe fn filter_sobel_horizontal<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterSobelHorizontal<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sobel_horizontal(stream_context, source, destination) }
}

/// Dispatches the vertical Sobel filter.
/// For AC4 layouts, destination alpha is not processed.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelVertical<L: ChannelLayout>: DataTypeLike {
    /// Applies the vertical Sobel filter.
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
    unsafe fn filter_sobel_vertical(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    C1,
    filter_sobel_vertical_u8_c1
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    C3,
    filter_sobel_vertical_u8_c3
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    C4,
    filter_sobel_vertical_u8_c4
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    AC4,
    filter_sobel_vertical_u8_ac4
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    C1,
    filter_sobel_vertical_i16_c1
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    C3,
    filter_sobel_vertical_i16_c3
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    C4,
    filter_sobel_vertical_i16_c4
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    AC4,
    filter_sobel_vertical_i16_ac4
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    C1,
    filter_sobel_vertical_f32_c1
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    C3,
    filter_sobel_vertical_f32_c3
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    C4,
    filter_sobel_vertical_f32_c4
);

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    AC4,
    filter_sobel_vertical_f32_ac4
);

/// Applies the vertical Sobel filter.
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
pub unsafe fn filter_sobel_vertical<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: FilterSobelVertical<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sobel_vertical(stream_context, source, destination) }
}
