use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering::directional::sobel::primary::raw::{
            filter_sobel_horizontal_border_i8_to_i16_c1,
            filter_sobel_horizontal_border_u8_to_i16_c1, filter_sobel_horizontal_i8_to_i16_c1,
            filter_sobel_horizontal_u8_to_i16_c1, filter_sobel_vertical_border_i8_to_i16_c1,
            filter_sobel_vertical_border_u8_to_i16_c1, filter_sobel_vertical_i8_to_i16_c1,
            filter_sobel_vertical_u8_to_i16_c1,
        },
        view::{C1, ImageOutputView, ImageView},
    },
    types::{BorderType, DataTypeLike, MaskSize, Point},
};

/// Dispatches the horizontal Sobel filter.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelHorizontalC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the horizontal Sobel filter.
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
    unsafe fn filter_sobel_horizontal_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalC1,
    filter_sobel_horizontal_typed,
    u8,
    i16,
    filter_sobel_horizontal_u8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalC1,
    filter_sobel_horizontal_typed,
    i8,
    i16,
    filter_sobel_horizontal_i8_to_i16_c1
);

/// Applies the horizontal Sobel filter.
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
pub unsafe fn filter_sobel_horizontal_typed<T: FilterSobelHorizontalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sobel_horizontal_typed(stream_context, source, destination, mask_size) }
}

/// Dispatches the vertical Sobel filter.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelVerticalC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the vertical Sobel filter.
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
    unsafe fn filter_sobel_vertical_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalC1,
    filter_sobel_vertical_typed,
    u8,
    i16,
    filter_sobel_vertical_u8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalC1,
    filter_sobel_vertical_typed,
    i8,
    i16,
    filter_sobel_vertical_i8_to_i16_c1
);

/// Applies the vertical Sobel filter.
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
pub unsafe fn filter_sobel_vertical_typed<T: FilterSobelVerticalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sobel_vertical_typed(stream_context, source, destination, mask_size) }
}

/// Dispatches the horizontal Sobel filter with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelHorizontalBorderC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the horizontal Sobel filter with border control.
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
    unsafe fn filter_sobel_horizontal_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalBorderC1,
    filter_sobel_horizontal_border_typed,
    u8,
    i16,
    filter_sobel_horizontal_border_u8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalBorderC1,
    filter_sobel_horizontal_border_typed,
    i8,
    i16,
    filter_sobel_horizontal_border_i8_to_i16_c1
);

/// Applies the horizontal Sobel filter with border control.
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
pub unsafe fn filter_sobel_horizontal_border_typed<T: FilterSobelHorizontalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_horizontal_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}

/// Dispatches the vertical Sobel filter with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, filter orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelVerticalBorderC1: DataTypeLike {
    /// Element type of the filtered output samples.
    type Output: DataTypeLike;

    /// Applies the vertical Sobel filter with border control.
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
    unsafe fn filter_sobel_vertical_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalBorderC1,
    filter_sobel_vertical_border_typed,
    u8,
    i16,
    filter_sobel_vertical_border_u8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalBorderC1,
    filter_sobel_vertical_border_typed,
    i8,
    i16,
    filter_sobel_vertical_border_i8_to_i16_c1
);

/// Applies the vertical Sobel filter with border control.
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
pub unsafe fn filter_sobel_vertical_border_typed<T: FilterSobelVerticalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_vertical_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
