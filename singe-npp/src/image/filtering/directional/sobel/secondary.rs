use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_same_size},
        view::{C1, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, MaskSize, Point},
};

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point horizontal Sobel filter.",
    filter_sobel_horizontal_mask_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizMask_32f_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point vertical Sobel filter.",
    filter_sobel_vertical_mask_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertMask_32f_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit unsigned to 16-bit signed second derivative, horizontal Sobel filter.",
    filter_sobel_horizontal_second_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecond_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit signed to 16-bit signed second derivative, horizontal Sobel filter.",
    filter_sobel_horizontal_second_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecond_8s16s_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point second derivative, horizontal Sobel filter.",
    filter_sobel_horizontal_second_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizSecond_32f_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit unsigned to 16-bit signed second derivative, vertical Sobel filter.",
    filter_sobel_vertical_second_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecond_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit signed to 16-bit signed second derivative, vertical Sobel filter.",
    filter_sobel_vertical_second_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecond_8s16s_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point second derivative, vertical Sobel filter.",
    filter_sobel_vertical_second_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertSecond_32f_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit unsigned to 16-bit signed second cross derivative Sobel filter.",
    filter_sobel_cross_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelCross_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit signed to 16-bit signed second cross derivative Sobel filter.",
    filter_sobel_cross_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelCross_8s16s_C1R_Ctx
);

impl_filter_masked_kernel!(
    "Single channel 32-bit floating-point second cross derivative Sobel filter.",
    filter_sobel_cross_f32_c1,
    f32,
    C1,
    nppiFilterSobelCross_32f_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point horizontal Sobel filter with border control.",
    filter_sobel_horizontal_mask_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point vertical Sobel filter with border control.",
    filter_sobel_vertical_mask_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertMaskBorder_32f_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed second derivative, horizontal Sobel filter with border control.",
    filter_sobel_horizontal_second_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecondBorder_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit signed to 16-bit signed second derivative, horizontal Sobel filter with border control.",
    filter_sobel_horizontal_second_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecondBorder_8s16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point second derivative, horizontal Sobel filter with border control.",
    filter_sobel_horizontal_second_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizSecondBorder_32f_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed second derivative, vertical Sobel filter with border control.",
    filter_sobel_vertical_second_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecondBorder_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit signed to 16-bit signed second derivative, vertical Sobel filter with border control.",
    filter_sobel_vertical_second_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecondBorder_8s16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point second derivative, vertical Sobel filter with border control.",
    filter_sobel_vertical_second_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertSecondBorder_32f_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed second cross derivative Sobel filter with border control.",
    filter_sobel_cross_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelCrossBorder_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit signed to 16-bit signed second cross derivative Sobel filter with border control.",
    filter_sobel_cross_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelCrossBorder_8s16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    "Single channel 32-bit floating-point second cross derivative Sobel filter with border control.",
    filter_sobel_cross_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelCrossBorder_32f_C1R_Ctx
);

/// Dispatches a Sobel derivative filter for a supported native source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelHorizontalSecondC1: DataTypeLike {
    /// Element type of the Sobel derivative output.
    type Output: DataTypeLike;

    /// Computes the horizontal second derivative using a Sobel filter.
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
    unsafe fn filter_sobel_horizontal_second_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalSecondC1,
    filter_sobel_horizontal_second_typed,
    u8,
    i16,
    filter_sobel_horizontal_second_u8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalSecondC1,
    filter_sobel_horizontal_second_typed,
    i8,
    i16,
    filter_sobel_horizontal_second_i8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalSecondC1,
    filter_sobel_horizontal_second_typed,
    f32,
    f32,
    filter_sobel_horizontal_second_f32_c1
);

/// Computes the horizontal second derivative using a Sobel filter.
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
pub unsafe fn filter_sobel_horizontal_second_typed<T: FilterSobelHorizontalSecondC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_horizontal_second_typed(stream_context, source, destination, mask_size)
    }
}

/// Dispatches a Sobel derivative filter for a supported native source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelVerticalSecondC1: DataTypeLike {
    /// Element type of the Sobel derivative output.
    type Output: DataTypeLike;

    /// Computes the vertical second derivative using a Sobel filter.
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
    unsafe fn filter_sobel_vertical_second_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalSecondC1,
    filter_sobel_vertical_second_typed,
    u8,
    i16,
    filter_sobel_vertical_second_u8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalSecondC1,
    filter_sobel_vertical_second_typed,
    i8,
    i16,
    filter_sobel_vertical_second_i8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalSecondC1,
    filter_sobel_vertical_second_typed,
    f32,
    f32,
    filter_sobel_vertical_second_f32_c1
);

/// Computes the vertical second derivative using a Sobel filter.
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
pub unsafe fn filter_sobel_vertical_second_typed<T: FilterSobelVerticalSecondC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sobel_vertical_second_typed(stream_context, source, destination, mask_size) }
}

/// Dispatches a Sobel derivative filter for a supported native source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelCrossC1: DataTypeLike {
    /// Element type of the Sobel derivative output.
    type Output: DataTypeLike;

    /// Computes the second cross derivative using a Sobel filter.
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
    unsafe fn filter_sobel_cross_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelCrossC1,
    filter_sobel_cross_typed,
    u8,
    i16,
    filter_sobel_cross_u8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelCrossC1,
    filter_sobel_cross_typed,
    i8,
    i16,
    filter_sobel_cross_i8_to_i16_c1
);

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelCrossC1,
    filter_sobel_cross_typed,
    f32,
    f32,
    filter_sobel_cross_f32_c1
);

/// Computes the second cross derivative using a Sobel filter.
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
pub unsafe fn filter_sobel_cross_typed<T: FilterSobelCrossC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_sobel_cross_typed(stream_context, source, destination, mask_size) }
}

/// Dispatches a Sobel derivative filter for a supported native source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelHorizontalSecondBorderC1: DataTypeLike {
    /// Element type of the Sobel derivative output.
    type Output: DataTypeLike;

    /// Computes the horizontal second derivative using a Sobel filter with border control.
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
    unsafe fn filter_sobel_horizontal_second_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalSecondBorderC1,
    filter_sobel_horizontal_second_border_typed,
    u8,
    i16,
    filter_sobel_horizontal_second_border_u8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalSecondBorderC1,
    filter_sobel_horizontal_second_border_typed,
    i8,
    i16,
    filter_sobel_horizontal_second_border_i8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalSecondBorderC1,
    filter_sobel_horizontal_second_border_typed,
    f32,
    f32,
    filter_sobel_horizontal_second_border_f32_c1
);

/// Computes the horizontal second derivative using a Sobel filter with border control.
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
pub unsafe fn filter_sobel_horizontal_second_border_typed<
    T: FilterSobelHorizontalSecondBorderC1,
>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_horizontal_second_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}

/// Dispatches a Sobel derivative filter for a supported native source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelVerticalSecondBorderC1: DataTypeLike {
    /// Element type of the Sobel derivative output.
    type Output: DataTypeLike;

    /// Computes the vertical second derivative using a Sobel filter with border control.
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
    unsafe fn filter_sobel_vertical_second_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalSecondBorderC1,
    filter_sobel_vertical_second_border_typed,
    u8,
    i16,
    filter_sobel_vertical_second_border_u8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalSecondBorderC1,
    filter_sobel_vertical_second_border_typed,
    i8,
    i16,
    filter_sobel_vertical_second_border_i8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalSecondBorderC1,
    filter_sobel_vertical_second_border_typed,
    f32,
    f32,
    filter_sobel_vertical_second_border_f32_c1
);

/// Computes the vertical second derivative using a Sobel filter with border control.
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
pub unsafe fn filter_sobel_vertical_second_border_typed<T: FilterSobelVerticalSecondBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_vertical_second_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}

/// Dispatches a Sobel derivative filter for a supported native source representation.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait FilterSobelCrossBorderC1: DataTypeLike {
    /// Element type of the Sobel derivative output.
    type Output: DataTypeLike;

    /// Computes the second cross derivative using a Sobel filter with border control.
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
    unsafe fn filter_sobel_cross_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelCrossBorderC1,
    filter_sobel_cross_border_typed,
    u8,
    i16,
    filter_sobel_cross_border_u8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelCrossBorderC1,
    filter_sobel_cross_border_typed,
    i8,
    i16,
    filter_sobel_cross_border_i8_to_i16_c1
);

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelCrossBorderC1,
    filter_sobel_cross_border_typed,
    f32,
    f32,
    filter_sobel_cross_border_f32_c1
);

/// Computes the second cross derivative using a Sobel filter with border control.
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
pub unsafe fn filter_sobel_cross_border_typed<T: FilterSobelCrossBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_sobel_cross_border_typed(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
