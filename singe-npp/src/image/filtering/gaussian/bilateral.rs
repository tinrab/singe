use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_bilateral_gauss_parameters, validate_border_roi},
        view::{C1, C3, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
};

impl_filter_bilateral_gauss_border!(
    "Single channel 8-bit unsigned bilateral Gauss filter with border control.",
    filter_bilateral_gauss_border_u8_c1,
    u8,
    C1,
    nppiFilterBilateralGaussBorder_8u_C1R_Ctx
);

impl_filter_bilateral_gauss_border!(
    "Three channel 8-bit unsigned bilateral Gauss filter with border control.",
    filter_bilateral_gauss_border_u8_c3,
    u8,
    C3,
    nppiFilterBilateralGaussBorder_8u_C3R_Ctx
);

impl_filter_bilateral_gauss_border!(
    "Single channel 16-bit unsigned bilateral Gauss filter with border control.",
    filter_bilateral_gauss_border_u16_c1,
    u16,
    C1,
    nppiFilterBilateralGaussBorder_16u_C1R_Ctx
);

impl_filter_bilateral_gauss_border!(
    "Three channel 16-bit unsigned bilateral Gauss filter with border control.",
    filter_bilateral_gauss_border_u16_c3,
    u16,
    C3,
    nppiFilterBilateralGaussBorder_16u_C3R_Ctx
);

impl_filter_bilateral_gauss_border!(
    "One channel 32-bit floating-point bilateral Gauss filter with border control.",
    filter_bilateral_gauss_border_f32_c1,
    f32,
    C1,
    nppiFilterBilateralGaussBorder_32f_C1R_Ctx
);

impl_filter_bilateral_gauss_border!(
    "Three channel 32-bit floating-point bilateral Gauss filter with border control.",
    filter_bilateral_gauss_border_f32_c3,
    f32,
    C3,
    nppiFilterBilateralGaussBorder_32f_C3R_Ctx
);

/// Dispatches bilateral Gaussian filtering with border control.
///
/// # Safety
///
/// Implementations must preserve native element and layout representations, filter
/// semantics, and parameter interpretation. Uphold the method's geometry validation,
/// coefficient and host-parameter requirements, memory-access rules, and execution lifetimes.
pub unsafe trait FilterBilateralGaussBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies bilateral Gaussian filtering with border control.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
    /// `radius` must be in `1..=32`, and `step_between_source_pixels` must be positive.
    /// Both squared sigma parameters must compare greater than zero, and `border_type` must be `BorderType::Replicate`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_bilateral_gauss_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_bilateral_gauss_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterBilateralGaussBorder<$layout> for $pixel_ty {
            unsafe fn filter_bilateral_gauss_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                radius: i32,
                step_between_source_pixels: i32,
                value_square_sigma: f32,
                position_square_sigma: f32,
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
                        step_between_source_pixels,
                        value_square_sigma,
                        position_square_sigma,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_bilateral_gauss_border_dispatch!(u8, C1, filter_bilateral_gauss_border_u8_c1);

impl_filter_bilateral_gauss_border_dispatch!(u8, C3, filter_bilateral_gauss_border_u8_c3);

impl_filter_bilateral_gauss_border_dispatch!(u16, C1, filter_bilateral_gauss_border_u16_c1);

impl_filter_bilateral_gauss_border_dispatch!(u16, C3, filter_bilateral_gauss_border_u16_c3);

impl_filter_bilateral_gauss_border_dispatch!(f32, C1, filter_bilateral_gauss_border_f32_c1);

impl_filter_bilateral_gauss_border_dispatch!(f32, C3, filter_bilateral_gauss_border_f32_c3);

/// Applies bilateral Gaussian filtering with border control.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
/// `radius` must be in `1..=32`, and `step_between_source_pixels` must be positive.
/// Both squared sigma parameters must compare greater than zero, and `border_type` must be `BorderType::Replicate`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_bilateral_gauss_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    radius: i32,
    step_between_source_pixels: i32,
    value_square_sigma: f32,
    position_square_sigma: f32,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterBilateralGaussBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_bilateral_gauss_border(
            stream_context,
            source,
            source_offset,
            destination,
            radius,
            step_between_source_pixels,
            value_square_sigma,
            position_square_sigma,
            border_type,
        )
    }
}
