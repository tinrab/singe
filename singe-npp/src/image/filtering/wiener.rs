use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{validate_border_roi, validate_filter_mask},
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
};

impl_filter_wiener_border!(
    "Single channel 8-bit unsigned Wiener filter with border control.",
    filter_wiener_border_u8_c1,
    u8,
    C1,
    1,
    nppiFilterWienerBorder_8u_C1R_Ctx
);

impl_filter_wiener_border!(
    "Three channel 8-bit unsigned Wiener filter with border control.",
    filter_wiener_border_u8_c3,
    u8,
    C3,
    3,
    nppiFilterWienerBorder_8u_C3R_Ctx
);

impl_filter_wiener_border!(
    "Four channel 8-bit unsigned Wiener filter with border control.",
    filter_wiener_border_u8_c4,
    u8,
    C4,
    4,
    nppiFilterWienerBorder_8u_C4R_Ctx
);

impl_filter_wiener_border!(
    "Four channel 8-bit unsigned Wiener filter with border control, ignoring alpha channel.",
    filter_wiener_border_u8_ac4,
    u8,
    AC4,
    3,
    nppiFilterWienerBorder_8u_AC4R_Ctx
);

impl_filter_wiener_border!(
    "Single channel 16-bit signed Wiener filter with border control.",
    filter_wiener_border_i16_c1,
    i16,
    C1,
    1,
    nppiFilterWienerBorder_16s_C1R_Ctx
);

impl_filter_wiener_border!(
    "Three channel 16-bit signed Wiener filter with border control.",
    filter_wiener_border_i16_c3,
    i16,
    C3,
    3,
    nppiFilterWienerBorder_16s_C3R_Ctx
);

impl_filter_wiener_border!(
    "Four channel 16-bit signed Wiener filter with border control.",
    filter_wiener_border_i16_c4,
    i16,
    C4,
    4,
    nppiFilterWienerBorder_16s_C4R_Ctx
);

impl_filter_wiener_border!(
    "Four channel 16-bit signed Wiener filter with border control, ignoring alpha channel.",
    filter_wiener_border_i16_ac4,
    i16,
    AC4,
    3,
    nppiFilterWienerBorder_16s_AC4R_Ctx
);

impl_filter_wiener_border!(
    "Single channel 32-bit float Wiener filter with border control.",
    filter_wiener_border_f32_c1,
    f32,
    C1,
    1,
    nppiFilterWienerBorder_32f_C1R_Ctx
);

impl_filter_wiener_border!(
    "Three channel 32-bit float Wiener filter with border control.",
    filter_wiener_border_f32_c3,
    f32,
    C3,
    3,
    nppiFilterWienerBorder_32f_C3R_Ctx
);

impl_filter_wiener_border!(
    "Four channel 32-bit float Wiener filter with border control.",
    filter_wiener_border_f32_c4,
    f32,
    C4,
    4,
    nppiFilterWienerBorder_32f_C4R_Ctx
);

impl_filter_wiener_border!(
    "Four channel 32-bit float Wiener filter with border control, ignoring alpha channel.",
    filter_wiener_border_f32_ac4,
    f32,
    AC4,
    3,
    nppiFilterWienerBorder_32f_AC4R_Ctx
);

/// Dispatches Wiener filtering with border control.
///
/// # Safety
///
/// Implementations must preserve native element and layout representations, filter
/// semantics, and parameter interpretation. Uphold the method's geometry validation,
/// coefficient and host-parameter requirements, memory-access rules, and execution lifetimes.
pub unsafe trait FilterWienerBorder<L: ChannelLayout, const N: usize>: DataTypeLike {
    /// Applies Wiener filtering with border control.
    ///
    /// Enqueues work on the selected stream.
    /// Uses local statistics from `mask_size` around `anchor` and one noise variance level per processed channel.
    /// NPP expects noise levels in `0.0..=1.0`; this wrapper does not validate their values.
    /// AC4 alpha and row padding remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// Mask dimensions must be positive, `anchor` must lie within the mask, and `border_type` must be `BorderType::Replicate`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Noise storage must satisfy the native address-space requirements and remain exclusively available for as long as the native routine uses it.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_wiener_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; N],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_wiener_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $noise_len:expr, $name:ident) => {
        unsafe impl FilterWienerBorder<$layout, $noise_len> for $pixel_ty {
            unsafe fn filter_wiener_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                mask_size: Size,
                anchor: Point,
                noise: &mut [f32; $noise_len],
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
                        anchor,
                        noise,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_wiener_border_dispatch!(u8, C1, 1, filter_wiener_border_u8_c1);

impl_filter_wiener_border_dispatch!(u8, C3, 3, filter_wiener_border_u8_c3);

impl_filter_wiener_border_dispatch!(u8, C4, 4, filter_wiener_border_u8_c4);

impl_filter_wiener_border_dispatch!(u8, AC4, 3, filter_wiener_border_u8_ac4);

impl_filter_wiener_border_dispatch!(i16, C1, 1, filter_wiener_border_i16_c1);

impl_filter_wiener_border_dispatch!(i16, C3, 3, filter_wiener_border_i16_c3);

impl_filter_wiener_border_dispatch!(i16, C4, 4, filter_wiener_border_i16_c4);

impl_filter_wiener_border_dispatch!(i16, AC4, 3, filter_wiener_border_i16_ac4);

impl_filter_wiener_border_dispatch!(f32, C1, 1, filter_wiener_border_f32_c1);

impl_filter_wiener_border_dispatch!(f32, C3, 3, filter_wiener_border_f32_c3);

impl_filter_wiener_border_dispatch!(f32, C4, 4, filter_wiener_border_f32_c4);

impl_filter_wiener_border_dispatch!(f32, AC4, 3, filter_wiener_border_f32_ac4);

/// Applies Wiener filtering with border control.
///
/// Enqueues work on the selected stream.
/// Uses local statistics from `mask_size` around `anchor` and one noise variance level per processed channel.
/// NPP expects noise levels in `0.0..=1.0`; this wrapper does not validate their values.
/// AC4 alpha and row padding remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// Mask dimensions must be positive, `anchor` must lie within the mask, and `border_type` must be `BorderType::Replicate`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Noise storage must satisfy the native address-space requirements and remain exclusively available for as long as the native routine uses it.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_wiener_border<T, L, const N: usize>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    mask_size: Size,
    anchor: Point,
    noise: &mut [f32; N],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterWienerBorder<L, N>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_wiener_border(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            noise,
            border_type,
        )
    }
}
