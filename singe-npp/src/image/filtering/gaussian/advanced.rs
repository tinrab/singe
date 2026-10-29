use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{
            validate_advanced_filter_kernel, validate_border_roi, validate_same_size,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
};

impl_filter_advanced!(
    "Single channel 8-bit unsigned Gauss filter.",
    filter_gauss_advanced_u8_c1,
    u8,
    C1,
    nppiFilterGaussAdvanced_8u_C1R_Ctx
);

impl_filter_advanced!(
    "Three channel 8-bit unsigned Gauss filter.",
    filter_gauss_advanced_u8_c3,
    u8,
    C3,
    nppiFilterGaussAdvanced_8u_C3R_Ctx
);

impl_filter_advanced!(
    "Four channel 8-bit unsigned Gauss filter.",
    filter_gauss_advanced_u8_c4,
    u8,
    C4,
    nppiFilterGaussAdvanced_8u_C4R_Ctx
);

impl_filter_advanced!(
    "Four channel 8-bit unsigned Gauss filter, ignoring alpha channel.",
    filter_gauss_advanced_u8_ac4,
    u8,
    AC4,
    nppiFilterGaussAdvanced_8u_AC4R_Ctx
);

impl_filter_advanced!(
    "Single channel 16-bit unsigned Gauss filter.",
    filter_gauss_advanced_u16_c1,
    u16,
    C1,
    nppiFilterGaussAdvanced_16u_C1R_Ctx
);

impl_filter_advanced!(
    "Three channel 16-bit unsigned Gauss filter.",
    filter_gauss_advanced_u16_c3,
    u16,
    C3,
    nppiFilterGaussAdvanced_16u_C3R_Ctx
);

impl_filter_advanced!(
    "Four channel 16-bit unsigned Gauss filter.",
    filter_gauss_advanced_u16_c4,
    u16,
    C4,
    nppiFilterGaussAdvanced_16u_C4R_Ctx
);

impl_filter_advanced!(
    "Four channel 16-bit unsigned Gauss filter, ignoring alpha channel.",
    filter_gauss_advanced_u16_ac4,
    u16,
    AC4,
    nppiFilterGaussAdvanced_16u_AC4R_Ctx
);

impl_filter_advanced!(
    "Single channel 16-bit signed Gauss filter.",
    filter_gauss_advanced_i16_c1,
    i16,
    C1,
    nppiFilterGaussAdvanced_16s_C1R_Ctx
);

impl_filter_advanced!(
    "Three channel 16-bit signed Gauss filter.",
    filter_gauss_advanced_i16_c3,
    i16,
    C3,
    nppiFilterGaussAdvanced_16s_C3R_Ctx
);

impl_filter_advanced!(
    "Four channel 16-bit signed Gauss filter.",
    filter_gauss_advanced_i16_c4,
    i16,
    C4,
    nppiFilterGaussAdvanced_16s_C4R_Ctx
);

impl_filter_advanced!(
    "Four channel 16-bit signed Gauss filter, ignoring alpha channel.",
    filter_gauss_advanced_i16_ac4,
    i16,
    AC4,
    nppiFilterGaussAdvanced_16s_AC4R_Ctx
);

impl_filter_advanced!(
    "Single channel 32-bit floating-point Gauss filter.",
    filter_gauss_advanced_f32_c1,
    f32,
    C1,
    nppiFilterGaussAdvanced_32f_C1R_Ctx
);

impl_filter_advanced!(
    "Three channel 32-bit floating-point Gauss filter.",
    filter_gauss_advanced_f32_c3,
    f32,
    C3,
    nppiFilterGaussAdvanced_32f_C3R_Ctx
);

impl_filter_advanced!(
    "Four channel 32-bit floating-point Gauss filter.",
    filter_gauss_advanced_f32_c4,
    f32,
    C4,
    nppiFilterGaussAdvanced_32f_C4R_Ctx
);

impl_filter_advanced!(
    "Four channel 32-bit floating-point Gauss filter, ignoring alpha channel.",
    filter_gauss_advanced_f32_ac4,
    f32,
    AC4,
    nppiFilterGaussAdvanced_32f_AC4R_Ctx
);

impl_filter_advanced_border!(
    "Single channel 8-bit unsigned Gauss filter with border control.",
    filter_gauss_advanced_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussAdvancedBorder_8u_C1R_Ctx
);

impl_filter_advanced_border!(
    "Three channel 8-bit unsigned Gauss filter with border control.",
    filter_gauss_advanced_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussAdvancedBorder_8u_C3R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 8-bit unsigned Gauss filter with border control.",
    filter_gauss_advanced_border_u8_c4,
    u8,
    C4,
    nppiFilterGaussAdvancedBorder_8u_C4R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 8-bit unsigned Gauss filter with border control, ignoring alpha channel.",
    filter_gauss_advanced_border_u8_ac4,
    u8,
    AC4,
    nppiFilterGaussAdvancedBorder_8u_AC4R_Ctx
);

impl_filter_advanced_border!(
    "Single channel 16-bit unsigned Gauss filter with border control.",
    filter_gauss_advanced_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussAdvancedBorder_16u_C1R_Ctx
);

impl_filter_advanced_border!(
    "Three channel 16-bit unsigned Gauss filter with border control.",
    filter_gauss_advanced_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussAdvancedBorder_16u_C3R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 16-bit unsigned Gauss filter with border control.",
    filter_gauss_advanced_border_u16_c4,
    u16,
    C4,
    nppiFilterGaussAdvancedBorder_16u_C4R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 16-bit unsigned Gauss filter with border control, ignoring alpha channel.",
    filter_gauss_advanced_border_u16_ac4,
    u16,
    AC4,
    nppiFilterGaussAdvancedBorder_16u_AC4R_Ctx
);

impl_filter_advanced_border!(
    "Single channel 16-bit signed Gauss filter with border control.",
    filter_gauss_advanced_border_i16_c1,
    i16,
    C1,
    nppiFilterGaussAdvancedBorder_16s_C1R_Ctx
);

impl_filter_advanced_border!(
    "Three channel 16-bit signed Gauss filter with border control.",
    filter_gauss_advanced_border_i16_c3,
    i16,
    C3,
    nppiFilterGaussAdvancedBorder_16s_C3R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 16-bit signed Gauss filter with border control.",
    filter_gauss_advanced_border_i16_c4,
    i16,
    C4,
    nppiFilterGaussAdvancedBorder_16s_C4R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 16-bit signed Gauss filter with border control, ignoring alpha channel.",
    filter_gauss_advanced_border_i16_ac4,
    i16,
    AC4,
    nppiFilterGaussAdvancedBorder_16s_AC4R_Ctx
);

impl_filter_advanced_border!(
    "Single channel 32-bit floating-point Gauss filter with border control.",
    filter_gauss_advanced_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussAdvancedBorder_32f_C1R_Ctx
);

impl_filter_advanced_border!(
    "Three channel 32-bit floating-point Gauss filter with border control.",
    filter_gauss_advanced_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussAdvancedBorder_32f_C3R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 32-bit floating-point Gauss filter with border control.",
    filter_gauss_advanced_border_f32_c4,
    f32,
    C4,
    nppiFilterGaussAdvancedBorder_32f_C4R_Ctx
);

impl_filter_advanced_border!(
    "Four channel 32-bit floating-point Gauss filter with border control, ignoring alpha channel.",
    filter_gauss_advanced_border_f32_ac4,
    f32,
    AC4,
    nppiFilterGaussAdvancedBorder_32f_AC4R_Ctx
);

/// Dispatches Gaussian filtering with caller-supplied coefficients.
///
/// # Safety
///
/// Implementations must preserve native element and layout representations, filter
/// semantics, and parameter interpretation. Uphold the method's geometry validation,
/// coefficient and host-parameter requirements, memory-access rules, and execution lifetimes.
pub unsafe trait FilterGaussAdvanced<L: ChannelLayout>: DataTypeLike {
    /// Applies Gaussian filtering with caller-supplied coefficients.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// Source and destination image sizes must match.
    /// `kernel` must be nonempty and its length must fit in `i32`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
    /// The source allocation must include every neighboring sample read by the filter, including any halo outside the declared view.
    /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_gauss_advanced(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
        kernel: &[f32],
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_advanced_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterGaussAdvanced<$layout> for $pixel_ty {
            unsafe fn filter_gauss_advanced(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                kernel: &[f32],
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination, kernel) }
            }
        }
    };
}

impl_filter_gauss_advanced_dispatch!(u8, C1, filter_gauss_advanced_u8_c1);

impl_filter_gauss_advanced_dispatch!(u8, C3, filter_gauss_advanced_u8_c3);

impl_filter_gauss_advanced_dispatch!(u8, C4, filter_gauss_advanced_u8_c4);

impl_filter_gauss_advanced_dispatch!(u8, AC4, filter_gauss_advanced_u8_ac4);

impl_filter_gauss_advanced_dispatch!(u16, C1, filter_gauss_advanced_u16_c1);

impl_filter_gauss_advanced_dispatch!(u16, C3, filter_gauss_advanced_u16_c3);

impl_filter_gauss_advanced_dispatch!(u16, C4, filter_gauss_advanced_u16_c4);

impl_filter_gauss_advanced_dispatch!(u16, AC4, filter_gauss_advanced_u16_ac4);

impl_filter_gauss_advanced_dispatch!(i16, C1, filter_gauss_advanced_i16_c1);

impl_filter_gauss_advanced_dispatch!(i16, C3, filter_gauss_advanced_i16_c3);

impl_filter_gauss_advanced_dispatch!(i16, C4, filter_gauss_advanced_i16_c4);

impl_filter_gauss_advanced_dispatch!(i16, AC4, filter_gauss_advanced_i16_ac4);

impl_filter_gauss_advanced_dispatch!(f32, C1, filter_gauss_advanced_f32_c1);

impl_filter_gauss_advanced_dispatch!(f32, C3, filter_gauss_advanced_f32_c3);

impl_filter_gauss_advanced_dispatch!(f32, C4, filter_gauss_advanced_f32_c4);

impl_filter_gauss_advanced_dispatch!(f32, AC4, filter_gauss_advanced_f32_ac4);

/// Applies Gaussian filtering with caller-supplied coefficients.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// Source and destination image sizes must match.
/// `kernel` must be nonempty and its length must fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
/// The source allocation must include every neighboring sample read by the filter, including any halo outside the declared view.
/// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_gauss_advanced<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
    kernel: &[f32],
) -> Result<()>
where
    T: FilterGaussAdvanced<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::filter_gauss_advanced(stream_context, source, destination, kernel) }
}

/// Dispatches Gaussian filtering with caller-supplied coefficients with border control.
///
/// # Safety
///
/// Implementations must preserve native element and layout representations, filter
/// semantics, and parameter interpretation. Uphold the method's geometry validation,
/// coefficient and host-parameter requirements, memory-access rules, and execution lifetimes.
pub unsafe trait FilterGaussAdvancedBorder<L: ChannelLayout>: DataTypeLike {
    /// Applies Gaussian filtering with caller-supplied coefficients with border control.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
    /// `kernel` must be nonempty and its length must fit in `i32`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
    /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_gauss_advanced_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_advanced_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterGaussAdvancedBorder<$layout> for $pixel_ty {
            unsafe fn filter_gauss_advanced_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_gauss_advanced_border_dispatch!(u8, C1, filter_gauss_advanced_border_u8_c1);

impl_filter_gauss_advanced_border_dispatch!(u8, C3, filter_gauss_advanced_border_u8_c3);

impl_filter_gauss_advanced_border_dispatch!(u8, C4, filter_gauss_advanced_border_u8_c4);

impl_filter_gauss_advanced_border_dispatch!(u8, AC4, filter_gauss_advanced_border_u8_ac4);

impl_filter_gauss_advanced_border_dispatch!(u16, C1, filter_gauss_advanced_border_u16_c1);

impl_filter_gauss_advanced_border_dispatch!(u16, C3, filter_gauss_advanced_border_u16_c3);

impl_filter_gauss_advanced_border_dispatch!(u16, C4, filter_gauss_advanced_border_u16_c4);

impl_filter_gauss_advanced_border_dispatch!(u16, AC4, filter_gauss_advanced_border_u16_ac4);

impl_filter_gauss_advanced_border_dispatch!(i16, C1, filter_gauss_advanced_border_i16_c1);

impl_filter_gauss_advanced_border_dispatch!(i16, C3, filter_gauss_advanced_border_i16_c3);

impl_filter_gauss_advanced_border_dispatch!(i16, C4, filter_gauss_advanced_border_i16_c4);

impl_filter_gauss_advanced_border_dispatch!(i16, AC4, filter_gauss_advanced_border_i16_ac4);

impl_filter_gauss_advanced_border_dispatch!(f32, C1, filter_gauss_advanced_border_f32_c1);

impl_filter_gauss_advanced_border_dispatch!(f32, C3, filter_gauss_advanced_border_f32_c3);

impl_filter_gauss_advanced_border_dispatch!(f32, C4, filter_gauss_advanced_border_f32_c4);

impl_filter_gauss_advanced_border_dispatch!(f32, AC4, filter_gauss_advanced_border_f32_ac4);

/// Applies Gaussian filtering with caller-supplied coefficients with border control.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
/// `kernel` must be nonempty and its length must fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
/// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_gauss_advanced_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    kernel: &[f32],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussAdvancedBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_gauss_advanced_border(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            border_type,
        )
    }
}
