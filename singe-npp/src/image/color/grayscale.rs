use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{validate_cfa_roi, validate_same_size},
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BayerGridPosition, DataTypeLike, ImageNormalization, InterpolationMode, Rectangle},
};

impl_rgb_to_gray!(
    "3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.",
    rgb_to_gray_u8_c3,
    u8,
    C3,
    nppiRGBToGray_8u_C3C1R_Ctx
);

impl_rgb_to_gray!(
    "4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.\nInput alpha is ignored.",
    rgb_to_gray_u8_ac4,
    u8,
    AC4,
    nppiRGBToGray_8u_AC4C1R_Ctx
);

impl_rgb_to_gray!(
    "3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.",
    rgb_to_gray_u16_c3,
    u16,
    C3,
    nppiRGBToGray_16u_C3C1R_Ctx
);

impl_rgb_to_gray!(
    "4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.\nInput alpha is ignored.",
    rgb_to_gray_u16_ac4,
    u16,
    AC4,
    nppiRGBToGray_16u_AC4C1R_Ctx
);

impl_rgb_to_gray!(
    "3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.",
    rgb_to_gray_i16_c3,
    i16,
    C3,
    nppiRGBToGray_16s_C3C1R_Ctx
);

impl_rgb_to_gray!(
    "4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.\nInput alpha is ignored.",
    rgb_to_gray_i16_ac4,
    i16,
    AC4,
    nppiRGBToGray_16s_AC4C1R_Ctx
);

impl_rgb_to_gray!(
    "3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.",
    rgb_to_gray_f32_c3,
    f32,
    C3,
    nppiRGBToGray_32f_C3C1R_Ctx
);

impl_rgb_to_gray!(
    "4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.\nInput alpha is ignored.",
    rgb_to_gray_f32_ac4,
    f32,
    AC4,
    nppiRGBToGray_32f_AC4C1R_Ctx
);

impl_generic_rgb_to_gray!(
    "Converts packed RGB to one-channel gray using fixed RGB weights.",
RgbToGrayC3, rgb_to_gray, rgb_to_gray_c3, C3, [
    u8 => rgb_to_gray_u8_c3,
    u16 => rgb_to_gray_u16_c3,
    i16 => rgb_to_gray_i16_c3,
    f32 => rgb_to_gray_f32_c3,
]
);

impl_generic_rgb_to_gray!(
    "Converts packed RGB with ignored input alpha to one-channel gray using fixed RGB weights.",
RgbToGrayAC4, rgb_to_gray, rgb_to_gray_ac4, AC4, [
    u8 => rgb_to_gray_u8_ac4,
    u16 => rgb_to_gray_u16_ac4,
    i16 => rgb_to_gray_i16_ac4,
    f32 => rgb_to_gray_f32_ac4,
]
);

impl_color_to_gray!(
    "3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.",
    color_to_gray_u8_c3,
    u8,
    C3,
    3,
    nppiColorToGray_8u_C3C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.\nInput alpha is ignored.",
    color_to_gray_u8_ac4,
    u8,
    AC4,
    3,
    nppiColorToGray_8u_AC4C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 8-bit unsigned packed RGBA to 1 channel 8-bit unsigned packed Gray conversion.",
    color_to_gray_u8_c4,
    u8,
    C4,
    4,
    nppiColorToGray_8u_C4C1R_Ctx
);

impl_color_to_gray!(
    "3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.",
    color_to_gray_u16_c3,
    u16,
    C3,
    3,
    nppiColorToGray_16u_C3C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.\nInput alpha is ignored.",
    color_to_gray_u16_ac4,
    u16,
    AC4,
    3,
    nppiColorToGray_16u_AC4C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 16-bit unsigned packed RGBA to 1 channel 16-bit unsigned packed Gray conversion.",
    color_to_gray_u16_c4,
    u16,
    C4,
    4,
    nppiColorToGray_16u_C4C1R_Ctx
);

impl_color_to_gray!(
    "3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.",
    color_to_gray_i16_c3,
    i16,
    C3,
    3,
    nppiColorToGray_16s_C3C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.\nInput alpha is ignored.",
    color_to_gray_i16_ac4,
    i16,
    AC4,
    3,
    nppiColorToGray_16s_AC4C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 16-bit signed packed RGBA to 1 channel 16-bit signed packed Gray conversion.",
    color_to_gray_i16_c4,
    i16,
    C4,
    4,
    nppiColorToGray_16s_C4C1R_Ctx
);

impl_color_to_gray!(
    "3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.",
    color_to_gray_f32_c3,
    f32,
    C3,
    3,
    nppiColorToGray_32f_C3C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.\nInput alpha is ignored.",
    color_to_gray_f32_ac4,
    f32,
    AC4,
    3,
    nppiColorToGray_32f_AC4C1R_Ctx
);

impl_color_to_gray!(
    "4 channel 32-bit floating point packed RGBA to 1 channel 32-bit floating point packed Gray conversion.",
    color_to_gray_f32_c4,
    f32,
    C4,
    4,
    nppiColorToGray_32f_C4C1R_Ctx
);

impl_generic_color_to_gray!(
    "Converts packed RGB to one-channel gray using caller-supplied coefficients.",
ColorToGrayC3, color_to_gray, color_to_gray_c3, C3, 3, [
    u8 => color_to_gray_u8_c3,
    u16 => color_to_gray_u16_c3,
    i16 => color_to_gray_i16_c3,
    f32 => color_to_gray_f32_c3,
]
);

impl_generic_color_to_gray!(
    "Converts packed RGB with ignored input alpha to one-channel gray using caller-supplied coefficients.",

    ColorToGrayAC4,
    color_to_gray,
    color_to_gray_ac4,
    AC4,
    3,
    [
        u8 => color_to_gray_u8_ac4,
        u16 => color_to_gray_u16_ac4,
        i16 => color_to_gray_i16_ac4,
        f32 => color_to_gray_f32_ac4,
    ]

);

impl_generic_color_to_gray!(
    "Converts packed RGBA to one-channel gray using caller-supplied coefficients.",
ColorToGrayC4, color_to_gray, color_to_gray_c4, C4, 4, [
    u8 => color_to_gray_u8_c4,
    u16 => color_to_gray_u16_c4,
    i16 => color_to_gray_i16_c4,
    f32 => color_to_gray_f32_c4,
]
);

impl_gradient_color_to_gray!(
    "3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray Gradient conversion.",
    gradient_color_to_gray_u8_c3,
    u8,
    nppiGradientColorToGray_8u_C3C1R_Ctx
);

impl_gradient_color_to_gray!(
    "3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray Gradient conversion.",
    gradient_color_to_gray_u16_c3,
    u16,
    nppiGradientColorToGray_16u_C3C1R_Ctx
);

impl_gradient_color_to_gray!(
    "3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray Gradient conversion.",
    gradient_color_to_gray_i16_c3,
    i16,
    nppiGradientColorToGray_16s_C3C1R_Ctx
);

impl_gradient_color_to_gray!(
    "3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray Gradient conversion.",
    gradient_color_to_gray_f32_c3,
    f32,
    nppiGradientColorToGray_32f_C3C1R_Ctx
);

/// Dispatches gray gradient conversion for a native element type.
///
/// # Safety
///
/// Implementations must preserve the native element representation and uphold the documented conversion contract.
pub unsafe trait GradientColorToGray: DataTypeLike {
    /// Converts packed RGB to a one-channel gray gradient.
    ///
    /// `norm` selects the gradient distance method passed to NPP.
    /// Enqueues the conversion without waiting for stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
    /// Keep image allocations valid until queued work completes, including every replay of a captured call.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_color_to_gray(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        destination: &mut ImageOutputView<'_, Self, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

macro_rules! impl_gradient_color_to_gray_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl GradientColorToGray for $ty {
            unsafe fn gradient_color_to_gray(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                norm: ImageNormalization,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, source, destination, norm) }
            }
        }
    };
}

impl_gradient_color_to_gray_dispatch!(u8, gradient_color_to_gray_u8_c3);

impl_gradient_color_to_gray_dispatch!(u16, gradient_color_to_gray_u16_c3);

impl_gradient_color_to_gray_dispatch!(i16, gradient_color_to_gray_i16_c3);

impl_gradient_color_to_gray_dispatch!(f32, gradient_color_to_gray_f32_c3);

/// Converts packed RGB to a one-channel gray gradient.
///
/// `norm` selects the gradient distance method passed to NPP.
/// Enqueues the conversion without waiting for stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, the image sizes differ, or NPP rejects the conversion parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
/// Keep image allocations valid until queued work completes, including every replay of a captured call.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_color_to_gray<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    destination: &mut ImageOutputView<'_, T, C1>,
    norm: ImageNormalization,
) -> Result<()>
where
    T: GradientColorToGray,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::gradient_color_to_gray(stream_context, source, destination, norm) }
}

impl_cfa_to_rgb!(
    "1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 8-bit unsigned packed RGB conversion.",
    cfa_to_rgb_u8_c3,
    u8,
    C3,
    nppiCFAToRGB_8u_C1C3R_Ctx
);

impl_cfa_to_rgba!(
    "1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 8-bit unsigned packed RGB conversion with alpha.",
    cfa_to_rgba_u8_ac4,
    u8,
    nppiCFAToRGBA_8u_C1AC4R_Ctx
);

impl_cfa_to_rgb!(
    "1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 16-bit unsigned packed RGB conversion.",
    cfa_to_rgb_u16_c3,
    u16,
    C3,
    nppiCFAToRGB_16u_C1C3R_Ctx
);

impl_cfa_to_rgba!(
    "1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 16-bit unsigned packed RGB conversion with alpha.",
    cfa_to_rgba_u16_ac4,
    u16,
    nppiCFAToRGBA_16u_C1AC4R_Ctx
);

impl_cfa_to_rgb!(
    "1 channel 32-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 32-bit unsigned packed RGB conversion.",
    cfa_to_rgb_u32_c3,
    u32,
    C3,
    nppiCFAToRGB_32u_C1C3R_Ctx
);

impl_cfa_to_rgba!(
    "1 channel 32-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 32-bit unsigned packed RGB conversion with alpha.",
    cfa_to_rgba_u32_ac4,
    u32,
    nppiCFAToRGBA_32u_C1AC4R_Ctx
);

/// Dispatches Bayer mosaic conversion for a native element type.
///
/// # Safety
///
/// Implementations must preserve the native element representation and uphold the documented conversion contract.
pub unsafe trait CfaToRgb: DataTypeLike {
    /// Converts a single-channel Bayer mosaic to packed RGB.
    ///
    /// `source_roi` selects a rectangle in the full source image; `grid` gives the Bayer registration at that rectangle's origin.
    /// Uses the native required `InterpolationMode::Undefined` interpolation setting.
    /// Enqueues the conversion without waiting for stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside the source without coordinate overflow, and match the destination size.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
    /// Keep image allocations valid until queued work completes, including every replay of a captured call.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn cfa_to_rgb(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_roi: Rectangle,
        destination: &mut ImageOutputView<'_, Self, C3>,
        grid: BayerGridPosition,
    ) -> Result<()>;
}

macro_rules! impl_cfa_to_rgb_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl CfaToRgb for $ty {
            unsafe fn cfa_to_rgb(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_roi: Rectangle,
                destination: &mut ImageOutputView<'_, Self, C3>,
                grid: BayerGridPosition,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, source, source_roi, destination, grid) }
            }
        }
    };
}

impl_cfa_to_rgb_dispatch!(u8, cfa_to_rgb_u8_c3);

impl_cfa_to_rgb_dispatch!(u16, cfa_to_rgb_u16_c3);

impl_cfa_to_rgb_dispatch!(u32, cfa_to_rgb_u32_c3);

/// Converts a single-channel Bayer mosaic to packed RGB.
///
/// `source_roi` selects a rectangle in the full source image; `grid` gives the Bayer registration at that rectangle's origin.
/// Uses the native required `InterpolationMode::Undefined` interpolation setting.
/// Enqueues the conversion without waiting for stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the conversion parameters.
/// The source ROI must have a nonnegative origin and positive dimensions, fit inside the source without coordinate overflow, and match the destination size.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
/// Keep image allocations valid until queued work completes, including every replay of a captured call.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn cfa_to_rgb<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_roi: Rectangle,
    destination: &mut ImageOutputView<'_, T, C3>,
    grid: BayerGridPosition,
) -> Result<()>
where
    T: CfaToRgb,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::cfa_to_rgb(stream_context, source, source_roi, destination, grid) }
}

/// Dispatches Bayer mosaic conversion for a native element type.
///
/// # Safety
///
/// Implementations must preserve the native element representation and uphold the documented conversion contract.
pub unsafe trait CfaToRgba: DataTypeLike {
    /// Converts a single-channel Bayer mosaic to packed RGBA.
    ///
    /// `source_roi` selects a rectangle in the full source image; `grid` gives the Bayer registration at that rectangle's origin.
    /// Uses the native required `InterpolationMode::Undefined` interpolation setting.
    /// Writes `alpha` to the fourth channel of every destination pixel.
    /// Enqueues the conversion without waiting for stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside the source without coordinate overflow, and match the destination size.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
    /// Keep image allocations valid until queued work completes, including every replay of a captured call.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn cfa_to_rgba(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_roi: Rectangle,
        destination: &mut ImageOutputView<'_, Self, AC4>,
        grid: BayerGridPosition,
        alpha: Self,
    ) -> Result<()>;
}

macro_rules! impl_cfa_to_rgba_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl CfaToRgba for $ty {
            unsafe fn cfa_to_rgba(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_roi: Rectangle,
                destination: &mut ImageOutputView<'_, Self, AC4>,
                grid: BayerGridPosition,
                alpha: Self,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, source, source_roi, destination, grid, alpha) }
            }
        }
    };
}

impl_cfa_to_rgba_dispatch!(u8, cfa_to_rgba_u8_ac4);

impl_cfa_to_rgba_dispatch!(u16, cfa_to_rgba_u16_ac4);

impl_cfa_to_rgba_dispatch!(u32, cfa_to_rgba_u32_ac4);

/// Converts a single-channel Bayer mosaic to packed RGBA.
///
/// `source_roi` selects a rectangle in the full source image; `grid` gives the Bayer registration at that rectangle's origin.
/// Uses the native required `InterpolationMode::Undefined` interpolation setting.
/// Writes `alpha` to the fourth channel of every destination pixel.
/// Enqueues the conversion without waiting for stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the conversion parameters.
/// The source ROI must have a nonnegative origin and positive dimensions, fit inside the source without coordinate overflow, and match the destination size.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and readable, and destination pixels must be exclusively writable and disjoint from the source during native use.
/// Keep image allocations valid until queued work completes, including every replay of a captured call.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn cfa_to_rgba<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_roi: Rectangle,
    destination: &mut ImageOutputView<'_, T, AC4>,
    grid: BayerGridPosition,
    alpha: T,
) -> Result<()>
where
    T: CfaToRgba,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::cfa_to_rgba(stream_context, source, source_roi, destination, grid, alpha) }
}
