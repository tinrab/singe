use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        color_descriptors::{
            validate_rgb_destination_planes, validate_rgb_source_planes, validate_same_size,
        },
        video::{
            VideoFormat, VideoOutputView, VideoView, rgb_16f_to_uyvp_10u_p3c3r,
            uyvp_10u_to_rgb_16f_c3c3r, uyvp_10u_to_rgba_16f_c3ac4r,
        },
        view::{AC4, C1, C3, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{ColorSpace, DataTypeLike, Point},
};

impl_rgb_to_uyvp_packed!(
    "Three unsigned 8-bit packed RGB channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.",
    rgb_to_uyvp_10u_u8_c3,
    u8,
    C3,
    nppiRGB_8u_ToUYVP_10u_C3C3R_Ctx
);

impl_rgb_to_uyvp_planar!(
    "Three unsigned 8-bit packed RGB channels color space conversion to three channel unsigned planar 10-bit UYVP pixel.",
    rgb_to_uyvp_10u_u8_c3_to_p3,
    u8,
    C3,
    nppiRGB_8u_ToUYVP_10u_C3P3R_Ctx
);

impl_rgb_to_uyvp_packed!(
    "Four unsigned 8-bit packed RGBA channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.",
    rgba_to_uyvp_10u_u8_ac4,
    u8,
    AC4,
    nppiRGBA_8u_ToUYVP_10u_AC4C3R_Ctx
);

impl_rgb_planar_to_uyvp_packed!(
    "Three unsigned 8-bit planar RGB channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.",
    rgb_to_uyvp_10u_u8_p3,
    u8,
    nppiRGB_8u_ToUYVP_10u_P3C3R_Ctx
);

impl_rgb_to_uyvp_packed!(
    "Three unsigned 16-bit packed RGB channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.",
    rgb_to_uyvp_10u_u16_c3,
    u16,
    C3,
    nppiRGB_16u_ToUYVP_10u_C3C3R_Ctx
);

impl_rgb_to_uyvp_packed!(
    "Four unsigned 16-bit packed RGBA channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.",
    rgba_to_uyvp_10u_u16_ac4,
    u16,
    AC4,
    nppiRGBA_16u_ToUYVP_10u_AC4C3R_Ctx
);

impl_rgb_planar_to_uyvp_packed!(
    "Three unsigned 16-bit planar RGB channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.",
    rgb_to_uyvp_10u_u16_p3,
    u16,
    nppiRGB_16u_ToUYVP_10u_P3C3R_Ctx
);

impl_uyvp_to_rgb_packed!(
    "Three channel unsigned packed 10-bit UYVP pixel color space conversion to three unsigned packed 8-bit channels RGB.",
    uyvp_10u_to_rgb_u8_c3,
    u8,
    C3,
    nppiUYVP_10u_ToRGB_8u_C3C3R_Ctx
);

impl_uyvp_to_rgb_planar!(
    "Three channel unsigned packed 10-bit UYVP pixel color space conversion to three unsigned planar 8-bit channels RGB.",
    uyvp_10u_to_rgb_u8_p3,
    u8,
    nppiUYVP_10u_ToRGB_8u_C3P3R_Ctx
);

impl_uyvp_to_rgb_packed!(
    "Three channel unsigned packed 10-bit UYVP pixel color space conversion to three unsigned packed 16-bit channels RGB.",
    uyvp_10u_to_rgb_u16_c3,
    u16,
    C3,
    nppiUYVP_10u_ToRGB_16u_C3C3R_Ctx
);

impl_uyvp_to_rgb_planar!(
    "Three channel unsigned packed 10-bit UYVP pixel color space conversion to three unsigned planar 16-bit channels RGB.",
    uyvp_10u_to_rgb_u16_p3,
    u16,
    nppiUYVP_10u_ToRGB_16u_C3P3R_Ctx
);

impl_uyvp_to_rgb_packed_constant_alpha!(
    "Three channel unsigned packed 10-bit UYVP pixel color space conversion to four unsigned packed 8-bit channels RGBA.",
    uyvp_10u_to_rgb_u8_ac4,
    u8,
    nppiUYVP_10u_ToRGB_8u_C3AC4R_Ctx
);

impl_uyvp_to_rgb_packed_constant_alpha!(
    "Three channel unsigned packed 10-bit UYVP pixel color space conversion to four unsigned packed 16-bit channels RGBA.",
    uyvp_10u_to_rgb_u16_ac4,
    u16,
    nppiUYVP_10u_ToRGB_16u_C3AC4R_Ctx
);

/// Dispatches conversion of packed RGB channels to packed 10-bit UYVP.
///
/// # Safety
///
/// Preserve the declared native ABI and raw execution contract.
/// Successful completed dispatch initializes the complete output ROI, including packed bytes and explicit alpha.
/// Implementations must preserve the documented color semantics, plane order, allocation lifetimes, and per-type completion behavior.
pub unsafe trait RgbToUyvp10u: DataTypeLike {
    /// Converts packed RGB channels to packed 10-bit UYVP.
    /// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
    ///
    /// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
    /// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
    /// The video view must use `VideoFormat::Uyvp10`.
    /// Integer implementations require equal full-resolution source and destination dimensions.
    /// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
    /// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
    /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn rgb_to_uyvp_10u(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination: &mut VideoOutputView<'_, u8>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_rgb_to_uyvp_10u_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl RgbToUyvp10u for $ty {
            unsafe fn rgb_to_uyvp_10u(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                source_offset: Point,
                destination: &mut VideoOutputView<'_, u8>,
                color_space: ColorSpace,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $function(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        color_space,
                    )
                }
            }
        }
    };
}

impl_rgb_to_uyvp_10u_dispatch!(u8, rgb_to_uyvp_10u_u8_c3);

impl_rgb_to_uyvp_10u_dispatch!(u16, rgb_to_uyvp_10u_u16_c3);

/// Converts packed RGB channels to packed 10-bit UYVP.
/// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
///
/// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
/// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
/// The video view must use `VideoFormat::Uyvp10`.
/// Integer implementations require equal full-resolution source and destination dimensions.
/// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
/// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
/// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn rgb_to_uyvp_10u<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: RgbToUyvp10u,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::rgb_to_uyvp_10u(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }
}

/// Dispatches conversion of packed RGBA color channels to packed 10-bit UYVP.
///
/// # Safety
///
/// Preserve the declared native ABI and raw execution contract.
/// Successful completed dispatch initializes the complete output ROI, including packed bytes and explicit alpha.
/// Implementations must preserve the documented color semantics, plane order, allocation lifetimes, and per-type completion behavior.
pub unsafe trait RgbaToUyvp10u: DataTypeLike {
    /// Converts packed RGBA color channels to packed 10-bit UYVP.
    /// Ignores source alpha during color conversion.
    /// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
    ///
    /// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
    /// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
    /// The video view must use `VideoFormat::Uyvp10`.
    /// Integer implementations require equal full-resolution source and destination dimensions.
    /// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
    /// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
    /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn rgba_to_uyvp_10u(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, AC4>,
        source_offset: Point,
        destination: &mut VideoOutputView<'_, u8>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_rgba_to_uyvp_10u_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl RgbaToUyvp10u for $ty {
            unsafe fn rgba_to_uyvp_10u(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, AC4>,
                source_offset: Point,
                destination: &mut VideoOutputView<'_, u8>,
                color_space: ColorSpace,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $function(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        color_space,
                    )
                }
            }
        }
    };
}

impl_rgba_to_uyvp_10u_dispatch!(u8, rgba_to_uyvp_10u_u8_ac4);

impl_rgba_to_uyvp_10u_dispatch!(u16, rgba_to_uyvp_10u_u16_ac4);

/// Converts packed RGBA color channels to packed 10-bit UYVP.
/// Ignores source alpha during color conversion.
/// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
///
/// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
/// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
/// The video view must use `VideoFormat::Uyvp10`.
/// Integer implementations require equal full-resolution source and destination dimensions.
/// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
/// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
/// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn rgba_to_uyvp_10u<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, AC4>,
    source_offset: Point,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: RgbaToUyvp10u,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::rgba_to_uyvp_10u(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }
}

/// Dispatches conversion of separate RGB source planes to packed 10-bit UYVP.
///
/// # Safety
///
/// Preserve the declared native ABI and raw execution contract.
/// Successful completed dispatch initializes the complete output ROI, including packed bytes and explicit alpha.
/// Implementations must preserve the documented color semantics, plane order, allocation lifetimes, and per-type completion behavior.
pub unsafe trait RgbPlanarToUyvp10u: DataTypeLike {
    /// Converts separate RGB source planes to packed 10-bit UYVP.
    /// Plane order is red, green, then blue.
    /// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
    ///
    /// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
    /// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
    /// The video view must use `VideoFormat::Uyvp10`.
    /// Integer implementations require equal full-resolution source and destination dimensions.
    /// Integer RGB planes must have equal dimensions and byte strides.
    /// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
    /// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
    /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn rgb_planar_to_uyvp_10u(
        stream_context: &StreamContext,
        source_0: &ImageView<'_, Self, C1>,
        source_1: &ImageView<'_, Self, C1>,
        source_2: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut VideoOutputView<'_, u8>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_rgb_planar_to_uyvp_10u_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl RgbPlanarToUyvp10u for $ty {
            unsafe fn rgb_planar_to_uyvp_10u(
                stream_context: &StreamContext,
                source_0: &ImageView<'_, Self, C1>,
                source_1: &ImageView<'_, Self, C1>,
                source_2: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut VideoOutputView<'_, u8>,
                color_space: ColorSpace,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $function(
                        stream_context,
                        source_0,
                        source_1,
                        source_2,
                        source_offset,
                        destination,
                        color_space,
                    )
                }
            }
        }
    };
}

impl_rgb_planar_to_uyvp_10u_dispatch!(u8, rgb_to_uyvp_10u_u8_p3);

impl_rgb_planar_to_uyvp_10u_dispatch!(u16, rgb_to_uyvp_10u_u16_p3);

/// Converts separate RGB source planes to packed 10-bit UYVP.
/// Plane order is red, green, then blue.
/// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
///
/// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
/// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
/// The video view must use `VideoFormat::Uyvp10`.
/// Integer implementations require equal full-resolution source and destination dimensions.
/// Integer RGB planes must have equal dimensions and byte strides.
/// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
/// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
/// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn rgb_planar_to_uyvp_10u<T>(
    stream_context: &StreamContext,
    source_0: &ImageView<'_, T, C1>,
    source_1: &ImageView<'_, T, C1>,
    source_2: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: RgbPlanarToUyvp10u,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::rgb_planar_to_uyvp_10u(
            stream_context,
            source_0,
            source_1,
            source_2,
            source_offset,
            destination,
            color_space,
        )
    }
}

/// Dispatches conversion of packed 10-bit UYVP to packed RGB channels.
///
/// # Safety
///
/// Preserve the declared native ABI and raw execution contract.
/// Successful completed dispatch initializes the complete output ROI, including packed bytes and explicit alpha.
/// Implementations must preserve the documented color semantics, plane order, allocation lifetimes, and per-type completion behavior.
pub unsafe trait Uyvp10uToRgb: DataTypeLike {
    /// Converts packed 10-bit UYVP to packed RGB channels.
    /// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
    ///
    /// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
    /// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
    /// The video view must use `VideoFormat::Uyvp10`.
    /// Integer implementations require equal full-resolution source and destination dimensions.
    /// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
    /// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
    /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn uyvp_10u_to_rgb(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, C3>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_uyvp_10u_to_rgb_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl Uyvp10uToRgb for $ty {
            unsafe fn uyvp_10u_to_rgb(
                stream_context: &StreamContext,
                source: &VideoView<'_, u8>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, C3>,
                color_space: ColorSpace,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $function(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        color_space,
                    )
                }
            }
        }
    };
}

impl_uyvp_10u_to_rgb_dispatch!(u8, uyvp_10u_to_rgb_u8_c3);

impl_uyvp_10u_to_rgb_dispatch!(u16, uyvp_10u_to_rgb_u16_c3);

/// Converts packed 10-bit UYVP to packed RGB channels.
/// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
///
/// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
/// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
/// The video view must use `VideoFormat::Uyvp10`.
/// Integer implementations require equal full-resolution source and destination dimensions.
/// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
/// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
/// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn uyvp_10u_to_rgb<T>(
    stream_context: &StreamContext,
    source: &VideoView<'_, u8>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, C3>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: Uyvp10uToRgb,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::uyvp_10u_to_rgb(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }
}

/// Dispatches conversion of packed 10-bit UYVP to separate RGB destination planes.
///
/// # Safety
///
/// Preserve the declared native ABI and raw execution contract.
/// Successful completed dispatch initializes the complete output ROI, including packed bytes and explicit alpha.
/// Implementations must preserve the documented color semantics, plane order, allocation lifetimes, and per-type completion behavior.
pub unsafe trait Uyvp10uToRgbP3: DataTypeLike {
    /// Converts packed 10-bit UYVP to separate RGB destination planes.
    /// Plane order is red, green, then blue.
    /// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
    ///
    /// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
    /// The video view must use `VideoFormat::Uyvp10`.
    /// Integer implementations require equal full-resolution source and destination dimensions.
    /// Integer RGB planes must have equal dimensions and byte strides.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
    /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn uyvp_10u_to_rgb_p3(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination_0: &mut ImageOutputView<'_, Self, C1>,
        destination_1: &mut ImageOutputView<'_, Self, C1>,
        destination_2: &mut ImageOutputView<'_, Self, C1>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_uyvp_10u_to_rgb_p3_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl Uyvp10uToRgbP3 for $ty {
            unsafe fn uyvp_10u_to_rgb_p3(
                stream_context: &StreamContext,
                source: &VideoView<'_, u8>,
                source_offset: Point,
                destination_0: &mut ImageOutputView<'_, Self, C1>,
                destination_1: &mut ImageOutputView<'_, Self, C1>,
                destination_2: &mut ImageOutputView<'_, Self, C1>,
                color_space: ColorSpace,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $function(
                        stream_context,
                        source,
                        source_offset,
                        destination_0,
                        destination_1,
                        destination_2,
                        color_space,
                    )
                }
            }
        }
    };
}

impl_uyvp_10u_to_rgb_p3_dispatch!(u8, uyvp_10u_to_rgb_u8_p3);

impl_uyvp_10u_to_rgb_p3_dispatch!(u16, uyvp_10u_to_rgb_u16_p3);

/// Converts packed 10-bit UYVP to separate RGB destination planes.
/// Plane order is red, green, then blue.
/// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
///
/// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
/// The video view must use `VideoFormat::Uyvp10`.
/// Integer implementations require equal full-resolution source and destination dimensions.
/// Integer RGB planes must have equal dimensions and byte strides.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
/// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn uyvp_10u_to_rgb_p3<T>(
    stream_context: &StreamContext,
    source: &VideoView<'_, u8>,
    source_offset: Point,
    destination_0: &mut ImageOutputView<'_, T, C1>,
    destination_1: &mut ImageOutputView<'_, T, C1>,
    destination_2: &mut ImageOutputView<'_, T, C1>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: Uyvp10uToRgbP3,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::uyvp_10u_to_rgb_p3(
            stream_context,
            source,
            source_offset,
            destination_0,
            destination_1,
            destination_2,
            color_space,
        )
    }
}

/// Dispatches conversion of packed 10-bit UYVP to packed RGBA channels with explicit alpha.
///
/// # Safety
///
/// Preserve the declared native ABI and raw execution contract.
/// Successful completed dispatch initializes the complete output ROI, including packed bytes and explicit alpha.
/// Implementations must preserve the documented color semantics, plane order, allocation lifetimes, and per-type completion behavior.
pub unsafe trait Uyvp10uToRgba: DataTypeLike {
    /// Converts packed 10-bit UYVP to packed RGBA channels with explicit alpha.
    /// Writes `alpha` to the fourth destination channel, including for the `AC4` output layout.
    /// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
    ///
    /// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
    /// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`. The supplied half alpha bits are copied exactly.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
    /// The video view must use `VideoFormat::Uyvp10`.
    /// Integer implementations require equal full-resolution source and destination dimensions.
    /// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
    /// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
    /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
    unsafe fn uyvp_10u_to_rgba(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, AC4>,
        color_space: ColorSpace,
        alpha: Self,
    ) -> Result<()>;
}

unsafe impl Uyvp10uToRgba for u8 {
    unsafe fn uyvp_10u_to_rgba(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, AC4>,
        color_space: ColorSpace,
        alpha: Self,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            uyvp_10u_to_rgb_u8_ac4(
                stream_context,
                source,
                source_offset,
                destination,
                color_space,
                alpha,
            )
        }
    }
}

unsafe impl Uyvp10uToRgba for u16 {
    unsafe fn uyvp_10u_to_rgba(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, AC4>,
        color_space: ColorSpace,
        alpha: Self,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            uyvp_10u_to_rgb_u16_ac4(
                stream_context,
                source,
                source_offset,
                destination,
                color_space,
                alpha,
            )
        }
    }
}

/// Converts packed 10-bit UYVP to packed RGBA channels with explicit alpha.
/// Writes `alpha` to the fourth destination channel, including for the `AC4` output layout.
/// `source_offset` must be `(0, 0)`. `color_space` selects the native conversion standard.
///
/// The provided `u8` and `u16` implementations enqueue work without waiting for stream completion.
/// The provided `f16` implementation uses completed integer-codec composition, prepares region producers, rejects stream capture, and retains temporary storage until native work completes. Half color channels use normalized values in `[0, 1]`. The supplied half alpha bits are copied exactly.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return errors if CUDA context binding or region-context validation fails, `source_offset` is nonzero, or NPP rejects the conversion.
/// The video view must use `VideoFormat::Uyvp10`.
/// Integer implementations require equal full-resolution source and destination dimensions.
/// The half implementation requires the destination pixel extent to fit within the source, with equal byte strides among source planes and among destination planes.
/// Frame or image view construction, checked sample arithmetic, temporary allocation or scalar conversion, producer waits, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution. Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain allocations through native completion, including submission errors and every captured replay for queued integer calls.
/// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
pub unsafe fn uyvp_10u_to_rgba<T>(
    stream_context: &StreamContext,
    source: &VideoView<'_, u8>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, AC4>,
    color_space: ColorSpace,
    alpha: T,
) -> Result<()>
where
    T: Uyvp10uToRgba,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::uyvp_10u_to_rgba(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
            alpha,
        )
    }
}

// Half codecs use completed composition because NPP 13.2's native half decoders
// return success without writing pixels. Owned conversion scratch cannot escape
// through capture. Direct native entry points remain public under raw::video.
macro_rules! impl_half_uyvp_encoder {
    ($trait:ident, $method:ident, $layout:ty, $format:ident, $convert:ident) => {
        unsafe impl $trait for f16 {
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut VideoOutputView<'_, u8>,
                color_space: ColorSpace,
            ) -> Result<()> {
                let source = VideoView::create(
                    VideoFormat::$format,
                    source.size(),
                    vec![(*source).into_samples()?],
                )?;
                $crate::image::video::$convert(
                    stream_context,
                    &source,
                    destination,
                    source_offset,
                    color_space,
                )
            }
        }
    };
}

impl_half_uyvp_encoder!(
    RgbToUyvp10u,
    rgb_to_uyvp_10u,
    C3,
    Rgb,
    rgb_16f_to_uyvp_10u_c3c3r
);

impl_half_uyvp_encoder!(
    RgbaToUyvp10u,
    rgba_to_uyvp_10u,
    AC4,
    Rgba,
    rgba_16f_to_uyvp_10u_ac4c3r
);

unsafe impl RgbPlanarToUyvp10u for f16 {
    unsafe fn rgb_planar_to_uyvp_10u(
        stream_context: &StreamContext,
        source_0: &ImageView<'_, Self, C1>,
        source_1: &ImageView<'_, Self, C1>,
        source_2: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut VideoOutputView<'_, u8>,
        color_space: ColorSpace,
    ) -> Result<()> {
        let source = VideoView::create(
            VideoFormat::RgbPlanar,
            source_0.size(),
            vec![*source_0, *source_1, *source_2],
        )?;
        rgb_16f_to_uyvp_10u_p3c3r(
            stream_context,
            &source,
            destination,
            source_offset,
            color_space,
        )
    }
}

unsafe impl Uyvp10uToRgb for f16 {
    unsafe fn uyvp_10u_to_rgb(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, C3>,
        color_space: ColorSpace,
    ) -> Result<()> {
        let mut output = VideoOutputView::create(
            VideoFormat::Rgb,
            destination.size(),
            vec![destination.output().into_samples()?],
        )?;
        uyvp_10u_to_rgb_16f_c3c3r(
            stream_context,
            source,
            &mut output,
            source_offset,
            color_space,
        )
    }
}

unsafe impl Uyvp10uToRgba for f16 {
    unsafe fn uyvp_10u_to_rgba(
        stream_context: &StreamContext,
        source: &VideoView<'_, u8>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, AC4>,
        color_space: ColorSpace,
        alpha: Self,
    ) -> Result<()> {
        let mut output = VideoOutputView::create(
            VideoFormat::Rgba,
            destination.size(),
            vec![destination.output().into_samples()?],
        )?;
        uyvp_10u_to_rgba_16f_c3ac4r(
            stream_context,
            source,
            &mut output,
            source_offset,
            color_space,
            alpha,
        )
    }
}
