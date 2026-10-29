use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry::{Resize, ResizeBatchAdvanced, ResizeSqrPixel},
        geometry_helpers::{resize_batch_advanced_descriptors, resize_batch_descriptors},
        view::{
            AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, PlanarImageOutputView,
            PlanarImageView,
        },
    },
    try_ffi,
    types::{DataTypeLike, InterpolationMode},
};

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_u8_c1,
    u8,
    C1,
    nppiResizeSqrPixel_8u_C1R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_u8_c3,
    u8,
    C3,
    nppiResizeSqrPixel_8u_C3R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_u8_c4,
    u8,
    C4,
    nppiResizeSqrPixel_8u_C4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_sqr_pixel_u8_ac4,
    u8,
    AC4,
    nppiResizeSqrPixel_8u_AC4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_u16_c1,
    u16,
    C1,
    nppiResizeSqrPixel_16u_C1R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_u16_c3,
    u16,
    C3,
    nppiResizeSqrPixel_16u_C3R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_u16_c4,
    u16,
    C4,
    nppiResizeSqrPixel_16u_C4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_sqr_pixel_u16_ac4,
    u16,
    AC4,
    nppiResizeSqrPixel_16u_AC4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_i16_c1,
    i16,
    C1,
    nppiResizeSqrPixel_16s_C1R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_i16_c3,
    i16,
    C3,
    nppiResizeSqrPixel_16s_C3R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_i16_c4,
    i16,
    C4,
    nppiResizeSqrPixel_16s_C4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_sqr_pixel_i16_ac4,
    i16,
    AC4,
    nppiResizeSqrPixel_16s_AC4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_f32_c1,
    f32,
    C1,
    nppiResizeSqrPixel_32f_C1R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_f32_c3,
    f32,
    C3,
    nppiResizeSqrPixel_32f_C3R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_f32_c4,
    f32,
    C4,
    nppiResizeSqrPixel_32f_C4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_sqr_pixel_f32_ac4,
    f32,
    AC4,
    nppiResizeSqrPixel_32f_AC4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_f64_c1,
    f64,
    C1,
    nppiResizeSqrPixel_64f_C1R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_f64_c3,
    f64,
    C3,
    nppiResizeSqrPixel_64f_C3R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",
    resize_sqr_pixel_f64_c4,
    f64,
    C4,
    nppiResizeSqrPixel_64f_C4R_Ctx
);

impl_resize_sqr_pixel!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_sqr_pixel_f64_ac4,
    f64,
    AC4,
    nppiResizeSqrPixel_64f_AC4R_Ctx
);

impl_generic_resize_sqr_pixel_operation!("Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",

    ResizeSqrPixelC1,
    resize_sqr_pixel,
    resize_sqr_pixel_c1,
    C1,
    [
        u8 => resize_sqr_pixel_u8_c1,
        u16 => resize_sqr_pixel_u16_c1,
        i16 => resize_sqr_pixel_i16_c1,
        f32 => resize_sqr_pixel_f32_c1,
        f64 => resize_sqr_pixel_f64_c1,
    ]
);

impl_generic_resize_sqr_pixel_operation!("Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",

    ResizeSqrPixelC3,
    resize_sqr_pixel,
    resize_sqr_pixel_c3,
    C3,
    [
        u8 => resize_sqr_pixel_u8_c3,
        u16 => resize_sqr_pixel_u16_c3,
        i16 => resize_sqr_pixel_i16_c3,
        f32 => resize_sqr_pixel_f32_c3,
        f64 => resize_sqr_pixel_f64_c3,
    ]
);

impl_generic_resize_sqr_pixel_operation!("Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`.",

    ResizeSqrPixelC4,
    resize_sqr_pixel,
    resize_sqr_pixel_c4,
    C4,
    [
        u8 => resize_sqr_pixel_u8_c4,
        u16 => resize_sqr_pixel_u16_c4,
        i16 => resize_sqr_pixel_i16_c4,
        f32 => resize_sqr_pixel_f32_c4,
        f64 => resize_sqr_pixel_f64_c4,
    ]
);

impl_generic_resize_sqr_pixel_operation!("Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. The destination alpha channel is not processed.",

    ResizeSqrPixelAc4,
    resize_sqr_pixel,
    resize_sqr_pixel_ac4,
    AC4,
    [
        u8 => resize_sqr_pixel_u8_ac4,
        u16 => resize_sqr_pixel_u16_ac4,
        i16 => resize_sqr_pixel_i16_ac4,
        f32 => resize_sqr_pixel_f32_ac4,
        f64 => resize_sqr_pixel_f64_ac4,
    ]
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_u8_p3,
    u8,
    3,
    nppiResizeSqrPixel_8u_P3R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_u8_p4,
    u8,
    4,
    nppiResizeSqrPixel_8u_P4R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_u16_p3,
    u16,
    3,
    nppiResizeSqrPixel_16u_P3R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_u16_p4,
    u16,
    4,
    nppiResizeSqrPixel_16u_P4R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_i16_p3,
    i16,
    3,
    nppiResizeSqrPixel_16s_P3R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_i16_p4,
    i16,
    4,
    nppiResizeSqrPixel_16s_P4R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_f32_p3,
    f32,
    3,
    nppiResizeSqrPixel_32f_P3R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_f32_p4,
    f32,
    4,
    nppiResizeSqrPixel_32f_P4R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_f64_p3,
    f64,
    3,
    nppiResizeSqrPixel_64f_P3R_Ctx
);

impl_resize_sqr_pixel_planar!(
    "Resizes the source ROI using the scale factors, shifts, and interpolation in `resize`. Processes all image planes.",
    resize_sqr_pixel_f64_p4,
    f64,
    4,
    nppiResizeSqrPixel_64f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`resize_sqr_pixel_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait ResizeSqrPixelPlanar<const PLANES: usize>: DataTypeLike {
    /// Resizes every plane using the scale factors, shifts, and interpolation in `resize`.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn resize_sqr_pixel_planar(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixel,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_resize_sqr_pixel_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl ResizeSqrPixelPlanar<$planes> for $ty {
            unsafe fn resize_sqr_pixel_planar(
                stream_context: &StreamContext,
                resize: &ResizeSqrPixel,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, resize, source, destination) }
            }
        }
    };
}

impl_resize_sqr_pixel_planar_dispatch!(u8, 3, resize_sqr_pixel_u8_p3);

impl_resize_sqr_pixel_planar_dispatch!(u8, 4, resize_sqr_pixel_u8_p4);

impl_resize_sqr_pixel_planar_dispatch!(u16, 3, resize_sqr_pixel_u16_p3);

impl_resize_sqr_pixel_planar_dispatch!(u16, 4, resize_sqr_pixel_u16_p4);

impl_resize_sqr_pixel_planar_dispatch!(i16, 3, resize_sqr_pixel_i16_p3);

impl_resize_sqr_pixel_planar_dispatch!(i16, 4, resize_sqr_pixel_i16_p4);

impl_resize_sqr_pixel_planar_dispatch!(f32, 3, resize_sqr_pixel_f32_p3);

impl_resize_sqr_pixel_planar_dispatch!(f32, 4, resize_sqr_pixel_f32_p4);

impl_resize_sqr_pixel_planar_dispatch!(f64, 3, resize_sqr_pixel_f64_p3);

impl_resize_sqr_pixel_planar_dispatch!(f64, 4, resize_sqr_pixel_f64_p4);

/// Resizes every plane using the scale factors, shifts, and interpolation in `resize`.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Each destination ROI must lie within its destination image.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn resize_sqr_pixel_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixel,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: ResizeSqrPixelPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::resize_sqr_pixel_planar(stream_context, resize, source, destination) }
}

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_u8_c1,
    u8,
    C1,
    nppiResize_8u_C1R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_u8_c3,
    u8,
    C3,
    nppiResize_8u_C3R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_u8_c4,
    u8,
    C4,
    nppiResize_8u_C4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. The destination alpha channel is not processed.",
    resize_u8_ac4,
    u8,
    AC4,
    nppiResize_8u_AC4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_u16_c1,
    u16,
    C1,
    nppiResize_16u_C1R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_u16_c3,
    u16,
    C3,
    nppiResize_16u_C3R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_u16_c4,
    u16,
    C4,
    nppiResize_16u_C4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. The destination alpha channel is not processed.",
    resize_u16_ac4,
    u16,
    AC4,
    nppiResize_16u_AC4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_i16_c1,
    i16,
    C1,
    nppiResize_16s_C1R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_i16_c3,
    i16,
    C3,
    nppiResize_16s_C3R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_i16_c4,
    i16,
    C4,
    nppiResize_16s_C4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. The destination alpha channel is not processed.",
    resize_i16_ac4,
    i16,
    AC4,
    nppiResize_16s_AC4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_f16_c1,
    f16,
    C1,
    nppiResize_16f_C1R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_f16_c3,
    f16,
    C3,
    nppiResize_16f_C3R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_f16_c4,
    f16,
    C4,
    nppiResize_16f_C4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_f32_c1,
    f32,
    C1,
    nppiResize_32f_C1R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_f32_c3,
    f32,
    C3,
    nppiResize_32f_C3R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`.",
    resize_f32_c4,
    f32,
    C4,
    nppiResize_32f_C4R_Ctx
);

impl_resize!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. The destination alpha channel is not processed.",
    resize_f32_ac4,
    f32,
    AC4,
    nppiResize_32f_AC4R_Ctx
);

impl_generic_resize_operation!("Resizes the source ROI into the destination ROI using `resize.interpolation`.",
ResizeC1, resize, resize_c1, C1, [
    u8 => resize_u8_c1,
    u16 => resize_u16_c1,
    i16 => resize_i16_c1,
    f16 => resize_f16_c1,
    f32 => resize_f32_c1,
]);

impl_generic_resize_operation!("Resizes the source ROI into the destination ROI using `resize.interpolation`.",
ResizeC3, resize, resize_c3, C3, [
    u8 => resize_u8_c3,
    u16 => resize_u16_c3,
    i16 => resize_i16_c3,
    f16 => resize_f16_c3,
    f32 => resize_f32_c3,
]);

impl_generic_resize_operation!("Resizes the source ROI into the destination ROI using `resize.interpolation`.",
ResizeC4, resize, resize_c4, C4, [
    u8 => resize_u8_c4,
    u16 => resize_u16_c4,
    i16 => resize_i16_c4,
    f16 => resize_f16_c4,
    f32 => resize_f32_c4,
]);

impl_generic_resize_operation!("Resizes the source ROI into the destination ROI using `resize.interpolation`. The destination alpha channel is not processed.",
ResizeAc4, resize, resize_ac4, AC4, [
    u8 => resize_u8_ac4,
    u16 => resize_u16_ac4,
    i16 => resize_i16_ac4,
    f32 => resize_f32_ac4,
]);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_u8_p3,
    u8,
    3,
    nppiResize_8u_P3R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_u8_p4,
    u8,
    4,
    nppiResize_8u_P4R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_u16_p3,
    u16,
    3,
    nppiResize_16u_P3R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_u16_p4,
    u16,
    4,
    nppiResize_16u_P4R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_i16_p3,
    i16,
    3,
    nppiResize_16s_P3R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_i16_p4,
    i16,
    4,
    nppiResize_16s_P4R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_f32_p3,
    f32,
    3,
    nppiResize_32f_P3R_Ctx
);

impl_resize_planar!(
    "Resizes the source ROI into the destination ROI using `resize.interpolation`. Processes all image planes.",
    resize_f32_p4,
    f32,
    4,
    nppiResize_32f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`resize_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait ResizePlanar<const PLANES: usize>: DataTypeLike {
    /// Resizes the source ROI into the destination ROI in every plane.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn resize_planar(
        stream_context: &StreamContext,
        resize: &Resize,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_resize_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl ResizePlanar<$planes> for $ty {
            unsafe fn resize_planar(
                stream_context: &StreamContext,
                resize: &Resize,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, resize, source, destination) }
            }
        }
    };
}

impl_resize_planar_dispatch!(u8, 3, resize_u8_p3);

impl_resize_planar_dispatch!(u8, 4, resize_u8_p4);

impl_resize_planar_dispatch!(u16, 3, resize_u16_p3);

impl_resize_planar_dispatch!(u16, 4, resize_u16_p4);

impl_resize_planar_dispatch!(i16, 3, resize_i16_p3);

impl_resize_planar_dispatch!(i16, 4, resize_i16_p4);

impl_resize_planar_dispatch!(f32, 3, resize_f32_p3);

impl_resize_planar_dispatch!(f32, 4, resize_f32_p4);

/// Resizes the source ROI into the destination ROI in every plane.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Each destination ROI must lie within its destination image.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn resize_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    resize: &Resize,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: ResizePlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::resize_planar(stream_context, resize, source, destination) }
}

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`.",
    resize_batch_u8_c1,
    u8,
    C1,
    nppiResizeBatch_8u_C1R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`.",
    resize_batch_u8_c3,
    u8,
    C3,
    nppiResizeBatch_8u_C3R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`.",
    resize_batch_u8_c4,
    u8,
    C4,
    nppiResizeBatch_8u_C4R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_batch_u8_ac4,
    u8,
    AC4,
    nppiResizeBatch_8u_AC4R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`.",
    resize_batch_f32_c1,
    f32,
    C1,
    nppiResizeBatch_32f_C1R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`.",
    resize_batch_f32_c3,
    f32,
    C3,
    nppiResizeBatch_32f_C3R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`.",
    resize_batch_f32_c4,
    f32,
    C4,
    nppiResizeBatch_32f_C4R_Ctx
);

impl_resize_batch!(
    "Resizes image pairs using the common ROIs and interpolation in `resize`. The destination alpha channel is not processed.",
    resize_batch_f32_ac4,
    f32,
    AC4,
    nppiResizeBatch_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`resize_batch`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait ResizeBatchOperation<L: ChannelLayout>: DataTypeLike {
    /// Resizes a batch of images using shared ROIs and interpolation.
    ///
    /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
    /// Native source and destination bounds use the smallest image width and height in each batch.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Batches require at least two entries and a count representable by the native interface.
    /// Source and destination batch lengths must match.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn resize_batch(
        stream_context: &StreamContext,
        resize: &Resize,
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageOutputView<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_resize_batch_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl ResizeBatchOperation<$layout> for $ty {
            unsafe fn resize_batch(
                stream_context: &StreamContext,
                resize: &Resize,
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageOutputView<'_, Self, $layout>],
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, resize, sources, destinations) }
            }
        }
    };
}

impl_resize_batch_operation!(u8, C1, resize_batch_u8_c1);

impl_resize_batch_operation!(u8, C3, resize_batch_u8_c3);

impl_resize_batch_operation!(u8, C4, resize_batch_u8_c4);

impl_resize_batch_operation!(u8, AC4, resize_batch_u8_ac4);

impl_resize_batch_operation!(f32, C1, resize_batch_f32_c1);

impl_resize_batch_operation!(f32, C3, resize_batch_f32_c3);

impl_resize_batch_operation!(f32, C4, resize_batch_f32_c4);

impl_resize_batch_operation!(f32, AC4, resize_batch_f32_ac4);

/// Resizes a batch of images using shared ROIs and interpolation.
///
/// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
/// Native source and destination bounds use the smallest image width and height in each batch.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Each destination ROI must lie within its destination image.
/// Batches require at least two entries and a count representable by the native interface.
/// Source and destination batch lengths must match.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn resize_batch<T, L>(
    stream_context: &StreamContext,
    resize: &Resize,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<()>
where
    T: ResizeBatchOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::resize_batch(stream_context, resize, sources, destinations) }
}

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_u8_c1_advanced,
    u8,
    C1,
    nppiResizeBatch_8u_C1R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_u8_c3_advanced,
    u8,
    C3,
    nppiResizeBatch_8u_C3R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_u8_c4_advanced,
    u8,
    C4,
    nppiResizeBatch_8u_C4R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`. The destination alpha channel is not processed.",
    resize_batch_u8_ac4_advanced,
    u8,
    AC4,
    nppiResizeBatch_8u_AC4R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_f16_c1_advanced,
    f16,
    C1,
    nppiResizeBatch_16f_C1R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_f16_c3_advanced,
    f16,
    C3,
    nppiResizeBatch_16f_C3R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_f16_c4_advanced,
    f16,
    C4,
    nppiResizeBatch_16f_C4R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_f32_c1_advanced,
    f32,
    C1,
    nppiResizeBatch_32f_C1R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_f32_c3_advanced,
    f32,
    C3,
    nppiResizeBatch_32f_C3R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`.",
    resize_batch_f32_c4_advanced,
    f32,
    C4,
    nppiResizeBatch_32f_C4R_Advanced_Ctx
);

impl_resize_batch_advanced!(
    "Resizes image pairs using their individual source and destination ROIs from `rois`. The destination alpha channel is not processed.",
    resize_batch_f32_ac4_advanced,
    f32,
    AC4,
    nppiResizeBatch_32f_AC4R_Advanced_Ctx
);

/// Element and channel-layout dispatch for [`resize_batch_advanced`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait ResizeBatchAdvancedOperation<L: ChannelLayout>: DataTypeLike {
    /// Resizes a batch of images using one ROI pair per image and shared interpolation.
    ///
    /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Batches require at least two entries and a count representable by the native interface.
    /// Source and destination batch lengths must match.
    /// `rois` must contain one ROI pair per image pair.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn resize_batch_advanced(
        stream_context: &StreamContext,
        rois: &[ResizeBatchAdvanced],
        interpolation: InterpolationMode,
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageOutputView<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_resize_batch_advanced_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl ResizeBatchAdvancedOperation<$layout> for $ty {
            unsafe fn resize_batch_advanced(
                stream_context: &StreamContext,
                rois: &[ResizeBatchAdvanced],
                interpolation: InterpolationMode,
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageOutputView<'_, Self, $layout>],
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, rois, interpolation, sources, destinations) }
            }
        }
    };
}

impl_resize_batch_advanced_operation!(u8, C1, resize_batch_u8_c1_advanced);

impl_resize_batch_advanced_operation!(u8, C3, resize_batch_u8_c3_advanced);

impl_resize_batch_advanced_operation!(u8, C4, resize_batch_u8_c4_advanced);

impl_resize_batch_advanced_operation!(u8, AC4, resize_batch_u8_ac4_advanced);

impl_resize_batch_advanced_operation!(f16, C1, resize_batch_f16_c1_advanced);

impl_resize_batch_advanced_operation!(f16, C3, resize_batch_f16_c3_advanced);

impl_resize_batch_advanced_operation!(f16, C4, resize_batch_f16_c4_advanced);

impl_resize_batch_advanced_operation!(f32, C1, resize_batch_f32_c1_advanced);

impl_resize_batch_advanced_operation!(f32, C3, resize_batch_f32_c3_advanced);

impl_resize_batch_advanced_operation!(f32, C4, resize_batch_f32_c4_advanced);

impl_resize_batch_advanced_operation!(f32, AC4, resize_batch_f32_ac4_advanced);

/// Resizes a batch of images using one ROI pair per image and shared interpolation.
///
/// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Each destination ROI must lie within its destination image.
/// Batches require at least two entries and a count representable by the native interface.
/// Source and destination batch lengths must match.
/// `rois` must contain one ROI pair per image pair.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn resize_batch_advanced<T, L>(
    stream_context: &StreamContext,
    rois: &[ResizeBatchAdvanced],
    interpolation: InterpolationMode,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<()>
where
    T: ResizeBatchAdvancedOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::resize_batch_advanced(stream_context, rois, interpolation, sources, destinations) }
}
