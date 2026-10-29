use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry::{Remap, Rotate},
        geometry_helpers::validate_same_size,
        view::{
            AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, PlanarImageOutputView,
            PlanarImageView,
        },
    },
    try_ffi,
    types::DataTypeLike,
};

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_u8_c1,
    u8,
    C1,
    nppiRotate_8u_C1R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_u8_c3,
    u8,
    C3,
    nppiRotate_8u_C3R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_u8_c4,
    u8,
    C4,
    nppiRotate_8u_C4R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts. The destination alpha channel is not processed.",
    rotate_u8_ac4,
    u8,
    AC4,
    nppiRotate_8u_AC4R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_u16_c1,
    u16,
    C1,
    nppiRotate_16u_C1R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_u16_c3,
    u16,
    C3,
    nppiRotate_16u_C3R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_u16_c4,
    u16,
    C4,
    nppiRotate_16u_C4R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts. The destination alpha channel is not processed.",
    rotate_u16_ac4,
    u16,
    AC4,
    nppiRotate_16u_AC4R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_f32_c1,
    f32,
    C1,
    nppiRotate_32f_C1R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_f32_c3,
    f32,
    C3,
    nppiRotate_32f_C3R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts.",
    rotate_f32_c4,
    f32,
    C4,
    nppiRotate_32f_C4R_Ctx
);

impl_rotate!(
    "Rotates the source ROI using `rotate.angle` in degrees, then applies the configured shifts. The destination alpha channel is not processed.",
    rotate_f32_ac4,
    f32,
    AC4,
    nppiRotate_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`rotate`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait RotateOperation<L: ChannelLayout>: DataTypeLike {
    /// Rotates the source ROI using the angle, shifts, and interpolation in `rotate`.
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
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn rotate(
        stream_context: &StreamContext,
        rotate: &Rotate,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_rotate_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl RotateOperation<$layout> for $ty {
            unsafe fn rotate(
                stream_context: &StreamContext,
                rotate: &Rotate,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, rotate, source, destination) }
            }
        }
    };
}

impl_rotate_operation!(u8, C1, rotate_u8_c1);

impl_rotate_operation!(u8, C3, rotate_u8_c3);

impl_rotate_operation!(u8, C4, rotate_u8_c4);

impl_rotate_operation!(u8, AC4, rotate_u8_ac4);

impl_rotate_operation!(u16, C1, rotate_u16_c1);

impl_rotate_operation!(u16, C3, rotate_u16_c3);

impl_rotate_operation!(u16, C4, rotate_u16_c4);

impl_rotate_operation!(u16, AC4, rotate_u16_ac4);

impl_rotate_operation!(f32, C1, rotate_f32_c1);

impl_rotate_operation!(f32, C3, rotate_f32_c3);

impl_rotate_operation!(f32, C4, rotate_f32_c4);

impl_rotate_operation!(f32, AC4, rotate_f32_ac4);

/// Rotates the source ROI using the angle, shifts, and interpolation in `rotate`.
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
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn rotate<T, L>(
    stream_context: &StreamContext,
    rotate: &Rotate,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: RotateOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::rotate(stream_context, rotate, source, destination) }
}

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_u8_c1,
    u8,
    f32,
    C1,
    nppiRemap_8u_C1R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_u8_c3,
    u8,
    f32,
    C3,
    nppiRemap_8u_C3R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_u8_c4,
    u8,
    f32,
    C4,
    nppiRemap_8u_C4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. The destination alpha channel is not processed.",
    remap_u8_ac4,
    u8,
    f32,
    AC4,
    nppiRemap_8u_AC4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_u16_c1,
    u16,
    f32,
    C1,
    nppiRemap_16u_C1R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_u16_c3,
    u16,
    f32,
    C3,
    nppiRemap_16u_C3R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_u16_c4,
    u16,
    f32,
    C4,
    nppiRemap_16u_C4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. The destination alpha channel is not processed.",
    remap_u16_ac4,
    u16,
    f32,
    AC4,
    nppiRemap_16u_AC4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_i16_c1,
    i16,
    f32,
    C1,
    nppiRemap_16s_C1R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_i16_c3,
    i16,
    f32,
    C3,
    nppiRemap_16s_C3R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_i16_c4,
    i16,
    f32,
    C4,
    nppiRemap_16s_C4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. The destination alpha channel is not processed.",
    remap_i16_ac4,
    i16,
    f32,
    AC4,
    nppiRemap_16s_AC4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_f32_c1,
    f32,
    f32,
    C1,
    nppiRemap_32f_C1R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_f32_c3,
    f32,
    f32,
    C3,
    nppiRemap_32f_C3R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_f32_c4,
    f32,
    f32,
    C4,
    nppiRemap_32f_C4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. The destination alpha channel is not processed.",
    remap_f32_ac4,
    f32,
    f32,
    AC4,
    nppiRemap_32f_AC4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_f64_c1,
    f64,
    f64,
    C1,
    nppiRemap_64f_C1R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_f64_c3,
    f64,
    f64,
    C3,
    nppiRemap_64f_C3R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation.",
    remap_f64_c4,
    f64,
    f64,
    C4,
    nppiRemap_64f_C4R_Ctx
);

impl_remap!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. The destination alpha channel is not processed.",
    remap_f64_ac4,
    f64,
    f64,
    AC4,
    nppiRemap_64f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`remap`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
/// `Map` must match the coordinate-map element type expected by that routine.
pub unsafe trait RemapOperation<L: ChannelLayout>: DataTypeLike {
    /// Element type used by the horizontal and vertical coordinate maps.
    type Map: DataTypeLike;

    /// Samples the source at coordinates from `x_map` and `y_map` using the configured interpolation.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Both coordinate maps must have the destination image size.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels and any coordinate maps must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain image and coordinate-map allocations through queued completion, including submission errors and captured graph replay.
    /// Pixel layouts, byte strides, ROI footprints, map coordinates, and address alignment must satisfy the native routine.
    unsafe fn remap(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &ImageView<'_, Self, L>,
        x_map: &ImageView<'_, Self::Map, C1>,
        y_map: &ImageView<'_, Self::Map, C1>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_remap_operation {
    ($ty:ty, $map_ty:ty, $layout:ty, $function:ident) => {
        unsafe impl RemapOperation<$layout> for $ty {
            type Map = $map_ty;

            unsafe fn remap(
                stream_context: &StreamContext,
                remap: &Remap,
                source: &ImageView<'_, Self, $layout>,
                x_map: &ImageView<'_, Self::Map, C1>,
                y_map: &ImageView<'_, Self::Map, C1>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, remap, source, x_map, y_map, destination) }
            }
        }
    };
}

impl_remap_operation!(u8, f32, C1, remap_u8_c1);

impl_remap_operation!(u8, f32, C3, remap_u8_c3);

impl_remap_operation!(u8, f32, C4, remap_u8_c4);

impl_remap_operation!(u8, f32, AC4, remap_u8_ac4);

impl_remap_operation!(u16, f32, C1, remap_u16_c1);

impl_remap_operation!(u16, f32, C3, remap_u16_c3);

impl_remap_operation!(u16, f32, C4, remap_u16_c4);

impl_remap_operation!(u16, f32, AC4, remap_u16_ac4);

impl_remap_operation!(i16, f32, C1, remap_i16_c1);

impl_remap_operation!(i16, f32, C3, remap_i16_c3);

impl_remap_operation!(i16, f32, C4, remap_i16_c4);

impl_remap_operation!(i16, f32, AC4, remap_i16_ac4);

impl_remap_operation!(f32, f32, C1, remap_f32_c1);

impl_remap_operation!(f32, f32, C3, remap_f32_c3);

impl_remap_operation!(f32, f32, C4, remap_f32_c4);

impl_remap_operation!(f32, f32, AC4, remap_f32_ac4);

impl_remap_operation!(f64, f64, C1, remap_f64_c1);

impl_remap_operation!(f64, f64, C3, remap_f64_c3);

impl_remap_operation!(f64, f64, C4, remap_f64_c4);

impl_remap_operation!(f64, f64, AC4, remap_f64_ac4);

/// Samples the source at coordinates from `x_map` and `y_map` using the configured interpolation.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Both coordinate maps must have the destination image size.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels and any coordinate maps must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain image and coordinate-map allocations through queued completion, including submission errors and captured graph replay.
/// Pixel layouts, byte strides, ROI footprints, map coordinates, and address alignment must satisfy the native routine.
pub unsafe fn remap<T, L>(
    stream_context: &StreamContext,
    remap: &Remap,
    source: &ImageView<'_, T, L>,
    x_map: &ImageView<'_, T::Map, C1>,
    y_map: &ImageView<'_, T::Map, C1>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: RemapOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::remap(stream_context, remap, source, x_map, y_map, destination) }
}

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_u8_p3,
    u8,
    f32,
    3,
    nppiRemap_8u_P3R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_u8_p4,
    u8,
    f32,
    4,
    nppiRemap_8u_P4R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_u16_p3,
    u16,
    f32,
    3,
    nppiRemap_16u_P3R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_u16_p4,
    u16,
    f32,
    4,
    nppiRemap_16u_P4R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_i16_p3,
    i16,
    f32,
    3,
    nppiRemap_16s_P3R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_i16_p4,
    i16,
    f32,
    4,
    nppiRemap_16s_P4R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_f32_p3,
    f32,
    f32,
    3,
    nppiRemap_32f_P3R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_f32_p4,
    f32,
    f32,
    4,
    nppiRemap_32f_P4R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_f64_p3,
    f64,
    f64,
    3,
    nppiRemap_64f_P3R_Ctx
);

impl_remap_planar!(
    "Samples the source using coordinates from `x_map` and `y_map` and the configured interpolation. Processes all image planes.",
    remap_f64_p4,
    f64,
    f64,
    4,
    nppiRemap_64f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`remap_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
/// `Map` must match the coordinate-map element type expected by that routine.
pub unsafe trait RemapPlanar<const PLANES: usize>: DataTypeLike {
    /// Element type used by the horizontal and vertical coordinate maps.
    type Map: DataTypeLike;

    /// Samples every source plane at coordinates from `x_map` and `y_map` using the configured interpolation.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Both coordinate maps must have the destination image size.
    /// Source and destination plane sizes must match.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels and any coordinate maps must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain image and coordinate-map allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
    /// Pixel layouts, byte strides, ROI footprints, map coordinates, and address alignment must satisfy the native routine.
    unsafe fn remap_planar(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &PlanarImageView<'_, Self, PLANES>,
        x_map: &ImageView<'_, Self::Map, C1>,
        y_map: &ImageView<'_, Self::Map, C1>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_remap_planar_dispatch {
    ($ty:ty, $map_ty:ty, $planes:literal, $function:ident) => {
        unsafe impl RemapPlanar<$planes> for $ty {
            type Map = $map_ty;

            unsafe fn remap_planar(
                stream_context: &StreamContext,
                remap: &Remap,
                source: &PlanarImageView<'_, Self, $planes>,
                x_map: &ImageView<'_, Self::Map, C1>,
                y_map: &ImageView<'_, Self::Map, C1>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, remap, source, x_map, y_map, destination) }
            }
        }
    };
}

impl_remap_planar_dispatch!(u8, f32, 3, remap_u8_p3);

impl_remap_planar_dispatch!(u8, f32, 4, remap_u8_p4);

impl_remap_planar_dispatch!(u16, f32, 3, remap_u16_p3);

impl_remap_planar_dispatch!(u16, f32, 4, remap_u16_p4);

impl_remap_planar_dispatch!(i16, f32, 3, remap_i16_p3);

impl_remap_planar_dispatch!(i16, f32, 4, remap_i16_p4);

impl_remap_planar_dispatch!(f32, f32, 3, remap_f32_p3);

impl_remap_planar_dispatch!(f32, f32, 4, remap_f32_p4);

impl_remap_planar_dispatch!(f64, f64, 3, remap_f64_p3);

impl_remap_planar_dispatch!(f64, f64, 4, remap_f64_p4);

/// Samples every source plane at coordinates from `x_map` and `y_map` using the configured interpolation.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Both coordinate maps must have the destination image size.
/// Source and destination plane sizes must match.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels and any coordinate maps must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain image and coordinate-map allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
/// Pixel layouts, byte strides, ROI footprints, map coordinates, and address alignment must satisfy the native routine.
pub unsafe fn remap_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    remap: &Remap,
    source: &PlanarImageView<'_, T, PLANES>,
    x_map: &ImageView<'_, T::Map, C1>,
    y_map: &ImageView<'_, T::Map, C1>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: RemapPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::remap_planar(stream_context, remap, source, x_map, y_map, destination) }
}
