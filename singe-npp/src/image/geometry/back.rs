use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry::{WarpAffine, WarpPerspective},
        geometry_helpers::validate_same_size,
        view::{
            AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, PlanarImageOutputView,
            PlanarImageView,
        },
    },
    try_ffi,
    types::DataTypeLike,
};

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_u8_c1,
    u8,
    C1,
    nppiWarpPerspectiveBack_8u_C1R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_u8_c3,
    u8,
    C3,
    nppiWarpPerspectiveBack_8u_C3R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_u8_c4,
    u8,
    C4,
    nppiWarpPerspectiveBack_8u_C4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_back_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspectiveBack_8u_AC4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_u16_c1,
    u16,
    C1,
    nppiWarpPerspectiveBack_16u_C1R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_u16_c3,
    u16,
    C3,
    nppiWarpPerspectiveBack_16u_C3R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_u16_c4,
    u16,
    C4,
    nppiWarpPerspectiveBack_16u_C4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_back_u16_ac4,
    u16,
    AC4,
    nppiWarpPerspectiveBack_16u_AC4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_i32_c1,
    i32,
    C1,
    nppiWarpPerspectiveBack_32s_C1R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_i32_c3,
    i32,
    C3,
    nppiWarpPerspectiveBack_32s_C3R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_i32_c4,
    i32,
    C4,
    nppiWarpPerspectiveBack_32s_C4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_back_i32_ac4,
    i32,
    AC4,
    nppiWarpPerspectiveBack_32s_AC4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_f32_c1,
    f32,
    C1,
    nppiWarpPerspectiveBack_32f_C1R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_f32_c3,
    f32,
    C3,
    nppiWarpPerspectiveBack_32f_C3R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`.",
    warp_perspective_back_f32_c4,
    f32,
    C4,
    nppiWarpPerspectiveBack_32f_C4R_Ctx
);

impl_warp_perspective_back!(
    "Applies the NPP backward perspective warp configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_back_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspectiveBack_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective_back`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectiveBackOperation<L: ChannelLayout>: DataTypeLike {
    /// Applies the backward perspective transformation and the configured interpolation.
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
    unsafe fn warp_perspective_back(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_back_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl WarpPerspectiveBackOperation<$layout> for $ty {
            unsafe fn warp_perspective_back(
                stream_context: &StreamContext,
                warp: &WarpPerspective,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, source, destination) }
            }
        }
    };
}

impl_warp_perspective_back_operation!(u8, C1, warp_perspective_back_u8_c1);

impl_warp_perspective_back_operation!(u8, C3, warp_perspective_back_u8_c3);

impl_warp_perspective_back_operation!(u8, C4, warp_perspective_back_u8_c4);

impl_warp_perspective_back_operation!(u8, AC4, warp_perspective_back_u8_ac4);

impl_warp_perspective_back_operation!(u16, C1, warp_perspective_back_u16_c1);

impl_warp_perspective_back_operation!(u16, C3, warp_perspective_back_u16_c3);

impl_warp_perspective_back_operation!(u16, C4, warp_perspective_back_u16_c4);

impl_warp_perspective_back_operation!(u16, AC4, warp_perspective_back_u16_ac4);

impl_warp_perspective_back_operation!(i32, C1, warp_perspective_back_i32_c1);

impl_warp_perspective_back_operation!(i32, C3, warp_perspective_back_i32_c3);

impl_warp_perspective_back_operation!(i32, C4, warp_perspective_back_i32_c4);

impl_warp_perspective_back_operation!(i32, AC4, warp_perspective_back_i32_ac4);

impl_warp_perspective_back_operation!(f32, C1, warp_perspective_back_f32_c1);

impl_warp_perspective_back_operation!(f32, C3, warp_perspective_back_f32_c3);

impl_warp_perspective_back_operation!(f32, C4, warp_perspective_back_f32_c4);

impl_warp_perspective_back_operation!(f32, AC4, warp_perspective_back_f32_ac4);

/// Applies the backward perspective transformation and the configured interpolation.
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
pub unsafe fn warp_perspective_back<T, L>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: WarpPerspectiveBackOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective_back(stream_context, warp, source, destination) }
}

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_u8_p3,
    u8,
    3,
    nppiWarpPerspectiveBack_8u_P3R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_u8_p4,
    u8,
    4,
    nppiWarpPerspectiveBack_8u_P4R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_u16_p3,
    u16,
    3,
    nppiWarpPerspectiveBack_16u_P3R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_u16_p4,
    u16,
    4,
    nppiWarpPerspectiveBack_16u_P4R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_i32_p3,
    i32,
    3,
    nppiWarpPerspectiveBack_32s_P3R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_i32_p4,
    i32,
    4,
    nppiWarpPerspectiveBack_32s_P4R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_f32_p3,
    f32,
    3,
    nppiWarpPerspectiveBack_32f_P3R_Ctx
);

impl_warp_perspective_back_planar!(
    "Applies the NPP backward perspective warp configured by `warp`. Processes all image planes.",
    warp_perspective_back_f32_p4,
    f32,
    4,
    nppiWarpPerspectiveBack_32f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective_back_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectiveBackPlanar<const PLANES: usize>: DataTypeLike {
    /// Applies the backward perspective transformation to every image plane and the configured interpolation.
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
    unsafe fn warp_perspective_back_planar(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_back_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl WarpPerspectiveBackPlanar<$planes> for $ty {
            unsafe fn warp_perspective_back_planar(
                stream_context: &StreamContext,
                warp: &WarpPerspective,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, source, destination) }
            }
        }
    };
}

impl_warp_perspective_back_planar_dispatch!(u8, 3, warp_perspective_back_u8_p3);

impl_warp_perspective_back_planar_dispatch!(u8, 4, warp_perspective_back_u8_p4);

impl_warp_perspective_back_planar_dispatch!(u16, 3, warp_perspective_back_u16_p3);

impl_warp_perspective_back_planar_dispatch!(u16, 4, warp_perspective_back_u16_p4);

impl_warp_perspective_back_planar_dispatch!(i32, 3, warp_perspective_back_i32_p3);

impl_warp_perspective_back_planar_dispatch!(i32, 4, warp_perspective_back_i32_p4);

impl_warp_perspective_back_planar_dispatch!(f32, 3, warp_perspective_back_f32_p3);

impl_warp_perspective_back_planar_dispatch!(f32, 4, warp_perspective_back_f32_p4);

/// Applies the backward perspective transformation to every image plane and the configured interpolation.
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
pub unsafe fn warp_perspective_back_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpPerspectiveBackPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective_back_planar(stream_context, warp, source, destination) }
}

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_u8_c1,
    u8,
    C1,
    nppiWarpAffineBack_8u_C1R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_u8_c3,
    u8,
    C3,
    nppiWarpAffineBack_8u_C3R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_u8_c4,
    u8,
    C4,
    nppiWarpAffineBack_8u_C4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`. The destination alpha channel is not processed.",
    warp_affine_back_u8_ac4,
    u8,
    AC4,
    nppiWarpAffineBack_8u_AC4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_u16_c1,
    u16,
    C1,
    nppiWarpAffineBack_16u_C1R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_u16_c3,
    u16,
    C3,
    nppiWarpAffineBack_16u_C3R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_u16_c4,
    u16,
    C4,
    nppiWarpAffineBack_16u_C4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`. The destination alpha channel is not processed.",
    warp_affine_back_u16_ac4,
    u16,
    AC4,
    nppiWarpAffineBack_16u_AC4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_i32_c1,
    i32,
    C1,
    nppiWarpAffineBack_32s_C1R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_i32_c3,
    i32,
    C3,
    nppiWarpAffineBack_32s_C3R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_i32_c4,
    i32,
    C4,
    nppiWarpAffineBack_32s_C4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`. The destination alpha channel is not processed.",
    warp_affine_back_i32_ac4,
    i32,
    AC4,
    nppiWarpAffineBack_32s_AC4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_f32_c1,
    f32,
    C1,
    nppiWarpAffineBack_32f_C1R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_f32_c3,
    f32,
    C3,
    nppiWarpAffineBack_32f_C3R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`.",
    warp_affine_back_f32_c4,
    f32,
    C4,
    nppiWarpAffineBack_32f_C4R_Ctx
);

impl_warp_affine_back!(
    "Applies the NPP backward affine warp configured by `warp`. The destination alpha channel is not processed.",
    warp_affine_back_f32_ac4,
    f32,
    AC4,
    nppiWarpAffineBack_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_affine_back`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpAffineBackOperation<L: ChannelLayout>: DataTypeLike {
    /// Applies the backward affine transformation and the configured interpolation.
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
    unsafe fn warp_affine_back(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_back_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl WarpAffineBackOperation<$layout> for $ty {
            unsafe fn warp_affine_back(
                stream_context: &StreamContext,
                warp: &WarpAffine,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, source, destination) }
            }
        }
    };
}

impl_warp_affine_back_operation!(u8, C1, warp_affine_back_u8_c1);

impl_warp_affine_back_operation!(u8, C3, warp_affine_back_u8_c3);

impl_warp_affine_back_operation!(u8, C4, warp_affine_back_u8_c4);

impl_warp_affine_back_operation!(u8, AC4, warp_affine_back_u8_ac4);

impl_warp_affine_back_operation!(u16, C1, warp_affine_back_u16_c1);

impl_warp_affine_back_operation!(u16, C3, warp_affine_back_u16_c3);

impl_warp_affine_back_operation!(u16, C4, warp_affine_back_u16_c4);

impl_warp_affine_back_operation!(u16, AC4, warp_affine_back_u16_ac4);

impl_warp_affine_back_operation!(i32, C1, warp_affine_back_i32_c1);

impl_warp_affine_back_operation!(i32, C3, warp_affine_back_i32_c3);

impl_warp_affine_back_operation!(i32, C4, warp_affine_back_i32_c4);

impl_warp_affine_back_operation!(i32, AC4, warp_affine_back_i32_ac4);

impl_warp_affine_back_operation!(f32, C1, warp_affine_back_f32_c1);

impl_warp_affine_back_operation!(f32, C3, warp_affine_back_f32_c3);

impl_warp_affine_back_operation!(f32, C4, warp_affine_back_f32_c4);

impl_warp_affine_back_operation!(f32, AC4, warp_affine_back_f32_ac4);

/// Applies the backward affine transformation and the configured interpolation.
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
pub unsafe fn warp_affine_back<T, L>(
    stream_context: &StreamContext,
    warp: &WarpAffine,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: WarpAffineBackOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_affine_back(stream_context, warp, source, destination) }
}

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_u8_p3,
    u8,
    3,
    nppiWarpAffineBack_8u_P3R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_u8_p4,
    u8,
    4,
    nppiWarpAffineBack_8u_P4R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_u16_p3,
    u16,
    3,
    nppiWarpAffineBack_16u_P3R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_u16_p4,
    u16,
    4,
    nppiWarpAffineBack_16u_P4R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_i32_p3,
    i32,
    3,
    nppiWarpAffineBack_32s_P3R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_i32_p4,
    i32,
    4,
    nppiWarpAffineBack_32s_P4R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_f32_p3,
    f32,
    3,
    nppiWarpAffineBack_32f_P3R_Ctx
);

impl_warp_affine_back_planar!(
    "Applies the NPP backward affine warp configured by `warp`. Processes all image planes.",
    warp_affine_back_f32_p4,
    f32,
    4,
    nppiWarpAffineBack_32f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_affine_back_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpAffineBackPlanar<const PLANES: usize>: DataTypeLike {
    /// Applies the backward affine transformation to every image plane and the configured interpolation.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Source and destination plane sizes must match.
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
    unsafe fn warp_affine_back_planar(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_back_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl WarpAffineBackPlanar<$planes> for $ty {
            unsafe fn warp_affine_back_planar(
                stream_context: &StreamContext,
                warp: &WarpAffine,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, source, destination) }
            }
        }
    };
}

impl_warp_affine_back_planar_dispatch!(u8, 3, warp_affine_back_u8_p3);

impl_warp_affine_back_planar_dispatch!(u8, 4, warp_affine_back_u8_p4);

impl_warp_affine_back_planar_dispatch!(u16, 3, warp_affine_back_u16_p3);

impl_warp_affine_back_planar_dispatch!(u16, 4, warp_affine_back_u16_p4);

impl_warp_affine_back_planar_dispatch!(i32, 3, warp_affine_back_i32_p3);

impl_warp_affine_back_planar_dispatch!(i32, 4, warp_affine_back_i32_p4);

impl_warp_affine_back_planar_dispatch!(f32, 3, warp_affine_back_f32_p3);

impl_warp_affine_back_planar_dispatch!(f32, 4, warp_affine_back_f32_p4);

/// Applies the backward affine transformation to every image plane and the configured interpolation.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Each destination ROI must lie within its destination image.
/// Source and destination plane sizes must match.
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
pub unsafe fn warp_affine_back_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpAffine,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpAffineBackPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_affine_back_planar(stream_context, warp, source, destination) }
}
