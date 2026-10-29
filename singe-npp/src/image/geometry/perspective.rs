use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry::{WarpPerspective, WarpPerspectiveBatch, WarpQuad},
        geometry_helpers::warp_perspective_batch_descriptors,
        view::{
            AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, PlanarImageOutputView,
            PlanarImageView,
        },
    },
    try_ffi,
    types::{DataTypeLike, PerspectiveCoefficients},
};

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_u8_c1,
    u8,
    C1,
    nppiWarpPerspective_8u_C1R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_u8_c3,
    u8,
    C3,
    nppiWarpPerspective_8u_C3R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_u8_c4,
    u8,
    C4,
    nppiWarpPerspective_8u_C4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspective_8u_AC4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_u16_c1,
    u16,
    C1,
    nppiWarpPerspective_16u_C1R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_u16_c3,
    u16,
    C3,
    nppiWarpPerspective_16u_C3R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_u16_c4,
    u16,
    C4,
    nppiWarpPerspective_16u_C4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_u16_ac4,
    u16,
    AC4,
    nppiWarpPerspective_16u_AC4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_i32_c1,
    i32,
    C1,
    nppiWarpPerspective_32s_C1R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_i32_c3,
    i32,
    C3,
    nppiWarpPerspective_32s_C3R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_i32_c4,
    i32,
    C4,
    nppiWarpPerspective_32s_C4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_i32_ac4,
    i32,
    AC4,
    nppiWarpPerspective_32s_AC4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_f16_c1,
    f16,
    C1,
    nppiWarpPerspective_16f_C1R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_f16_c3,
    f16,
    C3,
    nppiWarpPerspective_16f_C3R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_f16_c4,
    f16,
    C4,
    nppiWarpPerspective_16f_C4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_f32_c1,
    f32,
    C1,
    nppiWarpPerspective_32f_C1R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_f32_c3,
    f32,
    C3,
    nppiWarpPerspective_32f_C3R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`.",
    warp_perspective_f32_c4,
    f32,
    C4,
    nppiWarpPerspective_32f_C4R_Ctx
);

impl_warp_perspective!(
    "Applies the perspective transformation and interpolation configured by `warp`. The destination alpha channel is not processed.",
    warp_perspective_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspective_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectiveOperation<L: ChannelLayout>: DataTypeLike {
    /// Applies the perspective transformation and the configured interpolation.
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
    unsafe fn warp_perspective(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl WarpPerspectiveOperation<$layout> for $ty {
            unsafe fn warp_perspective(
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

impl_warp_perspective_operation!(u8, C1, warp_perspective_u8_c1);

impl_warp_perspective_operation!(u8, C3, warp_perspective_u8_c3);

impl_warp_perspective_operation!(u8, C4, warp_perspective_u8_c4);

impl_warp_perspective_operation!(u8, AC4, warp_perspective_u8_ac4);

impl_warp_perspective_operation!(u16, C1, warp_perspective_u16_c1);

impl_warp_perspective_operation!(u16, C3, warp_perspective_u16_c3);

impl_warp_perspective_operation!(u16, C4, warp_perspective_u16_c4);

impl_warp_perspective_operation!(u16, AC4, warp_perspective_u16_ac4);

impl_warp_perspective_operation!(i32, C1, warp_perspective_i32_c1);

impl_warp_perspective_operation!(i32, C3, warp_perspective_i32_c3);

impl_warp_perspective_operation!(i32, C4, warp_perspective_i32_c4);

impl_warp_perspective_operation!(i32, AC4, warp_perspective_i32_ac4);

impl_warp_perspective_operation!(f16, C1, warp_perspective_f16_c1);

impl_warp_perspective_operation!(f16, C3, warp_perspective_f16_c3);

impl_warp_perspective_operation!(f16, C4, warp_perspective_f16_c4);

impl_warp_perspective_operation!(f32, C1, warp_perspective_f32_c1);

impl_warp_perspective_operation!(f32, C3, warp_perspective_f32_c3);

impl_warp_perspective_operation!(f32, C4, warp_perspective_f32_c4);

impl_warp_perspective_operation!(f32, AC4, warp_perspective_f32_ac4);

/// Applies the perspective transformation and the configured interpolation.
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
pub unsafe fn warp_perspective<T, L>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: WarpPerspectiveOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective(stream_context, warp, source, destination) }
}

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_u8_c1,
    u8,
    C1,
    nppiWarpPerspectiveBatch_8u_C1R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_u8_c3,
    u8,
    C3,
    nppiWarpPerspectiveBatch_8u_C3R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_u8_c4,
    u8,
    C4,
    nppiWarpPerspectiveBatch_8u_C4R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair. The destination alpha channel is not processed.",
    warp_perspective_batch_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspectiveBatch_8u_AC4R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_f16_c1,
    f16,
    C1,
    nppiWarpPerspectiveBatch_16f_C1R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_f16_c3,
    f16,
    C3,
    nppiWarpPerspectiveBatch_16f_C3R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_f16_c4,
    f16,
    C4,
    nppiWarpPerspectiveBatch_16f_C4R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_f32_c1,
    f32,
    C1,
    nppiWarpPerspectiveBatch_32f_C1R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_f32_c3,
    f32,
    C3,
    nppiWarpPerspectiveBatch_32f_C3R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair.",
    warp_perspective_batch_f32_c4,
    f32,
    C4,
    nppiWarpPerspectiveBatch_32f_C4R_Ctx
);

impl_warp_perspective_batch!(
    "Applies perspective warps to image pairs using one matrix from `coefficients` per pair. The destination alpha channel is not processed.",
    warp_perspective_batch_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspectiveBatch_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective_batch`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectiveBatchOperation<L: ChannelLayout>: DataTypeLike {
    /// Applies the perspective transformation to a batch using one coefficient matrix per image and the configured interpolation.
    ///
    /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
    /// Native source bounds use the smallest source width and height in the batch.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source ROI dimensions must be positive and their endpoints representable.
    /// Each destination ROI must lie within its destination image.
    /// Batches require at least two entries and a count representable by the native interface.
    /// Source and destination batch lengths must match.
    /// `coefficients` must contain one matrix per image pair.
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
    unsafe fn warp_perspective_batch(
        stream_context: &StreamContext,
        warp: &WarpPerspectiveBatch,
        coefficients: &[PerspectiveCoefficients],
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageOutputView<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_batch_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl WarpPerspectiveBatchOperation<$layout> for $ty {
            unsafe fn warp_perspective_batch(
                stream_context: &StreamContext,
                warp: &WarpPerspectiveBatch,
                coefficients: &[PerspectiveCoefficients],
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageOutputView<'_, Self, $layout>],
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, coefficients, sources, destinations) }
            }
        }
    };
}

impl_warp_perspective_batch_operation!(u8, C1, warp_perspective_batch_u8_c1);

impl_warp_perspective_batch_operation!(u8, C3, warp_perspective_batch_u8_c3);

impl_warp_perspective_batch_operation!(u8, C4, warp_perspective_batch_u8_c4);

impl_warp_perspective_batch_operation!(u8, AC4, warp_perspective_batch_u8_ac4);

impl_warp_perspective_batch_operation!(f16, C1, warp_perspective_batch_f16_c1);

impl_warp_perspective_batch_operation!(f16, C3, warp_perspective_batch_f16_c3);

impl_warp_perspective_batch_operation!(f16, C4, warp_perspective_batch_f16_c4);

impl_warp_perspective_batch_operation!(f32, C1, warp_perspective_batch_f32_c1);

impl_warp_perspective_batch_operation!(f32, C3, warp_perspective_batch_f32_c3);

impl_warp_perspective_batch_operation!(f32, C4, warp_perspective_batch_f32_c4);

impl_warp_perspective_batch_operation!(f32, AC4, warp_perspective_batch_f32_ac4);

/// Applies the perspective transformation to a batch using one coefficient matrix per image and the configured interpolation.
///
/// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
/// Native source bounds use the smallest source width and height in the batch.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source ROI dimensions must be positive and their endpoints representable.
/// Each destination ROI must lie within its destination image.
/// Batches require at least two entries and a count representable by the native interface.
/// Source and destination batch lengths must match.
/// `coefficients` must contain one matrix per image pair.
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
pub unsafe fn warp_perspective_batch<T, L>(
    stream_context: &StreamContext,
    warp: &WarpPerspectiveBatch,
    coefficients: &[PerspectiveCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<()>
where
    T: WarpPerspectiveBatchOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective_batch(stream_context, warp, coefficients, sources, destinations) }
}

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_u8_p3,
    u8,
    3,
    nppiWarpPerspective_8u_P3R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_u8_p4,
    u8,
    4,
    nppiWarpPerspective_8u_P4R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_u16_p3,
    u16,
    3,
    nppiWarpPerspective_16u_P3R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_u16_p4,
    u16,
    4,
    nppiWarpPerspective_16u_P4R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_i32_p3,
    i32,
    3,
    nppiWarpPerspective_32s_P3R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_i32_p4,
    i32,
    4,
    nppiWarpPerspective_32s_P4R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_f32_p3,
    f32,
    3,
    nppiWarpPerspective_32f_P3R_Ctx
);

impl_warp_perspective_planar!(
    "Applies the perspective transformation and interpolation configured by `warp`. Processes all image planes.",
    warp_perspective_f32_p4,
    f32,
    4,
    nppiWarpPerspective_32f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectivePlanar<const PLANES: usize>: DataTypeLike {
    /// Applies the perspective transformation to every image plane and the configured interpolation.
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
    unsafe fn warp_perspective_planar(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl WarpPerspectivePlanar<$planes> for $ty {
            unsafe fn warp_perspective_planar(
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

impl_warp_perspective_planar_dispatch!(u8, 3, warp_perspective_u8_p3);

impl_warp_perspective_planar_dispatch!(u8, 4, warp_perspective_u8_p4);

impl_warp_perspective_planar_dispatch!(u16, 3, warp_perspective_u16_p3);

impl_warp_perspective_planar_dispatch!(u16, 4, warp_perspective_u16_p4);

impl_warp_perspective_planar_dispatch!(i32, 3, warp_perspective_i32_p3);

impl_warp_perspective_planar_dispatch!(i32, 4, warp_perspective_i32_p4);

impl_warp_perspective_planar_dispatch!(f32, 3, warp_perspective_f32_p3);

impl_warp_perspective_planar_dispatch!(f32, 4, warp_perspective_f32_p4);

/// Applies the perspective transformation to every image plane and the configured interpolation.
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
pub unsafe fn warp_perspective_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpPerspectivePlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective_planar(stream_context, warp, source, destination) }
}

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_u8_c1,
    u8,
    C1,
    nppiWarpPerspectiveQuad_8u_C1R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_u8_c3,
    u8,
    C3,
    nppiWarpPerspectiveQuad_8u_C3R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_u8_c4,
    u8,
    C4,
    nppiWarpPerspectiveQuad_8u_C4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. The destination alpha channel is not processed.",
    warp_perspective_quad_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspectiveQuad_8u_AC4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_u16_c1,
    u16,
    C1,
    nppiWarpPerspectiveQuad_16u_C1R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_u16_c3,
    u16,
    C3,
    nppiWarpPerspectiveQuad_16u_C3R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_u16_c4,
    u16,
    C4,
    nppiWarpPerspectiveQuad_16u_C4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. The destination alpha channel is not processed.",
    warp_perspective_quad_u16_ac4,
    u16,
    AC4,
    nppiWarpPerspectiveQuad_16u_AC4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_i32_c1,
    i32,
    C1,
    nppiWarpPerspectiveQuad_32s_C1R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_i32_c3,
    i32,
    C3,
    nppiWarpPerspectiveQuad_32s_C3R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_i32_c4,
    i32,
    C4,
    nppiWarpPerspectiveQuad_32s_C4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. The destination alpha channel is not processed.",
    warp_perspective_quad_i32_ac4,
    i32,
    AC4,
    nppiWarpPerspectiveQuad_32s_AC4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_f32_c1,
    f32,
    C1,
    nppiWarpPerspectiveQuad_32f_C1R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_f32_c3,
    f32,
    C3,
    nppiWarpPerspectiveQuad_32f_C3R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`.",
    warp_perspective_quad_f32_c4,
    f32,
    C4,
    nppiWarpPerspectiveQuad_32f_C4R_Ctx
);

impl_warp_perspective_quad!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. The destination alpha channel is not processed.",
    warp_perspective_quad_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspectiveQuad_32f_AC4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective_quad`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectiveQuadOperation<L: ChannelLayout>: DataTypeLike {
    /// Applies the perspective transformation defined by the source and destination quadrangles and the configured interpolation.
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
    unsafe fn warp_perspective_quad(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_quad_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl WarpPerspectiveQuadOperation<$layout> for $ty {
            unsafe fn warp_perspective_quad(
                stream_context: &StreamContext,
                warp: &WarpQuad,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, source, destination) }
            }
        }
    };
}

impl_warp_perspective_quad_operation!(u8, C1, warp_perspective_quad_u8_c1);

impl_warp_perspective_quad_operation!(u8, C3, warp_perspective_quad_u8_c3);

impl_warp_perspective_quad_operation!(u8, C4, warp_perspective_quad_u8_c4);

impl_warp_perspective_quad_operation!(u8, AC4, warp_perspective_quad_u8_ac4);

impl_warp_perspective_quad_operation!(u16, C1, warp_perspective_quad_u16_c1);

impl_warp_perspective_quad_operation!(u16, C3, warp_perspective_quad_u16_c3);

impl_warp_perspective_quad_operation!(u16, C4, warp_perspective_quad_u16_c4);

impl_warp_perspective_quad_operation!(u16, AC4, warp_perspective_quad_u16_ac4);

impl_warp_perspective_quad_operation!(i32, C1, warp_perspective_quad_i32_c1);

impl_warp_perspective_quad_operation!(i32, C3, warp_perspective_quad_i32_c3);

impl_warp_perspective_quad_operation!(i32, C4, warp_perspective_quad_i32_c4);

impl_warp_perspective_quad_operation!(i32, AC4, warp_perspective_quad_i32_ac4);

impl_warp_perspective_quad_operation!(f32, C1, warp_perspective_quad_f32_c1);

impl_warp_perspective_quad_operation!(f32, C3, warp_perspective_quad_f32_c3);

impl_warp_perspective_quad_operation!(f32, C4, warp_perspective_quad_f32_c4);

impl_warp_perspective_quad_operation!(f32, AC4, warp_perspective_quad_f32_ac4);

/// Applies the perspective transformation defined by the source and destination quadrangles and the configured interpolation.
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
pub unsafe fn warp_perspective_quad<T, L>(
    stream_context: &StreamContext,
    warp: &WarpQuad,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
) -> Result<()>
where
    T: WarpPerspectiveQuadOperation<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective_quad(stream_context, warp, source, destination) }
}

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_u8_p3,
    u8,
    3,
    nppiWarpPerspectiveQuad_8u_P3R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_u8_p4,
    u8,
    4,
    nppiWarpPerspectiveQuad_8u_P4R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_u16_p3,
    u16,
    3,
    nppiWarpPerspectiveQuad_16u_P3R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_u16_p4,
    u16,
    4,
    nppiWarpPerspectiveQuad_16u_P4R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_i32_p3,
    i32,
    3,
    nppiWarpPerspectiveQuad_32s_P3R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_i32_p4,
    i32,
    4,
    nppiWarpPerspectiveQuad_32s_P4R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_f32_p3,
    f32,
    3,
    nppiWarpPerspectiveQuad_32f_P3R_Ctx
);

impl_warp_perspective_quad_planar!(
    "Applies a perspective warp defined by the source and destination quadrangles in `warp`. Processes all image planes.",
    warp_perspective_quad_f32_p4,
    f32,
    4,
    nppiWarpPerspectiveQuad_32f_P4R_Ctx
);

/// Element and channel-layout dispatch for [`warp_perspective_quad_planar`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait WarpPerspectiveQuadPlanar<const PLANES: usize>: DataTypeLike {
    /// Applies the perspective transformation defined by the source and destination quadrangles to every image plane and the configured interpolation.
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
    unsafe fn warp_perspective_quad_planar(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_quad_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl WarpPerspectiveQuadPlanar<$planes> for $ty {
            unsafe fn warp_perspective_quad_planar(
                stream_context: &StreamContext,
                warp: &WarpQuad,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, warp, source, destination) }
            }
        }
    };
}

impl_warp_perspective_quad_planar_dispatch!(u8, 3, warp_perspective_quad_u8_p3);

impl_warp_perspective_quad_planar_dispatch!(u8, 4, warp_perspective_quad_u8_p4);

impl_warp_perspective_quad_planar_dispatch!(u16, 3, warp_perspective_quad_u16_p3);

impl_warp_perspective_quad_planar_dispatch!(u16, 4, warp_perspective_quad_u16_p4);

impl_warp_perspective_quad_planar_dispatch!(i32, 3, warp_perspective_quad_i32_p3);

impl_warp_perspective_quad_planar_dispatch!(i32, 4, warp_perspective_quad_i32_p4);

impl_warp_perspective_quad_planar_dispatch!(f32, 3, warp_perspective_quad_f32_p3);

impl_warp_perspective_quad_planar_dispatch!(f32, 4, warp_perspective_quad_f32_p4);

/// Applies the perspective transformation defined by the source and destination quadrangles to every image plane and the configured interpolation.
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
pub unsafe fn warp_perspective_quad_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpQuad,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpPerspectiveQuadPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::warp_perspective_quad_planar(stream_context, warp, source, destination) }
}
