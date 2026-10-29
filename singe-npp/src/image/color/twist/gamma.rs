use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::validate_same_size,
        view::{
            AC4, C3, ImageOutputView, ImageView, ImageViewMut, PlanarImageOutputView,
            PlanarImageView, PlanarImageViewMut,
        },
    },
    try_ffi,
    types::DataTypeLike,
};

impl_gamma_packed!(
    "3 channel 8-bit unsigned packed color not in place forward gamma correction.",
    gamma_forward_u8_c3,
    C3,
    nppiGammaFwd_8u_C3R_Ctx
);

impl_gamma_packed_in_place!(
    "3 channel 8-bit unsigned packed color in place forward gamma correction.",
    gamma_forward_u8_c3_in_place,
    C3,
    nppiGammaFwd_8u_C3IR_Ctx
);

impl_gamma_packed!(
    "4 channel 8-bit unsigned packed color with alpha not in place forward gamma correction.",
    gamma_forward_u8_ac4,
    AC4,
    nppiGammaFwd_8u_AC4R_Ctx
);

impl_gamma_packed_in_place!(
    "4 channel 8-bit unsigned packed color with alpha in place forward gamma correction.",
    gamma_forward_u8_ac4_in_place,
    AC4,
    nppiGammaFwd_8u_AC4IR_Ctx
);

impl_gamma_planar!(
    "3 channel 8-bit unsigned planar color not in place forward gamma correction.",
    gamma_forward_u8_p3,
    nppiGammaFwd_8u_P3R_Ctx
);

impl_gamma_planar_in_place!(
    "3 channel 8-bit unsigned planar color in place forward gamma correction.",
    gamma_forward_u8_p3_in_place,
    nppiGammaFwd_8u_IP3R_Ctx
);

impl_gamma_packed!(
    "3 channel 8-bit unsigned packed color not in place inverse gamma correction.",
    gamma_inverse_u8_c3,
    C3,
    nppiGammaInv_8u_C3R_Ctx
);

impl_gamma_packed_in_place!(
    "3 channel 8-bit unsigned packed color in place inverse gamma correction.",
    gamma_inverse_u8_c3_in_place,
    C3,
    nppiGammaInv_8u_C3IR_Ctx
);

impl_gamma_packed!(
    "4 channel 8-bit unsigned packed color with alpha not in place inverse gamma correction.",
    gamma_inverse_u8_ac4,
    AC4,
    nppiGammaInv_8u_AC4R_Ctx
);

impl_gamma_packed_in_place!(
    "4 channel 8-bit unsigned packed color with alpha in place inverse gamma correction.",
    gamma_inverse_u8_ac4_in_place,
    AC4,
    nppiGammaInv_8u_AC4IR_Ctx
);

impl_gamma_planar!(
    "3 channel 8-bit unsigned planar color not in place inverse gamma correction.",
    gamma_inverse_u8_p3,
    nppiGammaInv_8u_P3R_Ctx
);

impl_gamma_planar_in_place!(
    "3 channel 8-bit unsigned planar color in place inverse gamma correction.",
    gamma_inverse_u8_p3_in_place,
    nppiGammaInv_8u_IP3R_Ctx
);

impl_generic_gamma!(
    "Applies forward gamma correction to the image.",
GammaForwardC3, gamma_forward, gamma_forward_c3, C3, [
    u8 => gamma_forward_u8_c3
]
);

impl_generic_gamma_in_place!(
    "Applies forward gamma correction to the image. Updates the image in place.",
GammaForwardC3InPlace,
    gamma_forward_in_place,
    gamma_forward_c3_in_place,
    C3,
    [u8 => gamma_forward_u8_c3_in_place]
);

impl_generic_gamma!(
    "Applies forward gamma correction to the image. The alpha channel is not processed.",
GammaForwardAc4, gamma_forward, gamma_forward_ac4, AC4, [
    u8 => gamma_forward_u8_ac4
]
);

impl_generic_gamma_in_place!(
    "Applies forward gamma correction to the image. Updates the image in place. The alpha channel is not processed.",
GammaForwardAc4InPlace,
    gamma_forward_in_place,
    gamma_forward_ac4_in_place,
    AC4,
    [u8 => gamma_forward_u8_ac4_in_place]
);

impl_generic_gamma!(
    "Applies inverse gamma correction to the image.",
GammaInverseC3, gamma_inverse, gamma_inverse_c3, C3, [
    u8 => gamma_inverse_u8_c3
]
);

impl_generic_gamma_in_place!(
    "Applies inverse gamma correction to the image. Updates the image in place.",
GammaInverseC3InPlace,
    gamma_inverse_in_place,
    gamma_inverse_c3_in_place,
    C3,
    [u8 => gamma_inverse_u8_c3_in_place]
);

impl_generic_gamma!(
    "Applies inverse gamma correction to the image. The alpha channel is not processed.",
GammaInverseAc4, gamma_inverse, gamma_inverse_ac4, AC4, [
    u8 => gamma_inverse_u8_ac4
]
);

impl_generic_gamma_in_place!(
    "Applies inverse gamma correction to the image. Updates the image in place. The alpha channel is not processed.",
GammaInverseAc4InPlace,
    gamma_inverse_in_place,
    gamma_inverse_ac4_in_place,
    AC4,
    [u8 => gamma_inverse_u8_ac4_in_place]
);

/// Dispatches forward gamma correction for a native planar element type and plane count.
///
/// # Safety
///
/// Implementations must select a native operation with the matching element ABI and plane count.
/// Uphold the method's validation, gamma semantics, and raw execution contract.
pub unsafe trait GammaForwardPlanar<const PLANES: usize>: DataTypeLike {
    /// Applies forward gamma correction to the planar image.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gamma_forward_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

unsafe impl GammaForwardPlanar<3> for u8 {
    unsafe fn gamma_forward_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, 3>,
        destination: &mut PlanarImageOutputView<'_, Self, 3>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { gamma_forward_u8_p3(stream_context, source, destination) }
    }
}

/// Applies forward gamma correction to the planar image.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Each source and destination image pair must have matching sizes.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gamma_forward_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaForwardPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::gamma_forward_planar(stream_context, source, destination) }
}

/// Dispatches forward gamma correction for a native planar element type and plane count.
///
/// # Safety
///
/// Implementations must select a native operation with the matching element ABI and plane count.
/// Uphold the method's validation, gamma semantics, and raw execution contract.
pub unsafe trait GammaForwardPlanarInPlace<const PLANES: usize>: DataTypeLike {
    /// Applies forward gamma correction to the planar image in place.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gamma_forward_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

unsafe impl GammaForwardPlanarInPlace<3> for u8 {
    unsafe fn gamma_forward_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, 3>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { gamma_forward_u8_p3_in_place(stream_context, image) }
    }
}

/// Applies forward gamma correction to the planar image in place.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gamma_forward_planar_in_place<T, const PLANES: usize>(
    stream_context: &StreamContext,
    image: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaForwardPlanarInPlace<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::gamma_forward_planar_in_place(stream_context, image) }
}

/// Dispatches inverse gamma correction for a native planar element type and plane count.
///
/// # Safety
///
/// Implementations must select a native operation with the matching element ABI and plane count.
/// Uphold the method's validation, gamma semantics, and raw execution contract.
pub unsafe trait GammaInversePlanar<const PLANES: usize>: DataTypeLike {
    /// Applies inverse gamma correction to the planar image.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gamma_inverse_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
    ) -> Result<()>;
}

unsafe impl GammaInversePlanar<3> for u8 {
    unsafe fn gamma_inverse_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, 3>,
        destination: &mut PlanarImageOutputView<'_, Self, 3>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { gamma_inverse_u8_p3(stream_context, source, destination) }
    }
}

/// Applies inverse gamma correction to the planar image.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Each source and destination image pair must have matching sizes.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gamma_inverse_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaInversePlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::gamma_inverse_planar(stream_context, source, destination) }
}

/// Dispatches inverse gamma correction for a native planar element type and plane count.
///
/// # Safety
///
/// Implementations must select a native operation with the matching element ABI and plane count.
/// Uphold the method's validation, gamma semantics, and raw execution contract.
pub unsafe trait GammaInversePlanarInPlace<const PLANES: usize>: DataTypeLike {
    /// Applies inverse gamma correction to the planar image in place.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gamma_inverse_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

unsafe impl GammaInversePlanarInPlace<3> for u8 {
    unsafe fn gamma_inverse_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, 3>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { gamma_inverse_u8_p3_in_place(stream_context, image) }
    }
}

/// Applies inverse gamma correction to the planar image in place.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gamma_inverse_planar_in_place<T, const PLANES: usize>(
    stream_context: &StreamContext,
    image: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaInversePlanarInPlace<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::gamma_inverse_planar_in_place(stream_context, image) }
}
