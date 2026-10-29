use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color::{ColorTwistConstants4, ColorTwistMatrix, ColorTwistMatrix4},
        color_descriptors::validate_same_size,
        view::{
            AC4, C1, C2, C3, C4, ImageOutputView, ImageView, ImageViewMut, PlanarImageOutputView,
            PlanarImageView, PlanarImageViewMut,
        },
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_twist!(
    "1 channel 32-bit floating point color twist.",
    color_twist_f32_c1,
    f32,
    C1,
    nppiColorTwist_32f_C1R_Ctx
);

impl_color_twist_in_place!(
    "1 channel 32-bit floating point in place color twist.",
    color_twist_f32_c1_in_place,
    f32,
    C1,
    nppiColorTwist_32f_C1IR_Ctx
);

impl_color_twist!(
    "2 channel 32-bit floating point color twist.",
    color_twist_f32_c2,
    f32,
    C2,
    nppiColorTwist_32f_C2R_Ctx
);

impl_color_twist_in_place!(
    "2 channel 32-bit floating point in place color twist.",
    color_twist_f32_c2_in_place,
    f32,
    C2,
    nppiColorTwist_32f_C2IR_Ctx
);

impl_color_twist!(
    "3 channel 32-bit floating point color twist.",
    color_twist_f32_c3,
    f32,
    C3,
    nppiColorTwist_32f_C3R_Ctx
);

impl_color_twist_in_place!(
    "3 channel 32-bit floating point in place color twist.",
    color_twist_f32_c3_in_place,
    f32,
    C3,
    nppiColorTwist_32f_C3IR_Ctx
);

impl_color_twist!(
    "4 channel 32-bit floating point color twist, with alpha copy.",
    color_twist_f32_c4,
    f32,
    C4,
    nppiColorTwist_32f_C4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 32-bit floating point in place color twist, not affecting Alpha.",
    color_twist_f32_c4_in_place,
    f32,
    C4,
    nppiColorTwist_32f_C4IR_Ctx
);

impl_color_twist!(
    "4 channel 32-bit floating point color twist, not affecting Alpha.",
    color_twist_f32_ac4,
    f32,
    AC4,
    nppiColorTwist_32f_AC4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 32-bit floating point in place color twist, not affecting Alpha.",
    color_twist_f32_ac4_in_place,
    f32,
    AC4,
    nppiColorTwist_32f_AC4IR_Ctx
);

impl_color_twist_with_constants!(
    "4 channel 32-bit floating point color twist with 4x4 matrix and constant vector addition.",
    color_twist_f32_c4_with_constants,
    f32,
    nppiColorTwist_32fC_C4R_Ctx
);

impl_color_twist_with_constants_in_place!(
    "4 channel 32-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.",
    color_twist_f32_c4_with_constants_in_place,
    f32,
    nppiColorTwist_32fC_C4IR_Ctx
);

impl_color_twist_planar!(
    "3 channel 32-bit floating point planar color twist.",
    color_twist_f32_p3,
    f32,
    3,
    nppiColorTwist_32f_P3R_Ctx
);

impl_color_twist_planar_in_place!(
    "3 channel 32-bit floating point planar in place color twist.",
    color_twist_f32_p3_in_place,
    f32,
    3,
    nppiColorTwist_32f_IP3R_Ctx
);

impl_color_twist!(
    "1 channel 16-bit floating point color twist.",
    color_twist_f16_c1,
    f16,
    C1,
    nppiColorTwist32f_16f_C1R_Ctx
);

impl_color_twist_in_place!(
    "1 channel 16-bit floating point in place color twist.",
    color_twist_f16_c1_in_place,
    f16,
    C1,
    nppiColorTwist32f_16f_C1IR_Ctx
);

impl_color_twist!(
    "2 channel 16-bit floating point color twist.",
    color_twist_f16_c2,
    f16,
    C2,
    nppiColorTwist32f_16f_C2R_Ctx
);

impl_color_twist_in_place!(
    "2 channel 16-bit floating point in place color twist.",
    color_twist_f16_c2_in_place,
    f16,
    C2,
    nppiColorTwist32f_16f_C2IR_Ctx
);

impl_color_twist!(
    "3 channel 16-bit floating point color twist.",
    color_twist_f16_c3,
    f16,
    C3,
    nppiColorTwist32f_16f_C3R_Ctx
);

impl_color_twist_in_place!(
    "3 channel 16-bit floating point in place color twist.",
    color_twist_f16_c3_in_place,
    f16,
    C3,
    nppiColorTwist32f_16f_C3IR_Ctx
);

impl_color_twist!(
    "4 channel 16-bit floating point color twist, with alpha copy.",
    color_twist_f16_c4,
    f16,
    C4,
    nppiColorTwist32f_16f_C4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 16-bit floating point in place color twist, not affecting Alpha.",
    color_twist_f16_c4_in_place,
    f16,
    C4,
    nppiColorTwist32f_16f_C4IR_Ctx
);

impl_color_twist_with_constants!(
    "4 channel 16-bit floating point color twist with 4x4 matrix and constant vector addition.",
    color_twist_f16_c4_with_constants,
    f16,
    nppiColorTwist32fC_16f_C4R_Ctx
);

impl_color_twist_with_constants_in_place!(
    "4 channel 16-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.",
    color_twist_f16_c4_with_constants_in_place,
    f16,
    nppiColorTwist32fC_16f_C4IR_Ctx
);

impl_color_twist!(
    "1 channel 8-bit unsigned color twist.",
    color_twist_u8_c1,
    u8,
    C1,
    nppiColorTwist32f_8u_C1R_Ctx
);

impl_color_twist_in_place!(
    "1 channel 8-bit unsigned in place color twist.",
    color_twist_u8_c1_in_place,
    u8,
    C1,
    nppiColorTwist32f_8u_C1IR_Ctx
);

impl_color_twist!(
    "2 channel 8-bit unsigned color twist.",
    color_twist_u8_c2,
    u8,
    C2,
    nppiColorTwist32f_8u_C2R_Ctx
);

impl_color_twist_in_place!(
    "2 channel 8-bit unsigned in place color twist.",
    color_twist_u8_c2_in_place,
    u8,
    C2,
    nppiColorTwist32f_8u_C2IR_Ctx
);

impl_color_twist!(
    "3 channel 8-bit unsigned color twist.",
    color_twist_u8_c3,
    u8,
    C3,
    nppiColorTwist32f_8u_C3R_Ctx
);

impl_color_twist_in_place!(
    "3 channel 8-bit unsigned in place color twist.",
    color_twist_u8_c3_in_place,
    u8,
    C3,
    nppiColorTwist32f_8u_C3IR_Ctx
);

impl_color_twist!(
    "4 channel 8-bit unsigned color twist, with alpha copy.",
    color_twist_u8_c4,
    u8,
    C4,
    nppiColorTwist32f_8u_C4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 8-bit unsigned in place color twist, not affecting Alpha.",
    color_twist_u8_c4_in_place,
    u8,
    C4,
    nppiColorTwist32f_8u_C4IR_Ctx
);

impl_color_twist!(
    "4 channel 8-bit unsigned color twist, not affecting Alpha.",
    color_twist_u8_ac4,
    u8,
    AC4,
    nppiColorTwist32f_8u_AC4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 8-bit unsigned in place color twist, not affecting Alpha.",
    color_twist_u8_ac4_in_place,
    u8,
    AC4,
    nppiColorTwist32f_8u_AC4IR_Ctx
);

impl_color_twist_with_constants!(
    "4 channel 8-bit unsigned color twist with 4x4 matrix and constant vector addition.",
    color_twist_u8_c4_with_constants,
    u8,
    nppiColorTwist32fC_8u_C4R_Ctx
);

impl_color_twist_with_constants_in_place!(
    "4 channel 8-bit unsigned in place color twist with 4x4 matrix and an additional constant vector addition.",
    color_twist_u8_c4_with_constants_in_place,
    u8,
    nppiColorTwist32fC_8u_C4IR_Ctx
);

impl_color_twist_planar!(
    "3 channel 8-bit unsigned planar color twist.",
    color_twist_u8_p3,
    u8,
    3,
    nppiColorTwist32f_8u_P3R_Ctx
);

impl_color_twist_planar_in_place!(
    "3 channel 8-bit unsigned planar in place color twist.",
    color_twist_u8_p3_in_place,
    u8,
    3,
    nppiColorTwist32f_8u_IP3R_Ctx
);

impl_color_twist!(
    "1 channel 16-bit unsigned color twist.",
    color_twist_u16_c1,
    u16,
    C1,
    nppiColorTwist32f_16u_C1R_Ctx
);

impl_color_twist_in_place!(
    "1 channel 16-bit unsigned in place color twist.",
    color_twist_u16_c1_in_place,
    u16,
    C1,
    nppiColorTwist32f_16u_C1IR_Ctx
);

impl_color_twist!(
    "2 channel 16-bit unsigned color twist.",
    color_twist_u16_c2,
    u16,
    C2,
    nppiColorTwist32f_16u_C2R_Ctx
);

impl_color_twist_in_place!(
    "2 channel 16-bit unsigned in place color twist.",
    color_twist_u16_c2_in_place,
    u16,
    C2,
    nppiColorTwist32f_16u_C2IR_Ctx
);

impl_color_twist!(
    "3 channel 16-bit unsigned color twist.",
    color_twist_u16_c3,
    u16,
    C3,
    nppiColorTwist32f_16u_C3R_Ctx
);

impl_color_twist_in_place!(
    "3 channel 16-bit unsigned in place color twist.",
    color_twist_u16_c3_in_place,
    u16,
    C3,
    nppiColorTwist32f_16u_C3IR_Ctx
);

impl_color_twist!(
    "4 channel 16-bit unsigned color twist, not affecting Alpha.",
    color_twist_u16_ac4,
    u16,
    AC4,
    nppiColorTwist32f_16u_AC4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 16-bit unsigned in place color twist, not affecting Alpha.",
    color_twist_u16_ac4_in_place,
    u16,
    AC4,
    nppiColorTwist32f_16u_AC4IR_Ctx
);

impl_color_twist_planar!(
    "3 channel 16-bit unsigned planar color twist.",
    color_twist_u16_p3,
    u16,
    3,
    nppiColorTwist32f_16u_P3R_Ctx
);

impl_color_twist_planar_in_place!(
    "3 channel 16-bit unsigned planar in place color twist.",
    color_twist_u16_p3_in_place,
    u16,
    3,
    nppiColorTwist32f_16u_IP3R_Ctx
);

impl_color_twist!(
    "1 channel 8-bit signed color twist.",
    color_twist_i8_c1,
    i8,
    C1,
    nppiColorTwist32f_8s_C1R_Ctx
);

impl_color_twist_in_place!(
    "1 channel 8-bit signed in place color twist.",
    color_twist_i8_c1_in_place,
    i8,
    C1,
    nppiColorTwist32f_8s_C1IR_Ctx
);

impl_color_twist!(
    "2 channel 8-bit signed color twist.",
    color_twist_i8_c2,
    i8,
    C2,
    nppiColorTwist32f_8s_C2R_Ctx
);

impl_color_twist_in_place!(
    "2 channel 8-bit signed in place color twist.",
    color_twist_i8_c2_in_place,
    i8,
    C2,
    nppiColorTwist32f_8s_C2IR_Ctx
);

impl_color_twist!(
    "3 channel 8-bit signed color twist.",
    color_twist_i8_c3,
    i8,
    C3,
    nppiColorTwist32f_8s_C3R_Ctx
);

impl_color_twist_in_place!(
    "3 channel 8-bit signed in place color twist.",
    color_twist_i8_c3_in_place,
    i8,
    C3,
    nppiColorTwist32f_8s_C3IR_Ctx
);

impl_color_twist!(
    "4 channel 8-bit signed color twist, with alpha copy.",
    color_twist_i8_c4,
    i8,
    C4,
    nppiColorTwist32f_8s_C4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 8-bit signed in place color twist, not affecting Alpha.",
    color_twist_i8_c4_in_place,
    i8,
    C4,
    nppiColorTwist32f_8s_C4IR_Ctx
);

impl_color_twist!(
    "4 channel 8-bit signed color twist, not affecting Alpha.",
    color_twist_i8_ac4,
    i8,
    AC4,
    nppiColorTwist32f_8s_AC4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 8-bit signed in place color twist, not affecting Alpha.",
    color_twist_i8_ac4_in_place,
    i8,
    AC4,
    nppiColorTwist32f_8s_AC4IR_Ctx
);

impl_color_twist_planar!(
    "3 channel 8-bit signed planar color twist.",
    color_twist_i8_p3,
    i8,
    3,
    nppiColorTwist32f_8s_P3R_Ctx
);

impl_color_twist_planar_in_place!(
    "3 channel 8-bit signed planar in place color twist.",
    color_twist_i8_p3_in_place,
    i8,
    3,
    nppiColorTwist32f_8s_IP3R_Ctx
);

impl_color_twist!(
    "1 channel 16-bit signed color twist.",
    color_twist_i16_c1,
    i16,
    C1,
    nppiColorTwist32f_16s_C1R_Ctx
);

impl_color_twist_in_place!(
    "1 channel 16-bit signed in place color twist.",
    color_twist_i16_c1_in_place,
    i16,
    C1,
    nppiColorTwist32f_16s_C1IR_Ctx
);

impl_color_twist!(
    "2 channel 16-bit signed color twist.",
    color_twist_i16_c2,
    i16,
    C2,
    nppiColorTwist32f_16s_C2R_Ctx
);

impl_color_twist_in_place!(
    "2 channel 16-bit signed in place color twist.",
    color_twist_i16_c2_in_place,
    i16,
    C2,
    nppiColorTwist32f_16s_C2IR_Ctx
);

impl_color_twist!(
    "3 channel 16-bit signed color twist.",
    color_twist_i16_c3,
    i16,
    C3,
    nppiColorTwist32f_16s_C3R_Ctx
);

impl_color_twist_in_place!(
    "3 channel 16-bit signed in place color twist.",
    color_twist_i16_c3_in_place,
    i16,
    C3,
    nppiColorTwist32f_16s_C3IR_Ctx
);

impl_color_twist!(
    "4 channel 16-bit signed color twist, not affecting Alpha.",
    color_twist_i16_ac4,
    i16,
    AC4,
    nppiColorTwist32f_16s_AC4R_Ctx
);

impl_color_twist_in_place!(
    "4 channel 16-bit signed in place color twist, not affecting Alpha.",
    color_twist_i16_ac4_in_place,
    i16,
    AC4,
    nppiColorTwist32f_16s_AC4IR_Ctx
);

impl_color_twist_planar!(
    "3 channel 16-bit signed planar color twist.",
    color_twist_i16_p3,
    i16,
    3,
    nppiColorTwist32f_16s_P3R_Ctx
);

impl_color_twist_planar_in_place!(
    "3 channel 16-bit signed planar in place color twist.",
    color_twist_i16_p3_in_place,
    i16,
    3,
    nppiColorTwist32f_16s_IP3R_Ctx
);

impl_generic_color_twist!(
    "Applies the color transformation defined by `twist`.",
ColorTwistC1, color_twist, color_twist_c1, C1, [
    f32 => color_twist_f32_c1,
    f16 => color_twist_f16_c1,
    u8 => color_twist_u8_c1,
    u16 => color_twist_u16_c1,
    i8 => color_twist_i8_c1,
    i16 => color_twist_i16_c1,
]
);

impl_generic_color_twist_in_place!(
    "Applies the color transformation defined by `twist`. Updates the image in place.",
ColorTwistC1InPlace,
    color_twist_in_place,
    color_twist_c1_in_place,
    C1,
    [
        f32 => color_twist_f32_c1_in_place,
        f16 => color_twist_f16_c1_in_place,
        u8 => color_twist_u8_c1_in_place,
        u16 => color_twist_u16_c1_in_place,
        i8 => color_twist_i8_c1_in_place,
        i16 => color_twist_i16_c1_in_place,
    ]
);

impl_generic_color_twist!(
    "Applies the color transformation defined by `twist`.",
ColorTwistC2, color_twist, color_twist_c2, C2, [
    f32 => color_twist_f32_c2,
    f16 => color_twist_f16_c2,
    u8 => color_twist_u8_c2,
    u16 => color_twist_u16_c2,
    i8 => color_twist_i8_c2,
    i16 => color_twist_i16_c2,
]
);

impl_generic_color_twist_in_place!(
    "Applies the color transformation defined by `twist`. Updates the image in place.",
ColorTwistC2InPlace,
    color_twist_in_place,
    color_twist_c2_in_place,
    C2,
    [
        f32 => color_twist_f32_c2_in_place,
        f16 => color_twist_f16_c2_in_place,
        u8 => color_twist_u8_c2_in_place,
        u16 => color_twist_u16_c2_in_place,
        i8 => color_twist_i8_c2_in_place,
        i16 => color_twist_i16_c2_in_place,
    ]
);

impl_generic_color_twist!(
    "Applies the color transformation defined by `twist`.",
ColorTwistC3, color_twist, color_twist_c3, C3, [
    f32 => color_twist_f32_c3,
    f16 => color_twist_f16_c3,
    u8 => color_twist_u8_c3,
    u16 => color_twist_u16_c3,
    i8 => color_twist_i8_c3,
    i16 => color_twist_i16_c3,
]
);

impl_generic_color_twist_in_place!(
    "Applies the color transformation defined by `twist`. Updates the image in place.",
ColorTwistC3InPlace,
    color_twist_in_place,
    color_twist_c3_in_place,
    C3,
    [
        f32 => color_twist_f32_c3_in_place,
        f16 => color_twist_f16_c3_in_place,
        u8 => color_twist_u8_c3_in_place,
        u16 => color_twist_u16_c3_in_place,
        i8 => color_twist_i8_c3_in_place,
        i16 => color_twist_i16_c3_in_place,
    ]
);

impl_generic_color_twist!(
    "Applies the color transformation defined by `twist`.",
ColorTwistC4, color_twist, color_twist_c4, C4, [
    f32 => color_twist_f32_c4,
    f16 => color_twist_f16_c4,
    u8 => color_twist_u8_c4,
    i8 => color_twist_i8_c4,
]
);

impl_generic_color_twist_in_place!(
    "Applies the color transformation defined by `twist`. Updates the image in place.",
ColorTwistC4InPlace,
    color_twist_in_place,
    color_twist_c4_in_place,
    C4,
    [
        f32 => color_twist_f32_c4_in_place,
        f16 => color_twist_f16_c4_in_place,
        u8 => color_twist_u8_c4_in_place,
        i8 => color_twist_i8_c4_in_place,
    ]
);

impl_generic_color_twist!(
    "Applies the color transformation defined by `twist`. The alpha channel is not processed.",
ColorTwistAc4, color_twist, color_twist_ac4, AC4, [
    f32 => color_twist_f32_ac4,
    u8 => color_twist_u8_ac4,
    u16 => color_twist_u16_ac4,
    i8 => color_twist_i8_ac4,
    i16 => color_twist_i16_ac4,
]
);

impl_generic_color_twist_in_place!(
    "Applies the color transformation defined by `twist`. Updates the image in place. The alpha channel is not processed.",
ColorTwistAc4InPlace,
    color_twist_in_place,
    color_twist_ac4_in_place,
    AC4,
    [
        f32 => color_twist_f32_ac4_in_place,
        u8 => color_twist_u8_ac4_in_place,
        u16 => color_twist_u16_ac4_in_place,
        i8 => color_twist_i8_ac4_in_place,
        i16 => color_twist_i16_ac4_in_place,
    ]
);

impl_generic_color_twist_with_constants!(
    "Applies the color transformation defined by `twist` and `constants`.",
ColorTwistWithConstantsC4,
    color_twist_with_constants,
    color_twist_c4_with_constants,
    [
        f32 => color_twist_f32_c4_with_constants,
        f16 => color_twist_f16_c4_with_constants,
        u8 => color_twist_u8_c4_with_constants,
    ]
);

impl_generic_color_twist_with_constants_in_place!(
    "Applies the color transformation defined by `twist` and `constants`. Updates the image in place.",
ColorTwistWithConstantsC4InPlace,
    color_twist_with_constants_in_place,
    color_twist_c4_with_constants_in_place,
    [
        f32 => color_twist_f32_c4_with_constants_in_place,
        f16 => color_twist_f16_c4_with_constants_in_place,
        u8 => color_twist_u8_c4_with_constants_in_place,
    ]
);

/// Dispatches planar color twists for a native element type and plane count.
///
/// # Safety
///
/// Implementations must preserve the native element ABI, plane count, and matrix semantics.
/// Uphold the method's shared-geometry validation, memory-access rules, and queued lifetimes.
pub unsafe trait ColorTwistPlanar<const PLANES: usize>: DataTypeLike {
    /// Applies a color-twist matrix to the planar image.
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
    unsafe fn color_twist_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageOutputView<'_, Self, PLANES>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl ColorTwistPlanar<$planes> for $ty {
            unsafe fn color_twist_planar(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
                twist: ColorTwistMatrix,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, source, destination, twist) }
            }
        }
    };
}

impl_color_twist_planar_dispatch!(f32, 3, color_twist_f32_p3);

impl_color_twist_planar_dispatch!(u8, 3, color_twist_u8_p3);

impl_color_twist_planar_dispatch!(u16, 3, color_twist_u16_p3);

impl_color_twist_planar_dispatch!(i8, 3, color_twist_i8_p3);

impl_color_twist_planar_dispatch!(i16, 3, color_twist_i16_p3);

/// Applies a color-twist matrix to the planar image.
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
pub unsafe fn color_twist_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageOutputView<'_, T, PLANES>,
    twist: ColorTwistMatrix,
) -> Result<()>
where
    T: ColorTwistPlanar<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::color_twist_planar(stream_context, source, destination, twist) }
}

/// Dispatches in-place planar color twists for a native element type and plane count.
///
/// # Safety
///
/// Implementations must preserve the native element ABI, plane count, and matrix semantics.
/// Uphold the method's shared-geometry validation, memory-access rules, and queued lifetimes.
pub unsafe trait ColorTwistPlanarInPlace<const PLANES: usize>: DataTypeLike {
    /// Applies a color-twist matrix to the planar image in place.
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
    unsafe fn color_twist_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, PLANES>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_planar_in_place_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        unsafe impl ColorTwistPlanarInPlace<$planes> for $ty {
            unsafe fn color_twist_planar_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, Self, $planes>,
                twist: ColorTwistMatrix,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, image, twist) }
            }
        }
    };
}

impl_color_twist_planar_in_place_dispatch!(f32, 3, color_twist_f32_p3_in_place);

impl_color_twist_planar_in_place_dispatch!(u8, 3, color_twist_u8_p3_in_place);

impl_color_twist_planar_in_place_dispatch!(u16, 3, color_twist_u16_p3_in_place);

impl_color_twist_planar_in_place_dispatch!(i8, 3, color_twist_i8_p3_in_place);

impl_color_twist_planar_in_place_dispatch!(i16, 3, color_twist_i16_p3_in_place);

/// Applies a color-twist matrix to the planar image in place.
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
pub unsafe fn color_twist_planar_in_place<T, const PLANES: usize>(
    stream_context: &StreamContext,
    image: &mut PlanarImageViewMut<'_, T, PLANES>,
    twist: ColorTwistMatrix,
) -> Result<()>
where
    T: ColorTwistPlanarInPlace<PLANES>,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::color_twist_planar_in_place(stream_context, image, twist) }
}
