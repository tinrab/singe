use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color::{ColorTwistBatchConstantsMatrix4, ColorTwistMatrix},
        color_descriptors::{
            color_twist_batch_descriptors, color_twist_batch_in_place_descriptors,
            color_twist_batch_with_constants_descriptors,
            color_twist_batch_with_constants_in_place_descriptors,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_twist_batch!(
    "1 channel 8-bit unsigned integer color twist batch.",
    color_twist_batch_u8_c1,
    C1,
    nppiColorTwistBatch32f_8u_C1R_Ctx
);

impl_color_twist_batch!(
    "3 channel 8-bit unsigned integer color twist batch.",
    color_twist_batch_u8_c3,
    C3,
    nppiColorTwistBatch32f_8u_C3R_Ctx
);

impl_color_twist_batch!(
    "4 channel 8-bit unsigned integer color twist batch.",
    color_twist_batch_u8_c4,
    C4,
    nppiColorTwistBatch32f_8u_C4R_Ctx
);

impl_color_twist_batch!(
    "4 channel 8-bit unsigned integer color twist batch, not affecting Alpha.",
    color_twist_batch_u8_ac4,
    AC4,
    nppiColorTwistBatch32f_8u_AC4R_Ctx
);

impl_color_twist_batch_in_place!(
    "1 channel 8-bit unsigned integer in place color twist batch.",
    color_twist_batch_u8_c1_in_place,
    C1,
    nppiColorTwistBatch32f_8u_C1IR_Ctx
);

impl_color_twist_batch_in_place!(
    "3 channel 8-bit unsigned integer in place color twist batch.",
    color_twist_batch_u8_c3_in_place,
    C3,
    nppiColorTwistBatch32f_8u_C3IR_Ctx
);

impl_color_twist_batch_in_place!(
    "4 channel 8-bit unsigned integer in place color twist batch.",
    color_twist_batch_u8_c4_in_place,
    C4,
    nppiColorTwistBatch32f_8u_C4IR_Ctx
);

impl_color_twist_batch_in_place!(
    "4 channel 8-bit unsigned integer in place color twist batch, not affecting Alpha.",
    color_twist_batch_u8_ac4_in_place,
    AC4,
    nppiColorTwistBatch32f_8u_AC4IR_Ctx
);

impl_color_twist_batch_typed!(
    "1 channel 32-bit floating point color twist batch.",
    color_twist_batch_f32_c1,
    f32,
    C1,
    nppiColorTwistBatch_32f_C1R_Ctx
);

impl_color_twist_batch_typed!(
    "3 channel 32-bit floating point color twist batch.",
    color_twist_batch_f32_c3,
    f32,
    C3,
    nppiColorTwistBatch_32f_C3R_Ctx
);

impl_color_twist_batch_typed!(
    "4 channel 32-bit floating point color twist batch.",
    color_twist_batch_f32_c4,
    f32,
    C4,
    nppiColorTwistBatch_32f_C4R_Ctx
);

impl_color_twist_batch_typed!(
    "4 channel 32-bit floating point color twist batch, not affecting Alpha.",
    color_twist_batch_f32_ac4,
    f32,
    AC4,
    nppiColorTwistBatch_32f_AC4R_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "1 channel 32-bit floating point in place color twist batch.",
    color_twist_batch_f32_c1_in_place,
    f32,
    C1,
    nppiColorTwistBatch_32f_C1IR_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "3 channel 32-bit floating point in place color twist batch.",
    color_twist_batch_f32_c3_in_place,
    f32,
    C3,
    nppiColorTwistBatch_32f_C3IR_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "4 channel 32-bit floating point in place color twist batch.",
    color_twist_batch_f32_c4_in_place,
    f32,
    C4,
    nppiColorTwistBatch_32f_C4IR_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "4 channel 32-bit floating point in place color twist batch, not affecting Alpha.",
    color_twist_batch_f32_ac4_in_place,
    f32,
    AC4,
    nppiColorTwistBatch_32f_AC4IR_Ctx
);

impl_color_twist_batch_typed!(
    "1 channel 16-bit floating point color twist batch.",
    color_twist_batch_f16_c1,
    f16,
    C1,
    nppiColorTwistBatch32f_16f_C1R_Ctx
);

impl_color_twist_batch_typed!(
    "3 channel 16-bit floating point color twist batch.",
    color_twist_batch_f16_c3,
    f16,
    C3,
    nppiColorTwistBatch32f_16f_C3R_Ctx
);

impl_color_twist_batch_typed!(
    "4 channel 16-bit floating point color twist batch.",
    color_twist_batch_f16_c4,
    f16,
    C4,
    nppiColorTwistBatch32f_16f_C4R_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "1 channel 16-bit floating point in place color twist batch.",
    color_twist_batch_f16_c1_in_place,
    f16,
    C1,
    nppiColorTwistBatch32f_16f_C1IR_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "3 channel 16-bit floating point in place color twist batch.",
    color_twist_batch_f16_c3_in_place,
    f16,
    C3,
    nppiColorTwistBatch32f_16f_C3IR_Ctx
);

impl_color_twist_batch_in_place_typed!(
    "4 channel 16-bit floating point in place color twist batch.",
    color_twist_batch_f16_c4_in_place,
    f16,
    C4,
    nppiColorTwistBatch32f_16f_C4IR_Ctx
);

impl_color_twist_batch_with_constants_typed!(
    "4 channel 8-bit unsigned integer color twist with 4x5 matrix including a constant vector (20 coefficients total).",
    color_twist_batch_u8_c4_with_constants,
    u8,
    nppiColorTwistBatch32fC_8u_C4R_Ctx
);

impl_color_twist_batch_with_constants_in_place_typed!(
    "4 channel 8-bit unsigned integer in place color twist with 4x5 matrix including a constant vector (20 coefficients total).",
    color_twist_batch_u8_c4_with_constants_in_place,
    u8,
    nppiColorTwistBatch32fC_8u_C4IR_Ctx
);

impl_color_twist_batch_with_constants_typed!(
    "4 channel 16-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).",
    color_twist_batch_f16_c4_with_constants,
    f16,
    nppiColorTwistBatch32fC_16f_C4R_Ctx
);

impl_color_twist_batch_with_constants_in_place_typed!(
    "4 channel in place 16-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).",
    color_twist_batch_f16_c4_with_constants_in_place,
    f16,
    nppiColorTwistBatch32fC_16f_C4IR_Ctx
);

impl_color_twist_batch_with_constants_typed!(
    "4 channel 32-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).",
    color_twist_batch_f32_c4_with_constants,
    f32,
    nppiColorTwistBatch_32fC_C4R_Ctx
);

impl_color_twist_batch_with_constants_in_place_typed!(
    "4 channel in place 32-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).",
    color_twist_batch_f32_c4_with_constants_in_place,
    f32,
    nppiColorTwistBatch_32fC_C4IR_Ctx
);

/// Dispatches batched color twists for a native element type and channel layout.
///
/// # Safety
///
/// Implementations must preserve the native element and layout ABI, matrix interpretation,
/// and the method's batch validation, memory-access, and completion contracts.
pub unsafe trait ColorTwistBatch<L: ChannelLayout>: DataTypeLike {
    /// Applies one color-twist matrix per image.
    ///
    /// Each matrix has three rows of four coefficients, including additive terms.
    /// For `AC4`, alpha is not processed and retains its previous initialization state.
    /// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
    /// These values and matrix coefficients are passed to NPP without numerical validation.
    /// Pixels outside the common ROI remain unwritten.
    ///
    /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
    /// The processed ROI uses the smallest image width and smallest image height in the batch.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Batches must contain at least two images and have a count representable as `i32`.
    /// `twists` must contain one coefficient matrix per image.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    /// Source and destination batch lengths must match.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn color_twist_batch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, L>],
        twists: &[ColorTwistMatrix],
        destinations: &mut [ImageOutputView<'_, Self, L>],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl ColorTwistBatch<$layout> for $ty {
            unsafe fn color_twist_batch(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, Self, $layout>],
                twists: &[ColorTwistMatrix],
                destinations: &mut [ImageOutputView<'_, Self, $layout>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, sources, twists, destinations, min, max) }
            }
        }
    };
}

impl_color_twist_batch_dispatch!(u8, C1, color_twist_batch_u8_c1);

impl_color_twist_batch_dispatch!(u8, C3, color_twist_batch_u8_c3);

impl_color_twist_batch_dispatch!(u8, C4, color_twist_batch_u8_c4);

impl_color_twist_batch_dispatch!(u8, AC4, color_twist_batch_u8_ac4);

impl_color_twist_batch_dispatch!(f16, C1, color_twist_batch_f16_c1);

impl_color_twist_batch_dispatch!(f16, C3, color_twist_batch_f16_c3);

impl_color_twist_batch_dispatch!(f16, C4, color_twist_batch_f16_c4);

impl_color_twist_batch_dispatch!(f32, C1, color_twist_batch_f32_c1);

impl_color_twist_batch_dispatch!(f32, C3, color_twist_batch_f32_c3);

impl_color_twist_batch_dispatch!(f32, C4, color_twist_batch_f32_c4);

impl_color_twist_batch_dispatch!(f32, AC4, color_twist_batch_f32_ac4);

/// Applies one color-twist matrix per image.
///
/// Each matrix has three rows of four coefficients, including additive terms.
/// For `AC4`, alpha is not processed and retains its previous initialization state.
/// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
/// These values and matrix coefficients are passed to NPP without numerical validation.
/// Pixels outside the common ROI remain unwritten.
///
/// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
/// The processed ROI uses the smallest image width and smallest image height in the batch.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Each source and destination image pair must have matching sizes.
/// Batches must contain at least two images and have a count representable as `i32`.
/// `twists` must contain one coefficient matrix per image.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
/// Source and destination batch lengths must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn color_twist_batch<T, L>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, L>],
    twists: &[ColorTwistMatrix],
    destinations: &mut [ImageOutputView<'_, T, L>],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatch<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::color_twist_batch(stream_context, sources, twists, destinations, min, max) }
}

/// Dispatches batched color twists for a native element type and channel layout.
///
/// # Safety
///
/// Implementations must preserve the native element and layout ABI, matrix interpretation,
/// and the method's batch validation, memory-access, and completion contracts.
pub unsafe trait ColorTwistBatchInPlace<L: ChannelLayout>: DataTypeLike {
    /// Applies one color-twist matrix per image in place.
    ///
    /// Each matrix has three rows of four coefficients, including additive terms.
    /// For `AC4`, alpha is not processed and retains its previous initialization state.
    /// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
    /// These values and matrix coefficients are passed to NPP without numerical validation.
    /// Pixels outside the common ROI remain unwritten.
    ///
    /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
    /// The processed ROI uses the smallest image width and smallest image height in the batch.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Batches must contain at least two images and have a count representable as `i32`.
    /// `twists` must contain one coefficient matrix per image.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn color_twist_batch_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, Self, L>],
        twists: &[ColorTwistMatrix],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_in_place_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl ColorTwistBatchInPlace<$layout> for $ty {
            unsafe fn color_twist_batch_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, Self, $layout>],
                twists: &[ColorTwistMatrix],
                min: f32,
                max: f32,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, images, twists, min, max) }
            }
        }
    };
}

impl_color_twist_batch_in_place_dispatch!(u8, C1, color_twist_batch_u8_c1_in_place);

impl_color_twist_batch_in_place_dispatch!(u8, C3, color_twist_batch_u8_c3_in_place);

impl_color_twist_batch_in_place_dispatch!(u8, C4, color_twist_batch_u8_c4_in_place);

impl_color_twist_batch_in_place_dispatch!(u8, AC4, color_twist_batch_u8_ac4_in_place);

impl_color_twist_batch_in_place_dispatch!(f16, C1, color_twist_batch_f16_c1_in_place);

impl_color_twist_batch_in_place_dispatch!(f16, C3, color_twist_batch_f16_c3_in_place);

impl_color_twist_batch_in_place_dispatch!(f16, C4, color_twist_batch_f16_c4_in_place);

impl_color_twist_batch_in_place_dispatch!(f32, C1, color_twist_batch_f32_c1_in_place);

impl_color_twist_batch_in_place_dispatch!(f32, C3, color_twist_batch_f32_c3_in_place);

impl_color_twist_batch_in_place_dispatch!(f32, C4, color_twist_batch_f32_c4_in_place);

impl_color_twist_batch_in_place_dispatch!(f32, AC4, color_twist_batch_f32_ac4_in_place);

/// Applies one color-twist matrix per image in place.
///
/// Each matrix has three rows of four coefficients, including additive terms.
/// For `AC4`, alpha is not processed and retains its previous initialization state.
/// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
/// These values and matrix coefficients are passed to NPP without numerical validation.
/// Pixels outside the common ROI remain unwritten.
///
/// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
/// The processed ROI uses the smallest image width and smallest image height in the batch.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Batches must contain at least two images and have a count representable as `i32`.
/// `twists` must contain one coefficient matrix per image.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
/// Keep image allocations valid until this call completes its native work.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn color_twist_batch_in_place<T, L>(
    stream_context: &StreamContext,
    images: &mut [ImageViewMut<'_, T, L>],
    twists: &[ColorTwistMatrix],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatchInPlace<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::color_twist_batch_in_place(stream_context, images, twists, min, max) }
}

/// Dispatches batched color twists for a native element type with four-channel constant terms.
///
/// # Safety
///
/// Implementations must preserve the native element and layout ABI, matrix interpretation,
/// and the method's batch validation, memory-access, and completion contracts.
pub unsafe trait ColorTwistBatchWithConstantsC4: DataTypeLike {
    /// Applies one color-twist matrix per image.
    ///
    /// Each matrix has 20 coefficients: a 4-by-4 transform and four additive constants.
    /// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
    /// These values and matrix coefficients are passed to NPP without numerical validation.
    /// Pixels outside the common ROI remain unwritten.
    ///
    /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
    /// The processed ROI uses the smallest image width and smallest image height in the batch.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Batches must contain at least two images and have a count representable as `i32`.
    /// `twists` must contain one coefficient matrix per image.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    /// Source and destination batch lengths must match.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn color_twist_batch_with_constants_c4(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        destinations: &mut [ImageOutputView<'_, Self, C4>],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_with_constants_c4_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl ColorTwistBatchWithConstantsC4 for $ty {
            unsafe fn color_twist_batch_with_constants_c4(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, Self, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                destinations: &mut [ImageOutputView<'_, Self, C4>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, sources, twists, destinations, min, max) }
            }
        }
    };
}

impl_color_twist_batch_with_constants_c4_dispatch!(u8, color_twist_batch_u8_c4_with_constants);

impl_color_twist_batch_with_constants_c4_dispatch!(f16, color_twist_batch_f16_c4_with_constants);

impl_color_twist_batch_with_constants_c4_dispatch!(f32, color_twist_batch_f32_c4_with_constants);

/// Applies one color-twist matrix per image.
///
/// Each matrix has 20 coefficients: a 4-by-4 transform and four additive constants.
/// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
/// These values and matrix coefficients are passed to NPP without numerical validation.
/// Pixels outside the common ROI remain unwritten.
///
/// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
/// The processed ROI uses the smallest image width and smallest image height in the batch.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Each source and destination image pair must have matching sizes.
/// Batches must contain at least two images and have a count representable as `i32`.
/// `twists` must contain one coefficient matrix per image.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
/// Source and destination batch lengths must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn color_twist_batch_c4_with_constants<T>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
    destinations: &mut [ImageOutputView<'_, T, C4>],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatchWithConstantsC4,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::color_twist_batch_with_constants_c4(
            stream_context,
            sources,
            twists,
            destinations,
            min,
            max,
        )
    }
}

/// Dispatches batched color twists for a native element type with four-channel constant terms.
///
/// # Safety
///
/// Implementations must preserve the native element and layout ABI, matrix interpretation,
/// and the method's batch validation, memory-access, and completion contracts.
pub unsafe trait ColorTwistBatchWithConstantsC4InPlace: DataTypeLike {
    /// Applies one color-twist matrix per image in place.
    ///
    /// Each matrix has 20 coefficients: a 4-by-4 transform and four additive constants.
    /// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
    /// These values and matrix coefficients are passed to NPP without numerical validation.
    /// Pixels outside the common ROI remain unwritten.
    ///
    /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
    /// The processed ROI uses the smallest image width and smallest image height in the batch.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Batches must contain at least two images and have a count representable as `i32`.
    /// `twists` must contain one coefficient matrix per image.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn color_twist_batch_with_constants_c4_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, Self, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_with_constants_c4_in_place_dispatch {
    ($ty:ty, $function:ident) => {
        unsafe impl ColorTwistBatchWithConstantsC4InPlace for $ty {
            unsafe fn color_twist_batch_with_constants_c4_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, Self, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                min: f32,
                max: f32,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, images, twists, min, max) }
            }
        }
    };
}

impl_color_twist_batch_with_constants_c4_in_place_dispatch!(
    u8,
    color_twist_batch_u8_c4_with_constants_in_place
);

impl_color_twist_batch_with_constants_c4_in_place_dispatch!(
    f16,
    color_twist_batch_f16_c4_with_constants_in_place
);

impl_color_twist_batch_with_constants_c4_in_place_dispatch!(
    f32,
    color_twist_batch_f32_c4_with_constants_in_place
);

/// Applies one color-twist matrix per image in place.
///
/// Each matrix has 20 coefficients: a 4-by-4 transform and four additive constants.
/// `min` supplies the minimum clamp value; `max` supplies the maximum saturation and clamp value.
/// These values and matrix coefficients are passed to NPP without numerical validation.
/// Pixels outside the common ROI remain unwritten.
///
/// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
/// The processed ROI uses the smallest image width and smallest image height in the batch.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Batches must contain at least two images and have a count representable as `i32`.
/// `twists` must contain one coefficient matrix per image.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
/// Keep image allocations valid until this call completes its native work.
/// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn color_twist_batch_c4_with_constants_in_place<T>(
    stream_context: &StreamContext,
    images: &mut [ImageViewMut<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatchWithConstantsC4InPlace,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::color_twist_batch_with_constants_c4_in_place(stream_context, images, twists, min, max)
    }
}
