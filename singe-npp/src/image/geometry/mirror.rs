use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        geometry_helpers::{
            mirror_batch_descriptors, mirror_batch_in_place_descriptors, validate_same_size,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{Axis, DataTypeLike},
};

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_u8_c1,
    u8,
    C1,
    nppiMirror_8u_C1R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_u8_c3,
    u8,
    C3,
    nppiMirror_8u_C3R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_u8_c4,
    u8,
    C4,
    nppiMirror_8u_C4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination. The destination alpha channel is not processed.",
    mirror_u8_ac4,
    u8,
    AC4,
    nppiMirror_8u_AC4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_u16_c1,
    u16,
    C1,
    nppiMirror_16u_C1R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_u16_c3,
    u16,
    C3,
    nppiMirror_16u_C3R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_u16_c4,
    u16,
    C4,
    nppiMirror_16u_C4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination. The destination alpha channel is not processed.",
    mirror_u16_ac4,
    u16,
    AC4,
    nppiMirror_16u_AC4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_i16_c1,
    i16,
    C1,
    nppiMirror_16s_C1R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_i16_c3,
    i16,
    C3,
    nppiMirror_16s_C3R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_i16_c4,
    i16,
    C4,
    nppiMirror_16s_C4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination. The destination alpha channel is not processed.",
    mirror_i16_ac4,
    i16,
    AC4,
    nppiMirror_16s_AC4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_i32_c1,
    i32,
    C1,
    nppiMirror_32s_C1R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_i32_c3,
    i32,
    C3,
    nppiMirror_32s_C3R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_i32_c4,
    i32,
    C4,
    nppiMirror_32s_C4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination. The destination alpha channel is not processed.",
    mirror_i32_ac4,
    i32,
    AC4,
    nppiMirror_32s_AC4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_f32_c1,
    f32,
    C1,
    nppiMirror_32f_C1R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_f32_c3,
    f32,
    C3,
    nppiMirror_32f_C3R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination.",
    mirror_f32_c4,
    f32,
    C4,
    nppiMirror_32f_C4R_Ctx
);

impl_mirror!(
    "Mirrors the source image about `axis` into the destination. The destination alpha channel is not processed.",
    mirror_f32_ac4,
    f32,
    AC4,
    nppiMirror_32f_AC4R_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_u8_c1_in_place,
    u8,
    C1,
    nppiMirror_8u_C1IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_u8_c3_in_place,
    u8,
    C3,
    nppiMirror_8u_C3IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_u8_c4_in_place,
    u8,
    C4,
    nppiMirror_8u_C4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place. The destination alpha channel is not processed.",
    mirror_u8_ac4_in_place,
    u8,
    AC4,
    nppiMirror_8u_AC4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_u16_c1_in_place,
    u16,
    C1,
    nppiMirror_16u_C1IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_u16_c3_in_place,
    u16,
    C3,
    nppiMirror_16u_C3IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_u16_c4_in_place,
    u16,
    C4,
    nppiMirror_16u_C4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place. The destination alpha channel is not processed.",
    mirror_u16_ac4_in_place,
    u16,
    AC4,
    nppiMirror_16u_AC4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_i16_c1_in_place,
    i16,
    C1,
    nppiMirror_16s_C1IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_i16_c3_in_place,
    i16,
    C3,
    nppiMirror_16s_C3IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_i16_c4_in_place,
    i16,
    C4,
    nppiMirror_16s_C4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place. The destination alpha channel is not processed.",
    mirror_i16_ac4_in_place,
    i16,
    AC4,
    nppiMirror_16s_AC4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_i32_c1_in_place,
    i32,
    C1,
    nppiMirror_32s_C1IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_i32_c3_in_place,
    i32,
    C3,
    nppiMirror_32s_C3IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_i32_c4_in_place,
    i32,
    C4,
    nppiMirror_32s_C4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place. The destination alpha channel is not processed.",
    mirror_i32_ac4_in_place,
    i32,
    AC4,
    nppiMirror_32s_AC4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_f32_c1_in_place,
    f32,
    C1,
    nppiMirror_32f_C1IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_f32_c3_in_place,
    f32,
    C3,
    nppiMirror_32f_C3IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place.",
    mirror_f32_c4_in_place,
    f32,
    C4,
    nppiMirror_32f_C4IR_Ctx
);

impl_mirror_in_place!(
    "Mirrors `image` about `axis` in place. The destination alpha channel is not processed.",
    mirror_f32_ac4_in_place,
    f32,
    AC4,
    nppiMirror_32f_AC4IR_Ctx
);

/// Element and channel-layout dispatch for [`mirror`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait Mirror<L: ChannelLayout>: DataTypeLike {
    /// Mirrors the source image about `axis` into the destination.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination sizes must match.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready when the selected stream reads them.
    /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn mirror(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageOutputView<'_, Self, L>,
        axis: Axis,
    ) -> Result<()>;
}

macro_rules! impl_mirror_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl Mirror<$layout> for $ty {
            unsafe fn mirror(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                axis: Axis,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, source, destination, axis) }
            }
        }
    };
}

impl_mirror_dispatch!(u8, C1, mirror_u8_c1);

impl_mirror_dispatch!(u8, C3, mirror_u8_c3);

impl_mirror_dispatch!(u8, C4, mirror_u8_c4);

impl_mirror_dispatch!(u8, AC4, mirror_u8_ac4);

impl_mirror_dispatch!(u16, C1, mirror_u16_c1);

impl_mirror_dispatch!(u16, C3, mirror_u16_c3);

impl_mirror_dispatch!(u16, C4, mirror_u16_c4);

impl_mirror_dispatch!(u16, AC4, mirror_u16_ac4);

impl_mirror_dispatch!(i16, C1, mirror_i16_c1);

impl_mirror_dispatch!(i16, C3, mirror_i16_c3);

impl_mirror_dispatch!(i16, C4, mirror_i16_c4);

impl_mirror_dispatch!(i16, AC4, mirror_i16_ac4);

impl_mirror_dispatch!(i32, C1, mirror_i32_c1);

impl_mirror_dispatch!(i32, C3, mirror_i32_c3);

impl_mirror_dispatch!(i32, C4, mirror_i32_c4);

impl_mirror_dispatch!(i32, AC4, mirror_i32_ac4);

impl_mirror_dispatch!(f32, C1, mirror_f32_c1);

impl_mirror_dispatch!(f32, C3, mirror_f32_c3);

impl_mirror_dispatch!(f32, C4, mirror_f32_c4);

impl_mirror_dispatch!(f32, AC4, mirror_f32_ac4);

/// Mirrors the source image about `axis` into the destination.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Source and destination sizes must match.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn mirror<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
    axis: Axis,
) -> Result<()>
where
    T: Mirror<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::mirror(stream_context, source, destination, axis) }
}

/// Element and channel-layout dispatch for [`mirror_in_place`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait MirrorInPlace<L: ChannelLayout>: DataTypeLike {
    /// Mirrors the image about `axis` in place.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels and channels not processed by the native routine retain their existing values.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Images must contain initialized values ready for the selected stream and remain exclusively writable during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn mirror_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, Self, L>,
        axis: Axis,
    ) -> Result<()>;
}

macro_rules! impl_mirror_in_place_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl MirrorInPlace<$layout> for $ty {
            unsafe fn mirror_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, $layout>,
                axis: Axis,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, image, axis) }
            }
        }
    };
}

impl_mirror_in_place_dispatch!(u8, C1, mirror_u8_c1_in_place);

impl_mirror_in_place_dispatch!(u8, C3, mirror_u8_c3_in_place);

impl_mirror_in_place_dispatch!(u8, C4, mirror_u8_c4_in_place);

impl_mirror_in_place_dispatch!(u8, AC4, mirror_u8_ac4_in_place);

impl_mirror_in_place_dispatch!(u16, C1, mirror_u16_c1_in_place);

impl_mirror_in_place_dispatch!(u16, C3, mirror_u16_c3_in_place);

impl_mirror_in_place_dispatch!(u16, C4, mirror_u16_c4_in_place);

impl_mirror_in_place_dispatch!(u16, AC4, mirror_u16_ac4_in_place);

impl_mirror_in_place_dispatch!(i16, C1, mirror_i16_c1_in_place);

impl_mirror_in_place_dispatch!(i16, C3, mirror_i16_c3_in_place);

impl_mirror_in_place_dispatch!(i16, C4, mirror_i16_c4_in_place);

impl_mirror_in_place_dispatch!(i16, AC4, mirror_i16_ac4_in_place);

impl_mirror_in_place_dispatch!(i32, C1, mirror_i32_c1_in_place);

impl_mirror_in_place_dispatch!(i32, C3, mirror_i32_c3_in_place);

impl_mirror_in_place_dispatch!(i32, C4, mirror_i32_c4_in_place);

impl_mirror_in_place_dispatch!(i32, AC4, mirror_i32_ac4_in_place);

impl_mirror_in_place_dispatch!(f32, C1, mirror_f32_c1_in_place);

impl_mirror_in_place_dispatch!(f32, C3, mirror_f32_c3_in_place);

impl_mirror_in_place_dispatch!(f32, C4, mirror_f32_c4_in_place);

impl_mirror_in_place_dispatch!(f32, AC4, mirror_f32_ac4_in_place);

/// Mirrors the image about `axis` in place.
///
/// Enqueues work on the selected stream.
/// Pixels and channels not processed by the native routine retain their existing values.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Images must contain initialized values ready for the selected stream and remain exclusively writable during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn mirror_in_place<T, L>(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, T, L>,
    axis: Axis,
) -> Result<()>
where
    T: MirrorInPlace<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::mirror_in_place(stream_context, image, axis) }
}

impl_mirror_batch!(
    "Mirrors corresponding source and destination images about `axis`.",
    mirror_batch_f32_c1,
    f32,
    C1,
    nppiMirrorBatch_32f_C1R_Ctx
);

impl_mirror_batch!(
    "Mirrors corresponding source and destination images about `axis`.",
    mirror_batch_f32_c3,
    f32,
    C3,
    nppiMirrorBatch_32f_C3R_Ctx
);

impl_mirror_batch!(
    "Mirrors corresponding source and destination images about `axis`.",
    mirror_batch_f32_c4,
    f32,
    C4,
    nppiMirrorBatch_32f_C4R_Ctx
);

impl_mirror_batch!(
    "Mirrors corresponding source and destination images about `axis`. The destination alpha channel is not processed.",
    mirror_batch_f32_ac4,
    f32,
    AC4,
    nppiMirrorBatch_32f_AC4R_Ctx
);

impl_mirror_batch_in_place!(
    "Mirrors each image about `axis` in place.",
    mirror_batch_f32_c1_in_place,
    f32,
    C1,
    nppiMirrorBatch_32f_C1IR_Ctx
);

impl_mirror_batch_in_place!(
    "Mirrors each image about `axis` in place.",
    mirror_batch_f32_c3_in_place,
    f32,
    C3,
    nppiMirrorBatch_32f_C3IR_Ctx
);

impl_mirror_batch_in_place!(
    "Mirrors each image about `axis` in place.",
    mirror_batch_f32_c4_in_place,
    f32,
    C4,
    nppiMirrorBatch_32f_C4IR_Ctx
);

impl_mirror_batch_in_place!(
    "Mirrors each image about `axis` in place. The destination alpha channel is not processed.",
    mirror_batch_f32_ac4_in_place,
    f32,
    AC4,
    nppiMirrorBatch_32f_AC4IR_Ctx
);

/// Element and channel-layout dispatch for [`mirror_batch`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait MirrorBatch<L: ChannelLayout>: DataTypeLike {
    /// Mirrors a batch of images about `axis`.
    ///
    /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
    /// The processed ROI uses the smallest width and smallest height in the batch.
    /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Batches require at least two entries and a count representable by the native interface.
    /// Source and destination batch lengths must match.
    /// Each source/destination pair must have matching image sizes.
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
    unsafe fn mirror_batch(
        stream_context: &StreamContext,
        axis: Axis,
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageOutputView<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_mirror_batch_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl MirrorBatch<$layout> for $ty {
            unsafe fn mirror_batch(
                stream_context: &StreamContext,
                axis: Axis,
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageOutputView<'_, Self, $layout>],
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, axis, sources, destinations) }
            }
        }
    };
}

impl_mirror_batch_dispatch!(f32, C1, mirror_batch_f32_c1);

impl_mirror_batch_dispatch!(f32, C3, mirror_batch_f32_c3);

impl_mirror_batch_dispatch!(f32, C4, mirror_batch_f32_c4);

impl_mirror_batch_dispatch!(f32, AC4, mirror_batch_f32_ac4);

/// Mirrors a batch of images about `axis`.
///
/// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
/// The processed ROI uses the smallest width and smallest height in the batch.
/// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Batches require at least two entries and a count representable by the native interface.
/// Source and destination batch lengths must match.
/// Each source/destination pair must have matching image sizes.
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
pub unsafe fn mirror_batch<T, L>(
    stream_context: &StreamContext,
    axis: Axis,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<()>
where
    T: MirrorBatch<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::mirror_batch(stream_context, axis, sources, destinations) }
}

/// Element and channel-layout dispatch for [`mirror_batch_in_place`].
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait MirrorBatchInPlace<L: ChannelLayout>: DataTypeLike {
    /// Mirrors a batch of images about `axis` in place.
    ///
    /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
    /// The processed ROI uses the smallest width and smallest height in the batch.
    /// Pixels and channels not processed by the native routine retain their existing values.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Batches require at least two entries and a count representable by the native interface.
    /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
    /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Images must contain initialized values ready for the selected stream and remain exclusively writable during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
    unsafe fn mirror_batch_in_place(
        stream_context: &StreamContext,
        axis: Axis,
        images: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_mirror_batch_in_place_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        unsafe impl MirrorBatchInPlace<$layout> for $ty {
            unsafe fn mirror_batch_in_place(
                stream_context: &StreamContext,
                axis: Axis,
                images: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $function(stream_context, axis, images) }
            }
        }
    };
}

impl_mirror_batch_in_place_dispatch!(f32, C1, mirror_batch_f32_c1_in_place);

impl_mirror_batch_in_place_dispatch!(f32, C3, mirror_batch_f32_c3_in_place);

impl_mirror_batch_in_place_dispatch!(f32, C4, mirror_batch_f32_c4_in_place);

impl_mirror_batch_in_place_dispatch!(f32, AC4, mirror_batch_f32_ac4_in_place);

/// Mirrors a batch of images about `axis` in place.
///
/// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
/// The processed ROI uses the smallest width and smallest height in the batch.
/// Pixels and channels not processed by the native routine retain their existing values.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
/// Batches require at least two entries and a count representable by the native interface.
/// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
/// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Images must contain initialized values ready for the selected stream and remain exclusively writable during native use.
/// Keep image allocations valid until this call completes its native work.
/// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
pub unsafe fn mirror_batch_in_place<T, L>(
    stream_context: &StreamContext,
    axis: Axis,
    images: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: MirrorBatchInPlace<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::mirror_batch_in_place(stream_context, axis, images) }
}
