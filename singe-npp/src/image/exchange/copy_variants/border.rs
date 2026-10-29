use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange_validation::validate_border_placement,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.
///
/// Calls [`sys::nppiCopyReplicateBorder_8u_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_replicate_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyReplicateBorder_8u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyReplicateBorder_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.
///
/// Calls [`sys::nppiCopyReplicateBorder_8u_C3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_replicate_border_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageOutputView<'_, u8, C3>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyReplicateBorder_8u_C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyReplicateBorder_8u_C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.
///
/// Calls [`sys::nppiCopyReplicateBorder_8u_C4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_replicate_border_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageOutputView<'_, u8, C4>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyReplicateBorder_8u_C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyReplicateBorder_8u_C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels. The destination alpha channel is not processed.
///
/// Calls [`sys::nppiCopyReplicateBorder_8u_AC4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_replicate_border_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageOutputView<'_, u8, AC4>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyReplicateBorder_8u_AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyReplicateBorder_8u_AC4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_u16_c1,
    u16,
    C1,
    nppiCopyReplicateBorder_16u_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_u16_c3,
    u16,
    C3,
    nppiCopyReplicateBorder_16u_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_u16_c4,
    u16,
    C4,
    nppiCopyReplicateBorder_16u_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels. The destination alpha channel is not processed.",
    copy_replicate_border_u16_ac4,
    u16,
    AC4,
    nppiCopyReplicateBorder_16u_AC4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_i16_c1,
    i16,
    C1,
    nppiCopyReplicateBorder_16s_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_i16_c3,
    i16,
    C3,
    nppiCopyReplicateBorder_16s_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_i16_c4,
    i16,
    C4,
    nppiCopyReplicateBorder_16s_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels. The destination alpha channel is not processed.",
    copy_replicate_border_i16_ac4,
    i16,
    AC4,
    nppiCopyReplicateBorder_16s_AC4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_i32_c1,
    i32,
    C1,
    nppiCopyReplicateBorder_32s_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_i32_c3,
    i32,
    C3,
    nppiCopyReplicateBorder_32s_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_i32_c4,
    i32,
    C4,
    nppiCopyReplicateBorder_32s_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels. The destination alpha channel is not processed.",
    copy_replicate_border_i32_ac4,
    i32,
    AC4,
    nppiCopyReplicateBorder_32s_AC4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_f32_c1,
    f32,
    C1,
    nppiCopyReplicateBorder_32f_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_f32_c3,
    f32,
    C3,
    nppiCopyReplicateBorder_32f_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    copy_replicate_border_f32_c4,
    f32,
    C4,
    nppiCopyReplicateBorder_32f_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels. The destination alpha channel is not processed.",
    copy_replicate_border_f32_ac4,
    f32,
    AC4,
    nppiCopyReplicateBorder_32f_AC4R_Ctx
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    CopyReplicateBorderC1,
    copy_replicate_border_c1,
    C1,
    [
        u8 => copy_replicate_border_u8_c1,
        u16 => copy_replicate_border_u16_c1,
        i16 => copy_replicate_border_i16_c1,
        i32 => copy_replicate_border_i32_c1,
        f32 => copy_replicate_border_f32_c1,
    ]
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    CopyReplicateBorderC3,
    copy_replicate_border_c3,
    C3,
    [
        u8 => copy_replicate_border_u8_c3,
        u16 => copy_replicate_border_u16_c3,
        i16 => copy_replicate_border_i16_c3,
        i32 => copy_replicate_border_i32_c3,
        f32 => copy_replicate_border_f32_c3,
    ]
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels.",
    CopyReplicateBorderC4,
    copy_replicate_border_c4,
    C4,
    [
        u8 => copy_replicate_border_u8_c4,
        u16 => copy_replicate_border_u16_c4,
        i16 => copy_replicate_border_i16_c4,
        i32 => copy_replicate_border_i32_c4,
        f32 => copy_replicate_border_f32_c4,
    ]
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by replicating the source edge pixels. The destination alpha channel is not processed.",
    CopyReplicateBorderAC4,
    copy_replicate_border_ac4,
    AC4,
    [
        u8 => copy_replicate_border_u8_ac4,
        u16 => copy_replicate_border_u16_ac4,
        i16 => copy_replicate_border_i16_ac4,
        i32 => copy_replicate_border_i32_ac4,
        f32 => copy_replicate_border_f32_ac4,
    ]
);

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.
///
/// Calls [`sys::nppiCopyWrapBorder_8u_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_wrap_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyWrapBorder_8u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyWrapBorder_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.
///
/// Calls [`sys::nppiCopyWrapBorder_8u_C3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_wrap_border_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageOutputView<'_, u8, C3>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyWrapBorder_8u_C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyWrapBorder_8u_C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.
///
/// Calls [`sys::nppiCopyWrapBorder_8u_C4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_wrap_border_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageOutputView<'_, u8, C4>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyWrapBorder_8u_C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyWrapBorder_8u_C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically. The destination alpha channel is not processed.
///
/// Calls [`sys::nppiCopyWrapBorder_8u_AC4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_wrap_border_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageOutputView<'_, u8, AC4>,
    top: usize,
    left: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    // SAFETY: `nppiCopyWrapBorder_8u_AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopyWrapBorder_8u_AC4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_u16_c1,
    u16,
    C1,
    nppiCopyWrapBorder_16u_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_u16_c3,
    u16,
    C3,
    nppiCopyWrapBorder_16u_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_u16_c4,
    u16,
    C4,
    nppiCopyWrapBorder_16u_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically. The destination alpha channel is not processed.",
    copy_wrap_border_u16_ac4,
    u16,
    AC4,
    nppiCopyWrapBorder_16u_AC4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_i16_c1,
    i16,
    C1,
    nppiCopyWrapBorder_16s_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_i16_c3,
    i16,
    C3,
    nppiCopyWrapBorder_16s_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_i16_c4,
    i16,
    C4,
    nppiCopyWrapBorder_16s_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically. The destination alpha channel is not processed.",
    copy_wrap_border_i16_ac4,
    i16,
    AC4,
    nppiCopyWrapBorder_16s_AC4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_i32_c1,
    i32,
    C1,
    nppiCopyWrapBorder_32s_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_i32_c3,
    i32,
    C3,
    nppiCopyWrapBorder_32s_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_i32_c4,
    i32,
    C4,
    nppiCopyWrapBorder_32s_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically. The destination alpha channel is not processed.",
    copy_wrap_border_i32_ac4,
    i32,
    AC4,
    nppiCopyWrapBorder_32s_AC4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_f32_c1,
    f32,
    C1,
    nppiCopyWrapBorder_32f_C1R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_f32_c3,
    f32,
    C3,
    nppiCopyWrapBorder_32f_C3R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    copy_wrap_border_f32_c4,
    f32,
    C4,
    nppiCopyWrapBorder_32f_C4R_Ctx
);

impl_copy_border!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically. The destination alpha channel is not processed.",
    copy_wrap_border_f32_ac4,
    f32,
    AC4,
    nppiCopyWrapBorder_32f_AC4R_Ctx
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    CopyWrapBorderC1,
    copy_wrap_border_c1,
    C1,
    [
        u8 => copy_wrap_border_u8_c1,
        u16 => copy_wrap_border_u16_c1,
        i16 => copy_wrap_border_i16_c1,
        i32 => copy_wrap_border_i32_c1,
        f32 => copy_wrap_border_f32_c1,
    ]
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    CopyWrapBorderC3,
    copy_wrap_border_c3,
    C3,
    [
        u8 => copy_wrap_border_u8_c3,
        u16 => copy_wrap_border_u16_c3,
        i16 => copy_wrap_border_i16_c3,
        i32 => copy_wrap_border_i32_c3,
        f32 => copy_wrap_border_f32_c3,
    ]
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically.",
    CopyWrapBorderC4,
    copy_wrap_border_c4,
    C4,
    [
        u8 => copy_wrap_border_u8_c4,
        u16 => copy_wrap_border_u16_c4,
        i16 => copy_wrap_border_i16_c4,
        i32 => copy_wrap_border_i32_c4,
        f32 => copy_wrap_border_f32_c4,
    ]
);

impl_generic_copy_border_operation!(
    "Copies the source image into the destination at pixel offset `(left, top)` and fills the surrounding border by wrapping the source image periodically. The destination alpha channel is not processed.",
    CopyWrapBorderAC4,
    copy_wrap_border_ac4,
    AC4,
    [
        u8 => copy_wrap_border_u8_ac4,
        u16 => copy_wrap_border_u16_ac4,
        i16 => copy_wrap_border_i16_ac4,
        i32 => copy_wrap_border_i32_ac4,
        f32 => copy_wrap_border_f32_ac4,
    ]
);
