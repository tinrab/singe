use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange_validation::validate_transpose_size,
        view::{C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.
///
/// Calls [`sys::nppiTranspose_8u_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Destination width must equal source height, and destination height must equal source width.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn transpose_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C1>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_transpose_size(source.size(), destination.size())?;

    // SAFETY: `nppiTranspose_8u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiTranspose_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.
///
/// Calls [`sys::nppiTranspose_8u_C3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Destination width must equal source height, and destination height must equal source width.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn transpose_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageOutputView<'_, u8, C3>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_transpose_size(source.size(), destination.size())?;

    // SAFETY: `nppiTranspose_8u_C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiTranspose_8u_C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.
///
/// Calls [`sys::nppiTranspose_8u_C4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Destination width must equal source height, and destination height must equal source width.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn transpose_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageOutputView<'_, u8, C4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_transpose_size(source.size(), destination.size())?;

    // SAFETY: `nppiTranspose_8u_C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiTranspose_8u_C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_u16_c1,
    u16,
    C1,
    nppiTranspose_16u_C1R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_u16_c3,
    u16,
    C3,
    nppiTranspose_16u_C3R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_u16_c4,
    u16,
    C4,
    nppiTranspose_16u_C4R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_i16_c1,
    i16,
    C1,
    nppiTranspose_16s_C1R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_i16_c3,
    i16,
    C3,
    nppiTranspose_16s_C3R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_i16_c4,
    i16,
    C4,
    nppiTranspose_16s_C4R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_i32_c1,
    i32,
    C1,
    nppiTranspose_32s_C1R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_i32_c3,
    i32,
    C3,
    nppiTranspose_32s_C3R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_i32_c4,
    i32,
    C4,
    nppiTranspose_32s_C4R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_f32_c1,
    f32,
    C1,
    nppiTranspose_32f_C1R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_f32_c3,
    f32,
    C3,
    nppiTranspose_32f_C3R_Ctx
);

impl_transpose!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    transpose_f32_c4,
    f32,
    C4,
    nppiTranspose_32f_C4R_Ctx
);

impl_generic_transpose_operation!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    TransposeC1, transpose, transpose_c1, C1, [
    u8 => transpose_u8_c1,
    u16 => transpose_u16_c1,
    i16 => transpose_i16_c1,
    i32 => transpose_i32_c1,
    f32 => transpose_f32_c1,
    ]
);

impl_generic_transpose_operation!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    TransposeC3, transpose, transpose_c3, C3, [
    u8 => transpose_u8_c3,
    u16 => transpose_u16_c3,
    i16 => transpose_i16_c3,
    i32 => transpose_i32_c3,
    f32 => transpose_f32_c3,
    ]
);

impl_generic_transpose_operation!(
    "Transposes the source image so destination pixel `(x, y)` receives source pixel `(y, x)`.",
    TransposeC4, transpose, transpose_c4, C4, [
    u8 => transpose_u8_c4,
    u16 => transpose_u16_c4,
    i16 => transpose_i16_c4,
    i32 => transpose_i32_c4,
    f32 => transpose_f32_c4,
    ]
);
