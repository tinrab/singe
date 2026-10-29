use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange_validation::validate_same_size,
        view::{C3, C4, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Deinterleaves the packed source image into one destination plane per channel.
/// Calls [`sys::nppiCopy_8u_C3P3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// All planes must satisfy the native common-size and common-byte-stride requirements.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_u8_c3_to_p3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut PlanarImageOutputView<'_, u8, 3>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    for plane in destination.planes() {
        stream_context.validate_region(plane)?;
    }

    validate_same_size(source.size(), destination.planes()[0].size())?;

    let destination_planes = destination
        .planes_mut()
        .each_mut()
        .map(ImageOutputView::as_mut_ptr);

    // SAFETY: `nppiCopy_8u_C3P3R_Ctx` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3P3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination_planes.as_ptr().cast(),
            destination.planes()[0].step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Deinterleaves the packed source image into one destination plane per channel.
/// Calls [`sys::nppiCopy_8u_C4P4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// All planes must satisfy the native common-size and common-byte-stride requirements.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_u8_c4_to_p4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut PlanarImageOutputView<'_, u8, 4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    for plane in destination.planes() {
        stream_context.validate_region(plane)?;
    }

    validate_same_size(source.size(), destination.planes()[0].size())?;

    let destination_planes = destination
        .planes_mut()
        .each_mut()
        .map(ImageOutputView::as_mut_ptr);

    // SAFETY: `nppiCopy_8u_C4P4R_Ctx` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4P4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination_planes.as_ptr().cast(),
            destination.planes()[0].step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Interleaves the source planes into the packed destination image.
/// Calls [`sys::nppiCopy_8u_P3C3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// All planes must satisfy the native common-size and common-byte-stride requirements.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_u8_p3_to_c3(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, u8, 3>,
    destination: &mut ImageOutputView<'_, u8, C3>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;
    for plane in source.planes() {
        stream_context.validate_region(plane)?;
    }

    validate_same_size(source.planes()[0].size(), destination.size())?;

    let source_planes = source.planes().map(|plane| plane.as_ptr());

    // SAFETY: `nppiCopy_8u_P3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_P3C3R_Ctx(
            source_planes.as_ptr().cast(),
            source.planes()[0].step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Interleaves the source planes into the packed destination image.
/// Calls [`sys::nppiCopy_8u_P4C4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// All planes must satisfy the native common-size and common-byte-stride requirements.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_u8_p4_to_c4(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, u8, 4>,
    destination: &mut ImageOutputView<'_, u8, C4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;
    for plane in source.planes() {
        stream_context.validate_region(plane)?;
    }

    validate_same_size(source.planes()[0].size(), destination.size())?;

    let source_planes = source.planes().map(|plane| plane.as_ptr());

    // SAFETY: `nppiCopy_8u_P4C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_P4C4R_Ctx(
            source_planes.as_ptr().cast(),
            source.planes()[0].step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_packed_to_planar!(copy_u16_c3_to_p3, u16, C3, 3, nppiCopy_16u_C3P3R_Ctx);

impl_copy_packed_to_planar!(copy_u16_c4_to_p4, u16, C4, 4, nppiCopy_16u_C4P4R_Ctx);

impl_copy_packed_to_planar!(copy_i16_c3_to_p3, i16, C3, 3, nppiCopy_16s_C3P3R_Ctx);

impl_copy_packed_to_planar!(copy_i16_c4_to_p4, i16, C4, 4, nppiCopy_16s_C4P4R_Ctx);

impl_copy_packed_to_planar!(copy_i32_c3_to_p3, i32, C3, 3, nppiCopy_32s_C3P3R_Ctx);

impl_copy_packed_to_planar!(copy_i32_c4_to_p4, i32, C4, 4, nppiCopy_32s_C4P4R_Ctx);

impl_copy_packed_to_planar!(copy_f32_c3_to_p3, f32, C3, 3, nppiCopy_32f_C3P3R_Ctx);

impl_copy_packed_to_planar!(copy_f32_c4_to_p4, f32, C4, 4, nppiCopy_32f_C4P4R_Ctx);

impl_copy_planar_to_packed!(copy_u16_p3_to_c3, u16, C3, 3, nppiCopy_16u_P3C3R_Ctx);

impl_copy_planar_to_packed!(copy_u16_p4_to_c4, u16, C4, 4, nppiCopy_16u_P4C4R_Ctx);

impl_copy_planar_to_packed!(copy_i16_p3_to_c3, i16, C3, 3, nppiCopy_16s_P3C3R_Ctx);

impl_copy_planar_to_packed!(copy_i16_p4_to_c4, i16, C4, 4, nppiCopy_16s_P4C4R_Ctx);

impl_copy_planar_to_packed!(copy_i32_p3_to_c3, i32, C3, 3, nppiCopy_32s_P3C3R_Ctx);

impl_copy_planar_to_packed!(copy_i32_p4_to_c4, i32, C4, 4, nppiCopy_32s_P4C4R_Ctx);

impl_copy_planar_to_packed!(copy_f32_p3_to_c3, f32, C3, 3, nppiCopy_32f_P3C3R_Ctx);

impl_copy_planar_to_packed!(copy_f32_p4_to_c4, f32, C4, 4, nppiCopy_32f_P4C4R_Ctx);

impl_generic_copy_packed_to_planar_operation!(
    CopyC3ToP3,
    copy_c3_to_p3,
    C3,
    3,
    [
        u8 => copy_u8_c3_to_p3,
        u16 => copy_u16_c3_to_p3,
        i16 => copy_i16_c3_to_p3,
        i32 => copy_i32_c3_to_p3,
        f32 => copy_f32_c3_to_p3,
    ]
);

impl_generic_copy_packed_to_planar_operation!(
    CopyC4ToP4,
    copy_c4_to_p4,
    C4,
    4,
    [
        u8 => copy_u8_c4_to_p4,
        u16 => copy_u16_c4_to_p4,
        i16 => copy_i16_c4_to_p4,
        i32 => copy_i32_c4_to_p4,
        f32 => copy_f32_c4_to_p4,
    ]
);

impl_generic_copy_planar_to_packed_operation!(
    CopyP3ToC3,
    copy_p3_to_c3,
    C3,
    3,
    [
        u8 => copy_u8_p3_to_c3,
        u16 => copy_u16_p3_to_c3,
        i16 => copy_i16_p3_to_c3,
        i32 => copy_i32_p3_to_c3,
        f32 => copy_f32_p3_to_c3,
    ]
);

impl_generic_copy_planar_to_packed_operation!(
    CopyP4ToC4,
    copy_p4_to_c4,
    C4,
    4,
    [
        u8 => copy_u8_p4_to_c4,
        u16 => copy_u16_p4_to_c4,
        i16 => copy_i16_p4_to_c4,
        i32 => copy_i32_p4_to_c4,
        f32 => copy_f32_p4_to_c4,
    ]
);
