use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange_validation::{validate_channel_index, validate_same_size},
        view::{AC4, C1, C2, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Copies the zero-based source `channel` into the single-channel destination.
/// Calls [`sys::nppiCopy_8u_C3C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn extract_channel_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiCopy_8u_C3C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3C1R_Ctx(
            source.as_ptr().add(channel),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the zero-based source `channel` into the single-channel destination.
/// Calls [`sys::nppiCopy_8u_C4C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn extract_channel_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiCopy_8u_C4C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4C1R_Ctx(
            source.as_ptr().add(channel),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_extract_channel!(extract_channel_u16_c3, u16, C3, 3, nppiCopy_16u_C3C1R_Ctx);

impl_extract_channel!(extract_channel_u16_c4, u16, C4, 4, nppiCopy_16u_C4C1R_Ctx);

impl_extract_channel!(extract_channel_i16_c3, i16, C3, 3, nppiCopy_16s_C3C1R_Ctx);

impl_extract_channel!(extract_channel_i16_c4, i16, C4, 4, nppiCopy_16s_C4C1R_Ctx);

impl_extract_channel!(extract_channel_i32_c3, i32, C3, 3, nppiCopy_32s_C3C1R_Ctx);

impl_extract_channel!(extract_channel_i32_c4, i32, C4, 4, nppiCopy_32s_C4C1R_Ctx);

impl_extract_channel!(extract_channel_f32_c2, f32, C2, 2, nppiCopy_32f_C2C1R_Ctx);

impl_extract_channel!(extract_channel_f32_c3, f32, C3, 3, nppiCopy_32f_C3C1R_Ctx);

impl_extract_channel!(extract_channel_f32_c4, f32, C4, 4, nppiCopy_32f_C4C1R_Ctx);

impl_generic_extract_channel_operation!(
    ExtractChannelC2,
    extract_channel_c2,
    C2,
    [f32 => extract_channel_f32_c2]
);

impl_generic_extract_channel_operation!(
    ExtractChannelC3,
    extract_channel_c3,
    C3,
    [
        u8 => extract_channel_u8_c3,
        u16 => extract_channel_u16_c3,
        i16 => extract_channel_i16_c3,
        i32 => extract_channel_i32_c3,
        f32 => extract_channel_f32_c3,
    ]
);

impl_generic_extract_channel_operation!(
    ExtractChannelC4,
    extract_channel_c4,
    C4,
    [
        u8 => extract_channel_u8_c4,
        u16 => extract_channel_u16_c4,
        i16 => extract_channel_i16_c4,
        i32 => extract_channel_i32_c4,
        f32 => extract_channel_f32_c4,
    ]
);

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_8u_C3CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_u8_c3(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, u8, C3>,
    value: u8,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiSet_8u_C3CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_8u_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_8u_C4CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_u8_c4(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, u8, C4>,
    value: u8,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiSet_8u_C4CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_8u_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_16u_C3CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_u16_c3(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, u16, C3>,
    value: u16,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiSet_16u_C3CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_16u_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_16u_C4CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_u16_c4(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, u16, C4>,
    value: u16,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiSet_16u_C4CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_16u_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_16s_C3CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_i16_c3(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, i16, C3>,
    value: i16,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiSet_16s_C3CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_16s_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_16s_C4CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_i16_c4(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, i16, C4>,
    value: i16,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiSet_16s_C4CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_16s_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_32s_C3CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_i32_c3(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, i32, C3>,
    value: i32,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiSet_32s_C3CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_32s_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_32s_C4CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_i32_c4(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, i32, C4>,
    value: i32,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiSet_32s_C4CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_32s_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_32f_C3CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_f32_c3(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, f32, C3>,
    value: f32,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiSet_32f_C3CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_32f_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Fills the zero-based destination `channel` with `value`.
/// Calls [`sys::nppiSet_32f_C4CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Keep destination storage exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn set_channel_f32_c4(
    stream_context: &StreamContext,
    destination: &mut ImageOutputView<'_, f32, C4>,
    value: f32,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(destination)?;

    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiSet_32f_C4CR_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nppiSet_32f_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_generic_set_channel_operation!(
    SetChannelC3,
    set_channel_c3,
    C3,
    [
        u8 => set_channel_u8_c3,
        u16 => set_channel_u16_c3,
        i16 => set_channel_i16_c3,
        i32 => set_channel_i32_c3,
        f32 => set_channel_f32_c3,
    ]
);

impl_generic_set_channel_operation!(
    SetChannelC4,
    set_channel_c4,
    C4,
    [
        u8 => set_channel_u8_c4,
        u16 => set_channel_u16_c4,
        i16 => set_channel_i16_c4,
        i32 => set_channel_i32_c4,
        f32 => set_channel_f32_c4,
    ]
);

/// Copies the single-channel source into the zero-based destination `channel`.
/// Calls [`sys::nppiCopy_8u_C1C3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn insert_channel_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C3>,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 3)?;

    // SAFETY: `nppiCopy_8u_C1C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C1C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().add(channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the single-channel source into the zero-based destination `channel`.
/// Calls [`sys::nppiCopy_8u_C1C4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn insert_channel_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C4>,
    channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 4)?;

    // SAFETY: `nppiCopy_8u_C1C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C1C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().add(channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_insert_channel!(insert_channel_u16_c3, u16, C3, 3, nppiCopy_16u_C1C3R_Ctx);

impl_insert_channel!(insert_channel_u16_c4, u16, C4, 4, nppiCopy_16u_C1C4R_Ctx);

impl_insert_channel!(insert_channel_i16_c3, i16, C3, 3, nppiCopy_16s_C1C3R_Ctx);

impl_insert_channel!(insert_channel_i16_c4, i16, C4, 4, nppiCopy_16s_C1C4R_Ctx);

impl_insert_channel!(insert_channel_i32_c3, i32, C3, 3, nppiCopy_32s_C1C3R_Ctx);

impl_insert_channel!(insert_channel_i32_c4, i32, C4, 4, nppiCopy_32s_C1C4R_Ctx);

impl_insert_channel!(insert_channel_f32_c2, f32, C2, 2, nppiCopy_32f_C1C2R_Ctx);

impl_insert_channel!(insert_channel_f32_c3, f32, C3, 3, nppiCopy_32f_C1C3R_Ctx);

impl_insert_channel!(insert_channel_f32_c4, f32, C4, 4, nppiCopy_32f_C1C4R_Ctx);

impl_generic_insert_channel_operation!(
    InsertChannelC2,
    insert_channel_c2,
    C2,
    [f32 => insert_channel_f32_c2]
);

impl_generic_insert_channel_operation!(
    InsertChannelC3,
    insert_channel_c3,
    C3,
    [
        u8 => insert_channel_u8_c3,
        u16 => insert_channel_u16_c3,
        i16 => insert_channel_i16_c3,
        i32 => insert_channel_i32_c3,
        f32 => insert_channel_f32_c3,
    ]
);

impl_generic_insert_channel_operation!(
    InsertChannelC4,
    insert_channel_c4,
    C4,
    [
        u8 => insert_channel_u8_c4,
        u16 => insert_channel_u16_c4,
        i16 => insert_channel_i16_c4,
        i32 => insert_channel_i32_c4,
        f32 => insert_channel_f32_c4,
    ]
);

/// Duplicates the single-channel source into the processed destination channels.
/// Calls [`sys::nppiDup_8u_C1C3R_Ctx`].
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
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn duplicate_channel_u8_c1_to_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C3>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiDup_8u_C1C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiDup_8u_C1C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Duplicates the single-channel source into the processed destination channels.
/// Calls [`sys::nppiDup_8u_C1C4R_Ctx`].
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
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn duplicate_channel_u8_c1_to_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiDup_8u_C1C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiDup_8u_C1C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Duplicates the single-channel source into the processed destination channels.
/// Calls [`sys::nppiDup_8u_C1AC4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// The destination alpha channel is not processed.
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
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn duplicate_channel_u8_c1_to_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, AC4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiDup_8u_C1AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiDup_8u_C1AC4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Duplicates the single-channel source into the processed destination channels.
/// Calls [`sys::nppiDup_16u_C1C3R_Ctx`].
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
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn duplicate_channel_u16_c1_to_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageOutputView<'_, u16, C3>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiDup_16u_C1C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiDup_16u_C1C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Duplicates the single-channel source into the processed destination channels.
/// Calls [`sys::nppiDup_16u_C1C4R_Ctx`].
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
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn duplicate_channel_u16_c1_to_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageOutputView<'_, u16, C4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiDup_16u_C1C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiDup_16u_C1C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Duplicates the single-channel source into the processed destination channels.
/// Calls [`sys::nppiDup_16u_C1AC4R_Ctx`].
///
/// Enqueues work on the selected stream.
/// The destination alpha channel is not processed.
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
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn duplicate_channel_u16_c1_to_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageOutputView<'_, u16, AC4>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiDup_16u_C1AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiDup_16u_C1AC4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

macro_rules! impl_duplicate_channel {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Duplicates the single-channel source into the processed destination channels.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
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
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_duplicate_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the source and destination native element ABI and channel layout.
        /// They must duplicate every processed destination channel and uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Duplicates the single-channel source into the processed destination channels.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
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
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )+

        /// Duplicates the single-channel source into the processed destination channels.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
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
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination)
            }
}
    };
}

impl_duplicate_channel!(
    duplicate_channel_i16_c1_to_c3,
    i16,
    C3,
    nppiDup_16s_C1C3R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_i16_c1_to_c4,
    i16,
    C4,
    nppiDup_16s_C1C4R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_i16_c1_to_ac4,
    i16,
    AC4,
    nppiDup_16s_C1AC4R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_i32_c1_to_c3,
    i32,
    C3,
    nppiDup_32s_C1C3R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_i32_c1_to_c4,
    i32,
    C4,
    nppiDup_32s_C1C4R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_i32_c1_to_ac4,
    i32,
    AC4,
    nppiDup_32s_C1AC4R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_f32_c1_to_c3,
    f32,
    C3,
    nppiDup_32f_C1C3R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_f32_c1_to_c4,
    f32,
    C4,
    nppiDup_32f_C1C4R_Ctx
);

impl_duplicate_channel!(
    duplicate_channel_f32_c1_to_ac4,
    f32,
    AC4,
    nppiDup_32f_C1AC4R_Ctx
);

impl_generic_duplicate_channel_operation!(
    DuplicateChannelC1ToC3,
    duplicate_channel_c1_to_c3,
    C3,
    [
        u8 => duplicate_channel_u8_c1_to_c3,
        u16 => duplicate_channel_u16_c1_to_c3,
        i16 => duplicate_channel_i16_c1_to_c3,
        i32 => duplicate_channel_i32_c1_to_c3,
        f32 => duplicate_channel_f32_c1_to_c3,
    ]
);

impl_generic_duplicate_channel_operation!(
    DuplicateChannelC1ToC4,
    duplicate_channel_c1_to_c4,
    C4,
    [
        u8 => duplicate_channel_u8_c1_to_c4,
        u16 => duplicate_channel_u16_c1_to_c4,
        i16 => duplicate_channel_i16_c1_to_c4,
        i32 => duplicate_channel_i32_c1_to_c4,
        f32 => duplicate_channel_f32_c1_to_c4,
    ]
);

impl_generic_duplicate_channel_operation!(
    DuplicateChannelC1ToAC4,
    duplicate_channel_c1_to_ac4,
    AC4,
    [
        u8 => duplicate_channel_u8_c1_to_ac4,
        u16 => duplicate_channel_u16_c1_to_ac4,
        i16 => duplicate_channel_i16_c1_to_ac4,
        i32 => duplicate_channel_i32_c1_to_ac4,
        f32 => duplicate_channel_f32_c1_to_ac4,
    ]
);

/// Copies the zero-based `source_channel` into the zero-based `destination_channel`.
/// Calls [`sys::nppiCopy_8u_C3CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_channel_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_channel: usize,
    destination: &mut ImageOutputView<'_, u8, C3>,
    destination_channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(source_channel, 3)?;
    validate_channel_index(destination_channel, 3)?;

    // SAFETY: `nppiCopy_8u_C3CR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3CR_Ctx(
            source.as_ptr().add(source_channel),
            source.step(),
            destination.as_mut_ptr().add(destination_channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Copies the zero-based `source_channel` into the zero-based `destination_channel`.
/// Calls [`sys::nppiCopy_8u_C4CR_Ctx`].
///
/// Enqueues work on the selected stream.
/// Other destination channels remain unwritten.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Channel indices must be smaller than the channel count of the corresponding packed image.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_channel_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    source_channel: usize,
    destination: &mut ImageOutputView<'_, u8, C4>,
    destination_channel: usize,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(source_channel, 4)?;
    validate_channel_index(destination_channel, 4)?;

    // SAFETY: `nppiCopy_8u_C4CR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4CR_Ctx(
            source.as_ptr().add(source_channel),
            source.step(),
            destination.as_mut_ptr().add(destination_channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_channel!(copy_channel_u16_c3, u16, C3, 3, nppiCopy_16u_C3CR_Ctx);

impl_copy_channel!(copy_channel_u16_c4, u16, C4, 4, nppiCopy_16u_C4CR_Ctx);

impl_copy_channel!(copy_channel_i16_c3, i16, C3, 3, nppiCopy_16s_C3CR_Ctx);

impl_copy_channel!(copy_channel_i16_c4, i16, C4, 4, nppiCopy_16s_C4CR_Ctx);

impl_copy_channel!(copy_channel_i32_c3, i32, C3, 3, nppiCopy_32s_C3CR_Ctx);

impl_copy_channel!(copy_channel_i32_c4, i32, C4, 4, nppiCopy_32s_C4CR_Ctx);

impl_copy_channel!(copy_channel_f32_c3, f32, C3, 3, nppiCopy_32f_C3CR_Ctx);

impl_copy_channel!(copy_channel_f32_c4, f32, C4, 4, nppiCopy_32f_C4CR_Ctx);

impl_generic_copy_channel_operation!(
    CopyChannelC3,
    copy_channel_c3,
    C3,
    [
        u8 => copy_channel_u8_c3,
        u16 => copy_channel_u16_c3,
        i16 => copy_channel_i16_c3,
        i32 => copy_channel_i32_c3,
        f32 => copy_channel_f32_c3,
    ]
);

impl_generic_copy_channel_operation!(
    CopyChannelC4,
    copy_channel_c4,
    C4,
    [
        u8 => copy_channel_u8_c4,
        u16 => copy_channel_u16_c4,
        i16 => copy_channel_i16_c4,
        i32 => copy_channel_i32_c4,
        f32 => copy_channel_f32_c4,
    ]
);
