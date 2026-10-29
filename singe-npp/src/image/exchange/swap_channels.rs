use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange_validation::{
            validate_channel_order, validate_channel_order_with_fill, validate_same_size,
        },
        view::{AC4, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Reorders image channels using the zero-based source indices in `order`.
/// Calls [`sys::nppiSwapChannels_8u_C3R_Ctx`].
/// Destination channel `i` receives source channel `order[i]`.
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Entries of `order` must be distinct indices in `0..3`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn swap_channels_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageOutputView<'_, u8, C3>,
    order: [i32; 3],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_order(&order, 3)?;

    // SAFETY: `nppiSwapChannels_8u_C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSwapChannels_8u_C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            order.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Reorders image channels in place using the zero-based source indices in `order`.
/// Calls [`sys::nppiSwapChannels_8u_C3IR_Ctx`].
/// Destination channel `i` receives source channel `order[i]`.
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Entries of `order` must be distinct indices in `0..3`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image samples must be initialized and ready when read by the selected stream.
/// Keep the image exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn swap_channels_u8_c3_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, u8, C3>,
    order: [i32; 3],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;

    validate_channel_order(&order, 3)?;

    // SAFETY: `nppiSwapChannels_8u_C3IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSwapChannels_8u_C3IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            order.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Reorders image channels using the zero-based source indices in `order`.
/// Calls [`sys::nppiSwapChannels_8u_C4R_Ctx`].
/// Destination channel `i` receives source channel `order[i]`.
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Source and destination image sizes must match.
/// Entries of `order` must be distinct indices in `0..4`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn swap_channels_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageOutputView<'_, u8, C4>,
    order: [i32; 4],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;
    validate_channel_order(&order, 4)?;

    // SAFETY: `nppiSwapChannels_8u_C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSwapChannels_8u_C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            order.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Reorders image channels in place using the zero-based source indices in `order`.
/// Calls [`sys::nppiSwapChannels_8u_C4IR_Ctx`].
/// Destination channel `i` receives source channel `order[i]`.
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Entries of `order` must be distinct indices in `0..4`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image samples must be initialized and ready when read by the selected stream.
/// Keep the image exclusively writable during native use.
/// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn swap_channels_u8_c4_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, u8, C4>,
    order: [i32; 4],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;

    validate_channel_order(&order, 4)?;

    // SAFETY: `nppiSwapChannels_8u_C4IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSwapChannels_8u_C4IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            order.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_swap_channels_c4_to_c3!(swap_channels_u8_c4_to_c3, u8, nppiSwapChannels_8u_C4C3R_Ctx);

impl_swap_channels_c3_to_c4!(swap_channels_u8_c3_to_c4, u8, nppiSwapChannels_8u_C3C4R_Ctx);

impl_swap_channels_ac4!(swap_channels_u8_ac4, u8, nppiSwapChannels_8u_AC4R_Ctx);

impl_swap_channels!(
    swap_channels_u16_c3,
    u16,
    C3,
    3,
    nppiSwapChannels_16u_C3R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiSwapChannels_16u_C3IR_Ctx
);

impl_swap_channels!(
    swap_channels_u16_c4,
    u16,
    C4,
    4,
    nppiSwapChannels_16u_C4R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_u16_c4_in_place,
    u16,
    C4,
    4,
    nppiSwapChannels_16u_C4IR_Ctx
);

impl_swap_channels!(
    swap_channels_i16_c3,
    i16,
    C3,
    3,
    nppiSwapChannels_16s_C3R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiSwapChannels_16s_C3IR_Ctx
);

impl_swap_channels!(
    swap_channels_i16_c4,
    i16,
    C4,
    4,
    nppiSwapChannels_16s_C4R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_i16_c4_in_place,
    i16,
    C4,
    4,
    nppiSwapChannels_16s_C4IR_Ctx
);

impl_swap_channels!(
    swap_channels_i32_c3,
    i32,
    C3,
    3,
    nppiSwapChannels_32s_C3R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_i32_c3_in_place,
    i32,
    C3,
    3,
    nppiSwapChannels_32s_C3IR_Ctx
);

impl_swap_channels!(
    swap_channels_i32_c4,
    i32,
    C4,
    4,
    nppiSwapChannels_32s_C4R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_i32_c4_in_place,
    i32,
    C4,
    4,
    nppiSwapChannels_32s_C4IR_Ctx
);

impl_swap_channels!(
    swap_channels_f32_c3,
    f32,
    C3,
    3,
    nppiSwapChannels_32f_C3R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_f32_c3_in_place,
    f32,
    C3,
    3,
    nppiSwapChannels_32f_C3IR_Ctx
);

impl_swap_channels!(
    swap_channels_f32_c4,
    f32,
    C4,
    4,
    nppiSwapChannels_32f_C4R_Ctx
);

impl_swap_channels_in_place!(
    swap_channels_f32_c4_in_place,
    f32,
    C4,
    4,
    nppiSwapChannels_32f_C4IR_Ctx
);

impl_swap_channels_c4_to_c3!(
    swap_channels_u16_c4_to_c3,
    u16,
    nppiSwapChannels_16u_C4C3R_Ctx
);

impl_swap_channels_c3_to_c4!(
    swap_channels_u16_c3_to_c4,
    u16,
    nppiSwapChannels_16u_C3C4R_Ctx
);

impl_swap_channels_ac4!(swap_channels_u16_ac4, u16, nppiSwapChannels_16u_AC4R_Ctx);

impl_swap_channels_c4_to_c3!(
    swap_channels_i16_c4_to_c3,
    i16,
    nppiSwapChannels_16s_C4C3R_Ctx
);

impl_swap_channels_c3_to_c4!(
    swap_channels_i16_c3_to_c4,
    i16,
    nppiSwapChannels_16s_C3C4R_Ctx
);

impl_swap_channels_ac4!(swap_channels_i16_ac4, i16, nppiSwapChannels_16s_AC4R_Ctx);

impl_swap_channels_c4_to_c3!(
    swap_channels_i32_c4_to_c3,
    i32,
    nppiSwapChannels_32s_C4C3R_Ctx
);

impl_swap_channels_c3_to_c4!(
    swap_channels_i32_c3_to_c4,
    i32,
    nppiSwapChannels_32s_C3C4R_Ctx
);

impl_swap_channels_ac4!(swap_channels_i32_ac4, i32, nppiSwapChannels_32s_AC4R_Ctx);

impl_swap_channels_c4_to_c3!(
    swap_channels_f32_c4_to_c3,
    f32,
    nppiSwapChannels_32f_C4C3R_Ctx
);

impl_swap_channels_c3_to_c4!(
    swap_channels_f32_c3_to_c4,
    f32,
    nppiSwapChannels_32f_C3C4R_Ctx
);

impl_swap_channels_ac4!(swap_channels_f32_ac4, f32, nppiSwapChannels_32f_AC4R_Ctx);

impl_generic_swap_channels_operation!(
    SwapChannelsC3,
    swap_channels_c3,
    C3,
    3,
    [
        u8 => swap_channels_u8_c3,
        u16 => swap_channels_u16_c3,
        i16 => swap_channels_i16_c3,
        i32 => swap_channels_i32_c3,
        f32 => swap_channels_f32_c3,
    ]
);

impl_generic_swap_channels_in_place_operation!(
    SwapChannelsC3InPlace,
    swap_channels_c3_in_place,
    C3,
    3,
    [
        u8 => swap_channels_u8_c3_in_place,
        u16 => swap_channels_u16_c3_in_place,
        i16 => swap_channels_i16_c3_in_place,
        i32 => swap_channels_i32_c3_in_place,
        f32 => swap_channels_f32_c3_in_place,
    ]
);

impl_generic_swap_channels_operation!(
    SwapChannelsC4,
    swap_channels_c4,
    C4,
    4,
    [
        u8 => swap_channels_u8_c4,
        u16 => swap_channels_u16_c4,
        i16 => swap_channels_i16_c4,
        i32 => swap_channels_i32_c4,
        f32 => swap_channels_f32_c4,
    ]
);

impl_generic_swap_channels_in_place_operation!(
    SwapChannelsC4InPlace,
    swap_channels_c4_in_place,
    C4,
    4,
    [
        u8 => swap_channels_u8_c4_in_place,
        u16 => swap_channels_u16_c4_in_place,
        i16 => swap_channels_i16_c4_in_place,
        i32 => swap_channels_i32_c4_in_place,
        f32 => swap_channels_f32_c4_in_place,
    ]
);

impl_generic_swap_channels_c4_to_c3_operation!(
    SwapChannelsC4ToC3,
    swap_channels_c4_to_c3,
    [
        u8 => swap_channels_u8_c4_to_c3,
        u16 => swap_channels_u16_c4_to_c3,
        i16 => swap_channels_i16_c4_to_c3,
        i32 => swap_channels_i32_c4_to_c3,
        f32 => swap_channels_f32_c4_to_c3,
    ]
);

impl_generic_swap_channels_c3_to_c4_operation!(
    SwapChannelsC3ToC4,
    swap_channels_c3_to_c4,
    [
        u8 => swap_channels_u8_c3_to_c4,
        u16 => swap_channels_u16_c3_to_c4,
        i16 => swap_channels_i16_c3_to_c4,
        i32 => swap_channels_i32_c3_to_c4,
        f32 => swap_channels_f32_c3_to_c4,
    ]
);

impl_generic_swap_channels_operation!(
    SwapChannelsAC4,
    swap_channels_ac4,
    AC4,
    3,
    [
        u8 => swap_channels_u8_ac4,
        u16 => swap_channels_u16_ac4,
        i16 => swap_channels_i16_ac4,
        i32 => swap_channels_i32_ac4,
        f32 => swap_channels_f32_ac4,
    ]
);
