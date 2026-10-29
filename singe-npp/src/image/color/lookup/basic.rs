use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{
            lookup_table_channel_descriptors, lookup_table_channel_descriptors_f32,
            validate_lookup_table, validate_same_size,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_8u_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Source and destination image sizes must match.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    values: &DeviceMemory<i32>,
    levels: &DeviceMemory<i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    validate_same_size(source.size(), destination.size())?;
    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_8u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_8u_C1IR_Ctx`].
/// Updates the image in place.
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep the image exclusively writable and disjoint from all lookup storage during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_u8_c1_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, u8, C1>,
    values: &DeviceMemory<i32>,
    levels: &DeviceMemory<i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_8u_C1IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_8u_C1IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_16u_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Source and destination image sizes must match.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_u16_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageOutputView<'_, u16, C1>,
    values: &DeviceMemory<i32>,
    levels: &DeviceMemory<i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    validate_same_size(source.size(), destination.size())?;
    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_16u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_16u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_16u_C1IR_Ctx`].
/// Updates the image in place.
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep the image exclusively writable and disjoint from all lookup storage during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_u16_c1_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, u16, C1>,
    values: &DeviceMemory<i32>,
    levels: &DeviceMemory<i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_16u_C1IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_16u_C1IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_16s_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Source and destination image sizes must match.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_i16_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, i16, C1>,
    destination: &mut ImageOutputView<'_, i16, C1>,
    values: &DeviceMemory<i32>,
    levels: &DeviceMemory<i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    validate_same_size(source.size(), destination.size())?;
    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_16s_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_16s_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_16s_C1IR_Ctx`].
/// Updates the image in place.
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep the image exclusively writable and disjoint from all lookup storage during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_i16_c1_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, i16, C1>,
    values: &DeviceMemory<i32>,
    levels: &DeviceMemory<i32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_16s_C1IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_16s_C1IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_32f_C1R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Source and destination image sizes must match.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    values: &DeviceMemory<f32>,
    levels: &DeviceMemory<f32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    validate_same_size(source.size(), destination.size())?;
    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts color samples through lookup values and levels without interpolation.
/// Calls [`sys::nppiLUT_32f_C1IR_Ctx`].
/// Updates the image in place.
///
/// Enqueues work on the selected stream.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Images and both device tables must belong to the selected CUDA context.
/// Value and level counts must match, contain at least two entries, and fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep the image exclusively writable and disjoint from all lookup storage during native use.
/// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_f32_c1_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, f32, C1>,
    values: &DeviceMemory<f32>,
    levels: &DeviceMemory<f32>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;
    stream_context.validate_region(values)?;
    stream_context.validate_region(levels)?;

    let level_count = validate_lookup_table(values.len(), levels.len())?;

    // SAFETY: `nppiLUT_32f_C1IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiLUT_32f_C1IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            values.as_ptr().cast(),
            levels.as_ptr().cast(),
            level_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u8_c3,
    u8,
    C3,
    3,
    nppiLUT_8u_C3R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u8_c3_in_place,
    u8,
    C3,
    3,
    nppiLUT_8u_C3IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u8_c4,
    u8,
    C4,
    4,
    nppiLUT_8u_C4R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u8_c4_in_place,
    u8,
    C4,
    4,
    nppiLUT_8u_C4IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u8_ac4,
    u8,
    AC4,
    3,
    nppiLUT_8u_AC4R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u8_ac4_in_place,
    u8,
    AC4,
    3,
    nppiLUT_8u_AC4IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u16_c3,
    u16,
    C3,
    3,
    nppiLUT_16u_C3R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiLUT_16u_C3IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u16_c4,
    u16,
    C4,
    4,
    nppiLUT_16u_C4R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u16_c4_in_place,
    u16,
    C4,
    4,
    nppiLUT_16u_C4IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u16_ac4,
    u16,
    AC4,
    3,
    nppiLUT_16u_AC4R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_u16_ac4_in_place,
    u16,
    AC4,
    3,
    nppiLUT_16u_AC4IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_i16_c3,
    i16,
    C3,
    3,
    nppiLUT_16s_C3R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiLUT_16s_C3IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_i16_c4,
    i16,
    C4,
    4,
    nppiLUT_16s_C4R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_i16_c4_in_place,
    i16,
    C4,
    4,
    nppiLUT_16s_C4IR_Ctx
);

impl_lookup_table_packed!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_i16_ac4,
    i16,
    AC4,
    3,
    nppiLUT_16s_AC4R_Ctx
);

impl_lookup_table_packed_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_i16_ac4_in_place,
    i16,
    AC4,
    3,
    nppiLUT_16s_AC4IR_Ctx
);

impl_lookup_table_packed_f32!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_f32_c3,
    C3,
    3,
    nppiLUT_32f_C3R_Ctx
);

impl_lookup_table_packed_f32_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_f32_c3_in_place,
    C3,
    3,
    nppiLUT_32f_C3IR_Ctx
);

impl_lookup_table_packed_f32!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_f32_c4,
    C4,
    4,
    nppiLUT_32f_C4R_Ctx
);

impl_lookup_table_packed_f32_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_f32_c4_in_place,
    C4,
    4,
    nppiLUT_32f_C4IR_Ctx
);

impl_lookup_table_packed_f32!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_f32_ac4,
    AC4,
    3,
    nppiLUT_32f_AC4R_Ctx
);

impl_lookup_table_packed_f32_in_place!(
    "Maps color samples through user-defined input levels and output values without interpolation.",
    lookup_table_f32_ac4_in_place,
    AC4,
    3,
    nppiLUT_32f_AC4IR_Ctx
);

impl_generic_lookup_table_c1!(LookupTableC1, lookup_table, lookup_table_c1, [
    u8, i32 => lookup_table_u8_c1,
    u16, i32 => lookup_table_u16_c1,
    i16, i32 => lookup_table_i16_c1,
    f32, f32 => lookup_table_f32_c1,
]);

impl_generic_lookup_table_c1_in_place!(
    LookupTableC1InPlace,
    lookup_table_in_place,
    lookup_table_c1_in_place,
    [
        u8, i32 => lookup_table_u8_c1_in_place,
        u16, i32 => lookup_table_u16_c1_in_place,
        i16, i32 => lookup_table_i16_c1_in_place,
        f32, f32 => lookup_table_f32_c1_in_place,
    ]
);

impl_generic_lookup_table_packed!(LookupTableC3, lookup_table, lookup_table_c3, C3, 3, [
    u8, i32 => lookup_table_u8_c3,
    u16, i32 => lookup_table_u16_c3,
    i16, i32 => lookup_table_i16_c3,
    f32, f32 => lookup_table_f32_c3,
]);

impl_generic_lookup_table_packed_in_place!(
    LookupTableC3InPlace,
    lookup_table_in_place,
    lookup_table_c3_in_place,
    C3,
    3,
    [
        u8, i32 => lookup_table_u8_c3_in_place,
        u16, i32 => lookup_table_u16_c3_in_place,
        i16, i32 => lookup_table_i16_c3_in_place,
        f32, f32 => lookup_table_f32_c3_in_place,
    ]
);

impl_generic_lookup_table_packed!(LookupTableC4, lookup_table, lookup_table_c4, C4, 4, [
    u8, i32 => lookup_table_u8_c4,
    u16, i32 => lookup_table_u16_c4,
    i16, i32 => lookup_table_i16_c4,
    f32, f32 => lookup_table_f32_c4,
]);

impl_generic_lookup_table_packed_in_place!(
    LookupTableC4InPlace,
    lookup_table_in_place,
    lookup_table_c4_in_place,
    C4,
    4,
    [
        u8, i32 => lookup_table_u8_c4_in_place,
        u16, i32 => lookup_table_u16_c4_in_place,
        i16, i32 => lookup_table_i16_c4_in_place,
        f32, f32 => lookup_table_f32_c4_in_place,
    ]
);

impl_generic_lookup_table_packed!(LookupTableAc4, lookup_table, lookup_table_ac4, AC4, 3, [
    u8, i32 => lookup_table_u8_ac4,
    u16, i32 => lookup_table_u16_ac4,
    i16, i32 => lookup_table_i16_ac4,
    f32, f32 => lookup_table_f32_ac4,
]);

impl_generic_lookup_table_packed_in_place!(
    LookupTableAc4InPlace,
    lookup_table_in_place,
    lookup_table_ac4_in_place,
    AC4,
    3,
    [
        u8, i32 => lookup_table_u8_ac4_in_place,
        u16, i32 => lookup_table_u16_ac4_in_place,
        i16, i32 => lookup_table_i16_ac4_in_place,
        f32, f32 => lookup_table_f32_ac4_in_place,
    ]
);
