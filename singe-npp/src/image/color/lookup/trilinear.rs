use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{lookup_table_trilinear_descriptors, validate_same_size},
        view::{AC4, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Converts colors through a three-dimensional lookup table using trilinear interpolation.
/// Calls [`sys::nppiLUT_Trilinear_8u_C4R_Ctx`].
///
/// Rejects stream capture and returns after successful stream completion.
/// Copies source alpha to the destination unchanged.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Source and destination image sizes must match.
/// Images and the device value table must belong to the selected CUDA context.
/// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
/// Their checked length product must not exceed `values.len()`.
/// Capture checks and stream synchronization can also fail for implementations that wait for completion.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
/// Keep image and table allocations valid until this call completes its native work.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_trilinear_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageOutputView<'_, u8, C4>,
    values: &DeviceMemory<u32>,
    levels: [&[u8]; 3],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(values)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_same_size(source.size(), destination.size())?;
    let (mut level_pointers, mut level_counts) =
        lookup_table_trilinear_descriptors(values.len(), levels)?;

    // SAFETY: `nppiLUT_Trilinear_8u_C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiLUT_Trilinear_8u_C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            values.as_ptr().cast_mut(),
            level_pointers.as_mut_ptr().cast(),
            level_counts.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Converts colors through a three-dimensional lookup table using trilinear interpolation.
/// Calls [`sys::nppiLUT_Trilinear_8u_AC4R_Ctx`].
///
/// Rejects stream capture and returns after successful stream completion.
/// The destination alpha channel is not processed.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Source and destination image sizes must match.
/// Images and the device value table must belong to the selected CUDA context.
/// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
/// Their checked length product must not exceed `values.len()`.
/// Capture checks and stream synchronization can also fail for implementations that wait for completion.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
/// Keep image and table allocations valid until this call completes its native work.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_trilinear_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageOutputView<'_, u8, AC4>,
    values: &DeviceMemory<u32>,
    levels: [&[u8]; 3],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(values)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_same_size(source.size(), destination.size())?;
    let (mut level_pointers, mut level_counts) =
        lookup_table_trilinear_descriptors(values.len(), levels)?;

    // SAFETY: `nppiLUT_Trilinear_8u_AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiLUT_Trilinear_8u_AC4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            values.as_ptr().cast_mut(),
            level_pointers.as_mut_ptr().cast(),
            level_counts.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Converts colors through a three-dimensional lookup table using trilinear interpolation.
/// Calls [`sys::nppiLUT_Trilinear_8u_AC4IR_Ctx`].
/// Updates the image in place.
///
/// Rejects stream capture and returns after successful stream completion.
/// The destination alpha channel is not processed.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
/// Images and the device value table must belong to the selected CUDA context.
/// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
/// Their checked length product must not exceed `values.len()`.
/// Capture checks and stream synchronization can also fail for implementations that wait for completion.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Image inputs and lookup tables must be initialized and ready when read by native work.
/// Keep the image exclusively writable and disjoint from all lookup storage during native use.
/// Keep image and table allocations valid until this call completes its native work.
/// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn lookup_table_trilinear_u8_ac4_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, u8, AC4>,
    values: &DeviceMemory<u32>,
    levels: [&[u8]; 3],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(image)?;
    stream_context.validate_region(values)?;
    stream_context.stream().ensure_not_capturing()?;

    let (mut level_pointers, mut level_counts) =
        lookup_table_trilinear_descriptors(values.len(), levels)?;

    // SAFETY: `nppiLUT_Trilinear_8u_AC4IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiLUT_Trilinear_8u_AC4IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            values.as_ptr().cast_mut(),
            level_pointers.as_mut_ptr().cast(),
            level_counts.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

impl_generic_lookup_table_trilinear!(
    LookupTableTrilinearC4,
    lookup_table_trilinear,
    lookup_table_trilinear_c4,
    C4,
    [u8 => lookup_table_trilinear_u8_c4]
);

impl_generic_lookup_table_trilinear!(
    LookupTableTrilinearAC4,
    lookup_table_trilinear,
    lookup_table_trilinear_ac4,
    AC4,
    [u8 => lookup_table_trilinear_u8_ac4]
);

impl_generic_lookup_table_trilinear_in_place!(
    LookupTableTrilinearAC4InPlace,
    lookup_table_trilinear_in_place,
    lookup_table_trilinear_ac4_in_place,
    AC4,
    [u8 => lookup_table_trilinear_u8_ac4_in_place]
);
