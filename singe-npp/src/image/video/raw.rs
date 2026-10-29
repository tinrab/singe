//! Video conversion enqueue operations on the selected native CUDA stream.
//!
//! # Safety
//! Retain source and destination allocation owners until completion and through every captured replay.
//! Establish stream ordering for prior producers and conflicting accesses.
//! Sources must be initialized throughout their stored-sample footprint, and destinations must be exclusively writable.
//! Calls check frame schemas and actual CUDA contexts, but submission does not establish completed initialization.
//! Packed output padding bits have the native format's specified or unspecified values; pitched row padding is not produced.
use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::video::{VideoFormat, VideoOutputView, VideoView, validate_conversion},
    optional::{
        nppiI444_8u_ToUYVP_10u_P3C3R_Ctx, nppiNV12_8u_ToUYVP_10u_P2C3R_Ctx,
        nppiNV16_8u_ToUYVP_10u_P2C3R_Ctx, nppiP010_16u_ToUYVP_10u_P3C3R_Ctx,
        nppiP208_8u_ToUYVP_10u_C3C3R_Ctx, nppiP210_16u_ToUYVP_10u_P2C3R_Ctx,
        nppiUYVP_10u_ToI444_8u_C3P3R_Ctx, nppiUYVP_10u_ToNV16_8u_C3P2R_Ctx,
        nppiUYVP_10u_ToP208_8u_C3C3R_Ctx, nppiUYVP_10u_ToV210_10u_C3C3R_Ctx,
        nppiUYVP_10u_ToY410_10u_C3C3R_Ctx, nppiUYVP_10u_ToYUV444_8u_C3C3R_Ctx,
        nppiV210_10u_ToUYVP_10u_C3C3R_Ctx, nppiY410_10u_ToUYVP_10u_C3C3R_Ctx,
        nppiYUV444_8u_ToUYVP_10u_C3C3R_Ctx,
    },
    try_ffi,
    types::{ColorSpace, Point, Size},
};

/// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB BT.601 full-range color conversion (ITU-T T.871 / JFIF).
///
/// Coefficients derived from BT.601 luma weights (Kr=0.299, Kb=0.114) with full digital range Y=\[0..255\], Cb/Cr=\[0..255\] (offset 128). Use this variant for cameras that signal full-range BT.601 NV12 (e.g. OAK-D VPU encoder output), as distinct from the unspecified-range default `nppiNV12ToRGB_8u_P2C3R` and the BT.709 variants.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_to_rgb_601_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Rgb,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiNV12ToRGB_601_8u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiNV12ToRGB_601_8u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR BT.601 full-range color conversion (ITU-T T.871 / JFIF).
///
/// Coefficients derived from BT.601 luma weights (Kr=0.299, Kb=0.114) with full digital range Y=\[0..255\], Cb/Cr=\[0..255\] (offset 128).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Bgr`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_to_bgr_601_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Bgr,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiNV12ToBGR_601_8u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiNV12ToBGR_601_8u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// RGB to UYVP color conversion.
///
/// NPP 13.2's exercised half codec does not produce the advertised color values.
/// Use the completed conversion under `image::video` for normalized half RGB.
/// Native success alone does not establish initialized output for this entry point.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_16f_to_uyvp_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgb,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiRGB_16f_ToUYVP_10u_C3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGB_16f_ToUYVP_10u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGB to UYVP color conversion.
///
/// NPP 13.2's exercised half codec does not produce the advertised color values.
/// Use the completed conversion under `image::video` for normalized half RGB.
/// Native success alone does not establish initialized output for this entry point.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_16f_to_uyvp_10u_p3c3r(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::RgbPlanar,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16f> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiRGB_16f_ToUYVP_10u_P3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGB_16f_ToUYVP_10u_P3C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGBA to UYVP color conversion.
///
/// NPP 13.2's exercised half codec does not produce the advertised color values.
/// Use the completed conversion under `image::video` for normalized half RGB.
/// Native success alone does not establish initialized output for this entry point.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgba` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgba_16f_to_uyvp_10u_ac4c3r(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgba,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiRGBA_16f_ToUYVP_10u_AC4C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBA_16f_ToUYVP_10u_AC4C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// UYVP to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_rgb_8u_c3ac4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::Rgba,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVP_10u_ToRGB_8u_C3AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVP_10u_ToRGB_8u_C3AC4R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// UYVP to RGB color conversion.
///
/// NPP 13.2's exercised half codec does not produce the advertised color values.
/// Use the completed conversion under `image::video` for normalized half RGB.
/// Native success alone does not establish initialized output for this entry point.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_rgb_16f_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::Rgb,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVP_10u_ToRGB_16f_C3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVP_10u_ToRGB_16f_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// UYVP to RGBA color conversion.
///
/// NPP 13.2's exercised half codec does not produce the advertised color values.
/// Use the completed conversion under `image::video` for normalized half RGB.
/// Native success alone does not establish initialized output for this entry point.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_rgba_16f_c3ac4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: f32,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::Rgba,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVP_10u_ToRGBA_16f_C3AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVP_10u_ToRGBA_16f_C3AC4R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// UYVP to RGB color conversion.
///
/// NPP 13.2's exercised half codec does not produce the advertised color values.
/// Use the completed conversion under `image::video` for normalized half RGB.
/// Native success alone does not establish initialized output for this entry point.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::RgbPlanar`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_rgb_16f_c3p3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::RgbPlanar,
        source_offset,
        true,
        true,
    )?;
    let mut destination_pointers: Vec<*mut sys::Npp16f> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    // SAFETY: `nppiUYVP_10u_ToRGB_16f_C3P3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVP_10u_ToRGB_16f_C3P3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// 4:2:0 conversions.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_8u_to_uyvp_10u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        false,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    let source_steps: Vec<_> = source
        .planes
        .iter()
        .map(super::super::view::ImageView::step)
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiNV12_8u_ToUYVP_10u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source_steps.as_ptr(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// P010 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010Planar` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p010_16u_to_uyvp_10u_p3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P010Planar,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiP010_16u_ToUYVP_10u_P3C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// UYVP 10-bit packed YCbCr 4:2:2 transcoding to/from P208.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, `size` has nonpositive dimensions, or the native conversion fails.
/// Raw pointers, allocation capacities, contexts, and byte strides are not validated by this wrapper.
///
/// # Safety
///
/// `source` and `destination` must point to device allocations containing the
/// source and destination formats required by NPP, with valid byte pitches and
/// enough storage for `size`. The allocations must remain valid until the stream
/// completes. They must not overlap. NPP's header does not specify the P208 plane
/// placement, so this interface leaves that storage contract to the caller.
pub unsafe fn uyvp_10u_to_p208_8u_c3c3r(
    context: &StreamContext,
    source: *const u8,
    source_step: i32,
    destination: *mut u8,
    destination_step: i32,
    size: Size,
) -> Result<()> {
    context.cuda_context().bind()?;
    size.validate()?;

    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiUYVP_10u_ToP208_8u_C3C3R_Ctx(
            source,
            source_step,
            Point { x: 0, y: 0 }.into(),
            destination,
            destination_step,
            size.into(),
            context.as_raw()
        ))
    }
}

/// UYVP to NV16 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Nv16`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_nv16_8u_c3p2r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::Nv16,
        source_offset,
        true,
        false,
    )?;
    let mut destination_pointers: Vec<*mut sys::Npp8u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    let destination_steps: Vec<_> = destination
        .planes
        .iter()
        .map(super::super::view::ImageOutputView::step)
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiUYVP_10u_ToNV16_8u_C3P2R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination_pointers.as_mut_ptr(),
            destination_steps.as_ptr(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// UYVP to V210 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::V210`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_v210_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::V210,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiUYVP_10u_ToV210_10u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// UYVP 10-bit packed YCbCr 4:2:2 transcoding to/from P208.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, `size` has nonpositive dimensions, or the native conversion fails.
/// Raw pointers, allocation capacities, contexts, and byte strides are not validated by this wrapper.
///
/// # Safety
///
/// `source` and `destination` must point to device allocations containing the
/// source and destination formats required by NPP, with valid byte pitches and
/// enough storage for `size`. The allocations must remain valid until the stream
/// completes. They must not overlap. NPP's header does not specify the P208 plane
/// placement, so this interface leaves that storage contract to the caller.
pub unsafe fn p208_8u_to_uyvp_10u_c3c3r(
    context: &StreamContext,
    source: *const u8,
    source_step: i32,
    destination: *mut u8,
    destination_step: i32,
    size: Size,
) -> Result<()> {
    context.cuda_context().bind()?;
    size.validate()?;

    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiP208_8u_ToUYVP_10u_C3C3R_Ctx(
            source,
            source_step,
            Point { x: 0, y: 0 }.into(),
            destination,
            destination_step,
            size.into(),
            context.as_raw()
        ))
    }
}

/// NV16 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv16` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv16_8u_to_uyvp_10u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv16,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        false,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    let source_steps: Vec<_> = source
        .planes
        .iter()
        .map(super::super::view::ImageView::step)
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiNV16_8u_ToUYVP_10u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source_steps.as_ptr(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// V210 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::V210` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn v210_10u_to_uyvp_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::V210,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiV210_10u_ToUYVP_10u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// P210 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P210` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p210_16u_to_uyvp_10u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P210,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        false,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    let source_steps: Vec<_> = source
        .planes
        .iter()
        .map(super::super::view::ImageView::step)
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiP210_16u_ToUYVP_10u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source_steps.as_ptr(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// 4:4:4 conversions.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::I444`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_i444_8u_c3p3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::I444,
        source_offset,
        true,
        true,
    )?;
    let mut destination_pointers: Vec<*mut sys::Npp8u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiUYVP_10u_ToI444_8u_C3P3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// UYVP to YUV444 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Yuv444`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_yuv444_8u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::Yuv444,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiUYVP_10u_ToYUV444_8u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// UYVP to Y410 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Y410`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvp_10u_to_y410_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        VideoFormat::Y410,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiUYVP_10u_ToY410_10u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// I444 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::I444` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn i444_8u_to_uyvp_10u_p3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::I444,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiI444_8u_ToUYVP_10u_P3C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// YUV444 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Yuv444` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn yuv444_8u_to_uyvp_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Yuv444,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiYUV444_8u_ToUYVP_10u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// Y410 to UYVP color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Y410` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn y410_10u_to_uyvp_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Y410,
        destination,
        VideoFormat::Uyvp10,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(nppiY410_10u_ToUYVP_10u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// RGB to UYVY color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Uyvy`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_8u_to_uyvy_8u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgb,
        destination,
        VideoFormat::Uyvy,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiRGB_8u_ToUYVY_8u_C3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGB_8u_ToUYVY_8u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGB to UYVY color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Uyvy`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_8u_to_uyvy_8u_p3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::RgbPlanar,
        destination,
        VideoFormat::Uyvy,
        source_offset,
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiRGB_8u_ToUYVY_8u_P3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGB_8u_ToUYVY_8u_P3C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGBA to UYVY color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgba` and the destination format must be `VideoFormat::Uyvy`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgba_8u_to_uyvy_8u_ac4c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgba,
        destination,
        VideoFormat::Uyvy,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiRGBA_8u_ToUYVY_8u_AC4C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBA_8u_ToUYVY_8u_AC4C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGB to UYVY16 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Uyvy16`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_16u_to_uyvy16_16u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgb,
        destination,
        VideoFormat::Uyvy16,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiRGB_16u_ToUYVY16_16u_C3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGB_16u_ToUYVY16_16u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGB to UYVY16 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Uyvy16`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_16u_to_uyvy16_16u_p3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::RgbPlanar,
        destination,
        VideoFormat::Uyvy16,
        source_offset,
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiRGB_16u_ToUYVY16_16u_P3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGB_16u_ToUYVY16_16u_P3C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGBA to UYVY16 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgba` and the destination format must be `VideoFormat::Uyvy16`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgba_16u_to_uyvy16_16u_ac4c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgba,
        destination,
        VideoFormat::Uyvy16,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiRGBA_16u_ToUYVY16_16u_AC4C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBA_16u_ToUYVY16_16u_AC4C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// UYVY to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvy` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvy_8u_to_rgb_8u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvy,
        destination,
        VideoFormat::Rgb,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVY_8u_ToRGB_8u_C3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVY_8u_ToRGB_8u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// UYVY to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvy` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvy_8u_to_rgb_8u_c3ac4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvy,
        destination,
        VideoFormat::Rgba,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVY_8u_ToRGB_8u_C3AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVY_8u_ToRGB_8u_C3AC4R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// UYVY to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvy` and the destination format must be `VideoFormat::RgbPlanar`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvy_8u_to_rgb_8u_c3p3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvy,
        destination,
        VideoFormat::RgbPlanar,
        source_offset,
        true,
        true,
    )?;
    let mut destination_pointers: Vec<*mut sys::Npp8u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    // SAFETY: `nppiUYVY_8u_ToRGB_8u_C3P3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVY_8u_ToRGB_8u_C3P3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// UYVY16 to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvy16` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvy16_16u_to_rgb_16u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvy16,
        destination,
        VideoFormat::Rgb,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVY16_16u_ToRGB_16u_C3C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVY16_16u_ToRGB_16u_C3C3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// UYVY16 to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvy16` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvy16_16u_to_rgb_16u_c3ac4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: u16,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvy16,
        destination,
        VideoFormat::Rgba,
        source_offset,
        true,
        true,
    )?;
    // SAFETY: `nppiUYVY16_16u_ToRGB_16u_C3AC4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVY16_16u_ToRGB_16u_C3AC4R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// UYVY16 to RGB color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvy16` and the destination format must be `VideoFormat::RgbPlanar`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn uyvy16_16u_to_rgb_16u_c3p3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Uyvy16,
        destination,
        VideoFormat::RgbPlanar,
        source_offset,
        true,
        true,
    )?;
    let mut destination_pointers: Vec<*mut sys::Npp16u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    // SAFETY: `nppiUYVY16_16u_ToRGB_16u_C3P3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiUYVY16_16u_ToRGB_16u_C3P3R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            source_offset.into(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// NV12 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_to_rgb_8u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Rgba,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiNV12ToRGB_8u_P2C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiNV12ToRGB_8u_P2C4R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// NV12 to BGR color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Bgra`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_to_bgr_8u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Bgra,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiNV12ToBGR_8u_P2C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiNV12ToBGR_8u_P2C4R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// NV12 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_to_rgb_bt2020_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Rgb,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiNV12ToRGB_BT2020_8u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiNV12ToRGB_BT2020_8u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// NV12 to BGR color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Bgr`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn nv12_to_bgr_bt2020_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        VideoFormat::Bgr,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}

/// P010 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p010_to_rgb_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P010,
        destination,
        VideoFormat::Rgb,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiP010ToRGB_8u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiP010ToRGB_8u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// P010 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p010_to_rgb_8u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P010,
        destination,
        VideoFormat::Rgba,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiP010ToRGB_8u_P2C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiP010ToRGB_8u_P2C4R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// P010 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p010_to_rgb_16u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P010,
        destination,
        VideoFormat::Rgb,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiP010ToRGB_16u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiP010ToRGB_16u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// P010 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p010_to_rgb_16u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
    alpha: u16,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P010,
        destination,
        VideoFormat::Rgba,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiP010ToRGB_16u_P2C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiP010ToRGB_16u_P2C4R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// P016 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P016` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p016_to_rgb_16u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P016,
        destination,
        VideoFormat::Rgb,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiP016ToRGB_16u_P2C3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiP016ToRGB_16u_P2C3R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// P016 to RGB color conversion.
///
/// NPP 13.2's exercised implementation mixes luma/chroma values and reads beyond nominal rows.
/// The completed conversion under `image::video` uses color-twist kernels with checked source footprints.
/// Native callers must provide initialized capacity for the actual native reads, including any row padding.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P016` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn p016_to_rgb_16u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
    alpha: u16,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::P016,
        destination,
        VideoFormat::Rgba,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp16u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    // SAFETY: `nppiP016ToRGB_16u_P2C4R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiP016ToRGB_16u_P2C4R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination.planes[0].as_mut_ptr().cast(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            alpha,
            context.as_raw()
        ))
    }
}

/// RGB to NV12 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Nv12`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_to_nv12_8u_c3p2r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Rgb,
        destination,
        VideoFormat::Nv12,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let mut destination_pointers: Vec<*mut sys::Npp8u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    // SAFETY: `nppiRGBToNV12_8u_C3P2R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBToNV12_8u_C3P2R_Ctx(
            source.planes[0].as_ptr().cast(),
            source.planes[0].step(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// RGB to NV12 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Nv12`. The destination pixel extent must fit within the source extent.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn rgb_to_nv12_8u_p3p2r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::RgbPlanar,
        destination,
        VideoFormat::Nv12,
        Point { x: 0, y: 0 },
        true,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    let mut destination_pointers: Vec<*mut sys::Npp8u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    // SAFETY: `nppiRGBToNV12_8u_P3P2R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBToNV12_8u_P3P2R_Ctx(
            source_pointers.as_ptr(),
            source.planes[0].step(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            color_space.into(),
            context.as_raw()
        ))
    }
}

/// YUV420 to NV12 color conversion.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Yuv420` and the destination format must be `VideoFormat::Nv12`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
///
/// # Safety
///
/// Uphold the allocation lifetime, ordering, and initialized-source requirements of this module.
pub unsafe fn yuv420_to_nv12_8u_p3p2r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    context.cuda_context().bind()?;
    for plane in source.planes() {
        context.validate_region(plane)?;
    }
    for plane in destination.planes() {
        context.validate_region(plane)?;
    }

    validate_conversion(
        source,
        VideoFormat::Yuv420,
        destination,
        VideoFormat::Nv12,
        Point { x: 0, y: 0 },
        false,
        true,
    )?;
    let source_pointers: Vec<*const sys::Npp8u> = source
        .planes
        .iter()
        .map(|plane| plane.as_ptr().cast())
        .collect();
    let mut destination_pointers: Vec<*mut sys::Npp8u> = destination
        .planes
        .iter_mut()
        .map(|plane| plane.as_mut_ptr().cast())
        .collect();
    let source_steps: Vec<_> = source
        .planes
        .iter()
        .map(super::super::view::ImageView::step)
        .collect();
    // SAFETY: `nppiYUV420ToNV12_8u_P3P2R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiYUV420ToNV12_8u_P3P2R_Ctx(
            source_pointers.as_ptr(),
            source_steps.as_ptr(),
            destination_pointers.as_mut_ptr(),
            destination.planes[0].step(),
            destination.size.into(),
            context.as_raw()
        ))
    }
}
