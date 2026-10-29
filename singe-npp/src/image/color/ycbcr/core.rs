use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{
            color_batch_advanced_descriptors, color_batch_descriptors,
            color_planar_batch_advanced_descriptors, color_planar_batch_descriptors,
            subsampled_color_batch_advanced_descriptors, subsampled_color_batch_descriptors,
            validate_equal_step, validate_same_size,
        },
        view::{
            AC4, C1, C2, C3, C4, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView,
        },
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_convert_same_layout!(
    "Converts RGB color values to YCbCr color values.",
    rgb_to_ycbcr_u8_c3,
    u8,
    C3,
    nppiRGBToYCbCr_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to YCbCr color values. The destination alpha channel is not processed.",
    rgb_to_ycbcr_u8_ac4,
    u8,
    AC4,
    nppiRGBToYCbCr_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to YCbCr color values.",
    RgbToYcbcrC3, rgb_to_ycbcr, rgb_to_ycbcr_c3, C3, [
    u8 => rgb_to_ycbcr_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to YCbCr color values. The destination alpha channel is not processed.",
    RgbToYcbcrAC4, rgb_to_ycbcr, rgb_to_ycbcr_ac4, AC4, [
    u8 => rgb_to_ycbcr_u8_ac4,
    ]
);

impl_color_convert_different_layout!(
    "Converts RGB color values to `YCbCr422` color values.",
    rgb_to_ycbcr422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToYCbCr422_8u_C3C2R_Ctx
);

impl_color_convert_different_layout!(
    "Converts RGB color values to `YCrCb422` color values.",
    rgb_to_ycrcb422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToYCrCb422_8u_C3C2R_Ctx
);

/// Converts the separate red, green, and blue source planes to packed `YCbCr422` pixels.
/// Calls [`sys::nppiRGBToYCbCr422_8u_P3C2R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.
/// All source and destination views must have equal image sizes, and the three source planes must have equal byte strides.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn rgb_to_ycbcr422_u8_p3_to_c2(
    stream_context: &StreamContext,
    source_r: &ImageView<'_, u8, C1>,
    source_g: &ImageView<'_, u8, C1>,
    source_b: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C2>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source_r)?;
    stream_context.validate_region(source_g)?;
    stream_context.validate_region(source_b)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source_r.size(), destination.size())?;
    validate_same_size(source_r.size(), source_g.size())?;
    validate_same_size(source_r.size(), source_b.size())?;
    validate_equal_step(source_r.step(), source_g.step(), "planar rgb source step")?;
    validate_equal_step(source_r.step(), source_b.step(), "planar rgb source step")?;

    let source_planes = [
        source_r.as_ptr().cast(),
        source_g.as_ptr().cast(),
        source_b.as_ptr(),
    ];

    // SAFETY: `nppiRGBToYCbCr422_8u_P3C2R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBToYCbCr422_8u_P3C2R_Ctx(
            source_planes.as_ptr(),
            source_r.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Converts the separate red, green, and blue source planes to packed `YCrCb422` pixels.
/// Calls [`sys::nppiRGBToYCrCb422_8u_P3C2R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.
/// All source and destination views must have equal image sizes, and the three source planes must have equal byte strides.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn rgb_to_ycrcb422_u8_p3_to_c2(
    stream_context: &StreamContext,
    source_r: &ImageView<'_, u8, C1>,
    source_g: &ImageView<'_, u8, C1>,
    source_b: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C2>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source_r)?;
    stream_context.validate_region(source_g)?;
    stream_context.validate_region(source_b)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source_r.size(), destination.size())?;
    validate_same_size(source_r.size(), source_g.size())?;
    validate_same_size(source_r.size(), source_b.size())?;
    validate_equal_step(source_r.step(), source_g.step(), "planar rgb source step")?;
    validate_equal_step(source_r.step(), source_b.step(), "planar rgb source step")?;

    let source_planes = [
        source_r.as_ptr().cast(),
        source_g.as_ptr().cast(),
        source_b.as_ptr(),
    ];

    // SAFETY: `nppiRGBToYCrCb422_8u_P3C2R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRGBToYCrCb422_8u_P3C2R_Ctx(
            source_planes.as_ptr(),
            source_r.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_color_convert_different_layout!(
    "Converts RGB color values to `CbYCr422` color values.",
    rgb_to_cbycr422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToCbYCr422_8u_C3C2R_Ctx
);

impl_color_convert_different_layout!(
    "Applies forward gamma correction to RGB values and converts them to `CbYCr422`.",
    rgb_to_cbycr422_gamma_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts RGB color values to YCbCr color values.",
    rgb_to_ycbcr_u8_p3,
    u8,
    3,
    nppiRGBToYCbCr_8u_P3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_u8_c3,
    u8,
    C3,
    nppiYCbCrToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YCbCr color values to RGB color values. The destination alpha channel is not processed.",
    ycbcr_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiYCbCrToRGB_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts YCbCr color values to RGB color values.",
    YcbcrToRgbC3, ycbcr_to_rgb, ycbcr_to_rgb_c3, C3, [
    u8 => ycbcr_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts YCbCr color values to RGB color values. The destination alpha channel is not processed.",
    YcbcrToRgbAC4,
    ycbcr_to_rgb,
    ycbcr_to_rgb_ac4,
    AC4,
    [u8 => ycbcr_to_rgb_u8_ac4]
);

impl_color_convert_different_layout!(
    "Converts `YCbCr422` color values to RGB color values.",
    ycbcr422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYCbCr422ToRGB_8u_C2C3R_Ctx
);

/// Converts packed `YCbCr422` pixels to separate red, green, and blue destination planes.
/// Calls [`sys::nppiYCbCr422ToRGB_8u_C2P3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.
/// All source and destination views must have equal image sizes, and the three destination planes must have equal byte strides.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn ycbcr422_to_rgb_u8_c2_to_p3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C2>,
    destination_r: &mut ImageOutputView<'_, u8, C1>,
    destination_g: &mut ImageOutputView<'_, u8, C1>,
    destination_b: &mut ImageOutputView<'_, u8, C1>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination_r)?;
    stream_context.validate_region(destination_g)?;
    stream_context.validate_region(destination_b)?;

    validate_same_size(source.size(), destination_r.size())?;
    validate_same_size(source.size(), destination_g.size())?;
    validate_same_size(source.size(), destination_b.size())?;
    validate_equal_step(
        destination_r.step(),
        destination_g.step(),
        "planar rgb destination step",
    )?;
    validate_equal_step(
        destination_r.step(),
        destination_b.step(),
        "planar rgb destination step",
    )?;

    let mut destination_planes = [
        destination_r.as_mut_ptr().cast(),
        destination_g.as_mut_ptr().cast(),
        destination_b.as_mut_ptr().cast(),
    ];

    // SAFETY: `nppiYCbCr422ToRGB_8u_C2P3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiYCbCr422ToRGB_8u_C2P3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination_planes.as_mut_ptr(),
            destination_r.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_color_convert_same_layout!(
    "Converts `YCbCr422` color values to `YCrCb422` color values.",
    ycbcr422_to_ycrcb422_u8_c2,
    u8,
    C2,
    nppiYCbCr422ToYCrCb422_8u_C2R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts `YCbCr422` color values to `YCrCb422` color values.",
    Ycbcr422ToYcrcb422C2,
    ycbcr422_to_ycrcb422,
    ycbcr422_to_ycrcb422_c2,
    C2,
    [u8 => ycbcr422_to_ycrcb422_u8_c2]
);

impl_color_convert_same_layout!(
    "Converts `YCbCr422` color values to `CbYCr422` color values.",
    ycbcr422_to_cbycr422_u8_c2,
    u8,
    C2,
    nppiYCbCr422ToCbYCr422_8u_C2R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts `YCbCr422` color values to `CbYCr422` color values.",
    Ycbcr422ToCbycr422C2,
    ycbcr422_to_cbycr422,
    ycbcr422_to_cbycr422_c2,
    C2,
    [u8 => ycbcr422_to_cbycr422_u8_c2]
);

impl_color_convert_different_layout!(
    "Converts `YCrCb422` color values to RGB color values.",
    ycrcb422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYCrCb422ToRGB_8u_C2C3R_Ctx
);

/// Converts packed `YCrCb422` pixels to separate red, green, and blue destination planes.
/// Calls [`sys::nppiYCrCb422ToRGB_8u_C2P3R_Ctx`].
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.
/// All source and destination views must have equal image sizes, and the three destination planes must have equal byte strides.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn ycrcb422_to_rgb_u8_c2_to_p3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C2>,
    destination_r: &mut ImageOutputView<'_, u8, C1>,
    destination_g: &mut ImageOutputView<'_, u8, C1>,
    destination_b: &mut ImageOutputView<'_, u8, C1>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination_r)?;
    stream_context.validate_region(destination_g)?;
    stream_context.validate_region(destination_b)?;

    validate_same_size(source.size(), destination_r.size())?;
    validate_same_size(source.size(), destination_g.size())?;
    validate_same_size(source.size(), destination_b.size())?;
    validate_equal_step(
        destination_r.step(),
        destination_g.step(),
        "planar rgb destination step",
    )?;
    validate_equal_step(
        destination_r.step(),
        destination_b.step(),
        "planar rgb destination step",
    )?;

    let mut destination_planes = [
        destination_r.as_mut_ptr().cast(),
        destination_g.as_mut_ptr().cast(),
        destination_b.as_mut_ptr().cast(),
    ];

    // SAFETY: `nppiYCrCb422ToRGB_8u_C2P3R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiYCrCb422ToRGB_8u_C2P3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination_planes.as_mut_ptr(),
            destination_r.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_color_convert_different_layout!(
    "Converts `CbYCr422` color values to RGB color values.",
    cbycr422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiCbYCr422ToRGB_8u_C2C3R_Ctx
);

impl_color_convert_different_layout!(
    "Converts BGR color values to `CbYCr422` color values. Source alpha is ignored.",
    bgr_to_cbycr422_u8_ac4_to_c2,
    u8,
    AC4,
    C2,
    nppiBGRToCbYCr422_8u_AC4C2R_Ctx
);

impl_color_convert_different_layout!(
    "Converts BGR color values to `CbYCr422` color values.",
    bgr_to_cbycr422_709hdtv_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiBGRToCbYCr422_709HDTV_8u_C3C2R_Ctx
);

impl_color_convert_different_layout!(
    "Converts BGR color values to `CbYCr422` color values. Source alpha is ignored.",
    bgr_to_cbycr422_709hdtv_u8_ac4_to_c2,
    u8,
    AC4,
    C2,
    nppiBGRToCbYCr422_709HDTV_8u_AC4C2R_Ctx
);

impl_color_convert_same_layout!(
    "Converts `CbYCr422` color values to `YCbCr422` color values.",
    cbycr422_to_ycbcr422_u8_c2,
    u8,
    C2,
    nppiCbYCr422ToYCbCr422_8u_C2R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts `CbYCr422` color values to `YCbCr422` color values.",
    Cbycr422ToYcbcr422C2,
    cbycr422_to_ycbcr422,
    cbycr422_to_ycbcr422_c2,
    C2,
    [u8 => cbycr422_to_ycbcr422_u8_c2]
);

impl_color_convert_different_layout_constant_alpha!(
    "Converts `CbYCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.",
    cbycr422_to_bgr_u8_c2_to_c4,
    u8,
    C2,
    nppiCbYCr422ToBGR_8u_C2C4R_Ctx
);

impl_color_convert_different_layout!(
    "Converts `CbYCr422` color values to BGR color values.",
    cbycr422_to_bgr_709hdtv_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx
);

impl_color_convert_different_layout_constant_alpha!(
    "Converts `CbYCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.",
    cbycr422_to_bgr_709hdtv_u8_c2_to_c4,
    u8,
    C2,
    nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx
);

impl_color_convert_batch_same_layout!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_batch_u8_c3,
    u8,
    C3,
    nppiYCbCrToRGBBatch_8u_C3R_Ctx
);

impl_color_convert_batch_same_layout_advanced!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYCbCrToRGBBatch_8u_C3R_Advanced_Ctx
);

impl_color_convert_batch_planar_to_packed!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToRGBBatch_8u_P3C3R_Ctx
);

impl_color_convert_batch_planar_to_packed_advanced!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts `YCbCr422` color values to RGB color values.",
    ycbcr422_to_rgb_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts `YCbCr422` color values to RGB color values.",
    ycbcr422_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts `YCbCr420` color values to RGB color values.",
    ycbcr420_to_rgb_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx,
    2,
    2
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts `YCbCr420` color values to RGB color values.",
    ycbcr420_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);

impl_color_convert_planar_to_planar!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_u8_p3,
    u8,
    3,
    nppiYCbCrToRGB_8u_P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts RGB color values to YCbCr color values.",
    rgb_to_ycbcr_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiRGBToYCbCr_8u_C3P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts RGB color values to YCbCr color values.",
    rgb_to_ycbcr_u8_ac4_to_p3,
    u8,
    AC4,
    3,
    nppiRGBToYCbCr_8u_AC4P3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts YCbCr color values to RGB color values.",
    ycbcr_to_rgb_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToRGB_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_packed_constant_alpha!(
    "Converts YCbCr color values to RGB color values. Writes `alpha` to every destination alpha component.",
    ycbcr_to_rgb_u8_p3_to_c4,
    u8,
    3,
    nppiYCbCrToRGB_8u_P3C4R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to YCbCr color values.",
    bgr_to_ycbcr_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiBGRToYCbCr_8u_C3P3R_Ctx
);

impl_color_convert_different_layout!(
    "Converts BGR color values to `YCbCr422` color values.",
    bgr_to_ycbcr422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiBGRToYCbCr422_8u_C3C2R_Ctx
);

impl_color_convert_different_layout!(
    "Converts BGR color values to `YCbCr422` color values. Source alpha is ignored.",
    bgr_to_ycbcr422_u8_ac4_to_c2,
    u8,
    AC4,
    C2,
    nppiBGRToYCbCr422_8u_AC4C2R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to YCbCr color values.",
    bgr_to_ycbcr_u8_ac4_to_p3,
    u8,
    AC4,
    3,
    nppiBGRToYCbCr_8u_AC4P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to YCbCr color values.",
    bgr_to_ycbcr_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiBGRToYCbCr_8u_AC4P4R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts YCbCr color values to BGR color values.",
    ycbcr_to_bgr_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToBGR_8u_P3C3R_Ctx
);

impl_color_convert_different_layout!(
    "Converts `YCbCr422` color values to BGR color values.",
    ycbcr422_to_bgr_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYCbCr422ToBGR_8u_C2C3R_Ctx
);

impl_color_convert_different_layout_constant_alpha!(
    "Converts `YCbCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.",
    ycbcr422_to_bgr_u8_c2_to_c4,
    u8,
    C2,
    nppiYCbCr422ToBGR_8u_C2C4R_Ctx
);

impl_color_convert_planar_to_packed_constant_alpha!(
    "Converts YCbCr color values to BGR color values. Writes `alpha` to every destination alpha component.",
    ycbcr_to_bgr_u8_p3_to_c4,
    u8,
    3,
    nppiYCbCrToBGR_8u_P3C4R_Ctx
);

impl_color_convert_batch_same_layout!(
    "Converts YCbCr color values to BGR color values.",
    ycbcr_to_bgr_batch_u8_c3,
    u8,
    C3,
    nppiYCbCrToBGRBatch_8u_C3R_Ctx
);

impl_color_convert_batch_same_layout_advanced!(
    "Converts YCbCr color values to BGR color values.",
    ycbcr_to_bgr_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYCbCrToBGRBatch_8u_C3R_Advanced_Ctx
);

impl_color_convert_batch_planar_to_packed!(
    "Converts YCbCr color values to BGR color values.",
    ycbcr_to_bgr_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToBGRBatch_8u_P3C3R_Ctx
);

impl_color_convert_batch_planar_to_packed_advanced!(
    "Converts YCbCr color values to BGR color values.",
    ycbcr_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYCbCrToBGRBatch_8u_P3C3R_Advanced_Ctx
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts `YCbCr422` color values to BGR color values.",
    ycbcr422_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr422ToBGRBatch_8u_P3C3R_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts `YCbCr422` color values to BGR color values.",
    ycbcr422_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts `YCbCr420` color values to BGR color values.",
    ycbcr420_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr420ToBGRBatch_8u_P3C3R_Ctx,
    2,
    2
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts `YCbCr420` color values to BGR color values.",
    ycbcr420_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);

impl_color_convert_planar_to_packed!(
    "Converts YCbCr color values to BGR color values.",
    ycbcr_to_bgr_709csc_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_packed_constant_alpha!(
    "Converts YCbCr color values to BGR color values. Writes `alpha` to every destination alpha component.",
    ycbcr_to_bgr_709csc_u8_p3_to_c4,
    u8,
    3,
    nppiYCbCrToBGR_709CSC_8u_P3C4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to YCC color values.",
    rgb_to_ycc_u8_c3,
    u8,
    C3,
    nppiRGBToYCC_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to YCC color values. The destination alpha channel is not processed.",
    rgb_to_ycc_u8_ac4,
    u8,
    AC4,
    nppiRGBToYCC_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YCC color values to RGB color values.",
    ycc_to_rgb_u8_c3,
    u8,
    C3,
    nppiYCCToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YCC color values to RGB color values. The destination alpha channel is not processed.",
    ycc_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiYCCToRGB_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to YCC color values.",
    RgbToYccC3, rgb_to_ycc, rgb_to_ycc_c3, C3, [
    u8 => rgb_to_ycc_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to YCC color values. The destination alpha channel is not processed.",
    RgbToYccAC4, rgb_to_ycc, rgb_to_ycc_ac4, AC4, [
    u8 => rgb_to_ycc_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts YCC color values to RGB color values.",
    YccToRgbC3, ycc_to_rgb, ycc_to_rgb_c3, C3, [
    u8 => ycc_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts YCC color values to RGB color values. The destination alpha channel is not processed.",
    YccToRgbAC4, ycc_to_rgb, ycc_to_rgb_ac4, AC4, [
    u8 => ycc_to_rgb_u8_ac4,
    ]
);
