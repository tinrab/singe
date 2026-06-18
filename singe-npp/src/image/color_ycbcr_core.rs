use super::*;

impl_color_convert_same_layout!(rgb_to_ycbcr_u8_c3, u8, C3, nppiRGBToYCbCr_8u_C3R_Ctx);
impl_color_convert_same_layout!(rgb_to_ycbcr_u8_ac4, u8, AC4, nppiRGBToYCbCr_8u_AC4R_Ctx);
impl_generic_color_convert_same_layout!(RgbToYcbcrC3, rgb_to_ycbcr, rgb_to_ycbcr_c3, C3, [
    u8 => rgb_to_ycbcr_u8_c3,
]);
impl_generic_color_convert_same_layout!(RgbToYcbcrAC4, rgb_to_ycbcr, rgb_to_ycbcr_ac4, AC4, [
    u8 => rgb_to_ycbcr_u8_ac4,
]);
impl_color_convert_different_layout!(
    rgb_to_ycbcr422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToYCbCr422_8u_C3C2R_Ctx
);
impl_color_convert_different_layout!(
    rgb_to_ycrcb422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToYCrCb422_8u_C3C2R_Ctx
);
pub(crate) fn rgb_to_ycbcr422_u8_p3_to_c2(
    stream_context: &StreamContext,
    source_r: &ImageView<'_, u8, C1>,
    source_g: &ImageView<'_, u8, C1>,
    source_b: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C2>,
) -> Result<()> {
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
pub(crate) fn rgb_to_ycrcb422_u8_p3_to_c2(
    stream_context: &StreamContext,
    source_r: &ImageView<'_, u8, C1>,
    source_g: &ImageView<'_, u8, C1>,
    source_b: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C2>,
) -> Result<()> {
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
    rgb_to_cbycr422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToCbYCr422_8u_C3C2R_Ctx
);
impl_color_convert_different_layout!(
    rgb_to_cbycr422_gamma_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx
);
impl_color_convert_planar_to_planar!(rgb_to_ycbcr_u8_p3, u8, 3, nppiRGBToYCbCr_8u_P3R_Ctx);
impl_color_convert_same_layout!(ycbcr_to_rgb_u8_c3, u8, C3, nppiYCbCrToRGB_8u_C3R_Ctx);
impl_color_convert_same_layout!(ycbcr_to_rgb_u8_ac4, u8, AC4, nppiYCbCrToRGB_8u_AC4R_Ctx);
impl_generic_color_convert_same_layout!(YcbcrToRgbC3, ycbcr_to_rgb, ycbcr_to_rgb_c3, C3, [
    u8 => ycbcr_to_rgb_u8_c3,
]);
impl_generic_color_convert_same_layout!(
    YcbcrToRgbAC4,
    ycbcr_to_rgb,
    ycbcr_to_rgb_ac4,
    AC4,
    [u8 => ycbcr_to_rgb_u8_ac4]
);
impl_color_convert_different_layout!(
    ycbcr422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYCbCr422ToRGB_8u_C2C3R_Ctx
);
pub(crate) fn ycbcr422_to_rgb_u8_c2_to_p3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C2>,
    destination_r: &mut ImageViewMut<'_, u8, C1>,
    destination_g: &mut ImageViewMut<'_, u8, C1>,
    destination_b: &mut ImageViewMut<'_, u8, C1>,
) -> Result<()> {
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
    ycbcr422_to_ycrcb422_u8_c2,
    u8,
    C2,
    nppiYCbCr422ToYCrCb422_8u_C2R_Ctx
);
impl_generic_color_convert_same_layout!(
    Ycbcr422ToYcrcb422C2,
    ycbcr422_to_ycrcb422,
    ycbcr422_to_ycrcb422_c2,
    C2,
    [u8 => ycbcr422_to_ycrcb422_u8_c2]
);
impl_color_convert_same_layout!(
    ycbcr422_to_cbycr422_u8_c2,
    u8,
    C2,
    nppiYCbCr422ToCbYCr422_8u_C2R_Ctx
);
impl_generic_color_convert_same_layout!(
    Ycbcr422ToCbycr422C2,
    ycbcr422_to_cbycr422,
    ycbcr422_to_cbycr422_c2,
    C2,
    [u8 => ycbcr422_to_cbycr422_u8_c2]
);
impl_color_convert_different_layout!(
    ycrcb422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYCrCb422ToRGB_8u_C2C3R_Ctx
);
pub(crate) fn ycrcb422_to_rgb_u8_c2_to_p3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C2>,
    destination_r: &mut ImageViewMut<'_, u8, C1>,
    destination_g: &mut ImageViewMut<'_, u8, C1>,
    destination_b: &mut ImageViewMut<'_, u8, C1>,
) -> Result<()> {
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
    cbycr422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiCbYCr422ToRGB_8u_C2C3R_Ctx
);
impl_color_convert_different_layout!(
    bgr_to_cbycr422_u8_ac4_to_c2,
    u8,
    AC4,
    C2,
    nppiBGRToCbYCr422_8u_AC4C2R_Ctx
);
impl_color_convert_different_layout!(
    bgr_to_cbycr422_709hdtv_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiBGRToCbYCr422_709HDTV_8u_C3C2R_Ctx
);
impl_color_convert_different_layout!(
    bgr_to_cbycr422_709hdtv_u8_ac4_to_c2,
    u8,
    AC4,
    C2,
    nppiBGRToCbYCr422_709HDTV_8u_AC4C2R_Ctx
);
impl_color_convert_same_layout!(
    cbycr422_to_ycbcr422_u8_c2,
    u8,
    C2,
    nppiCbYCr422ToYCbCr422_8u_C2R_Ctx
);
impl_generic_color_convert_same_layout!(
    Cbycr422ToYcbcr422C2,
    cbycr422_to_ycbcr422,
    cbycr422_to_ycbcr422_c2,
    C2,
    [u8 => cbycr422_to_ycbcr422_u8_c2]
);
impl_color_convert_different_layout_constant_alpha!(
    cbycr422_to_bgr_u8_c2_to_c4,
    u8,
    C2,
    nppiCbYCr422ToBGR_8u_C2C4R_Ctx
);
impl_color_convert_different_layout!(
    cbycr422_to_bgr_709hdtv_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx
);
impl_color_convert_different_layout_constant_alpha!(
    cbycr422_to_bgr_709hdtv_u8_c2_to_c4,
    u8,
    C2,
    nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx
);
impl_color_convert_batch_same_layout!(
    ycbcr_to_rgb_batch_u8_c3,
    u8,
    C3,
    nppiYCbCrToRGBBatch_8u_C3R_Ctx
);
impl_color_convert_batch_same_layout_advanced!(
    ycbcr_to_rgb_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYCbCrToRGBBatch_8u_C3R_Advanced_Ctx
);
impl_color_convert_batch_planar_to_packed!(
    ycbcr_to_rgb_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToRGBBatch_8u_P3C3R_Ctx
);
impl_color_convert_batch_planar_to_packed_advanced!(
    ycbcr_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx
);
impl_subsampled_color_convert_batch_planar_to_packed!(
    ycbcr422_to_rgb_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx,
    2,
    1
);
impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    ycbcr422_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);
impl_subsampled_color_convert_batch_planar_to_packed!(
    ycbcr420_to_rgb_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx,
    2,
    2
);
impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    ycbcr420_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);
impl_color_convert_planar_to_planar!(ycbcr_to_rgb_u8_p3, u8, 3, nppiYCbCrToRGB_8u_P3R_Ctx);
impl_color_convert_packed_to_planar!(
    rgb_to_ycbcr_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiRGBToYCbCr_8u_C3P3R_Ctx
);
impl_color_convert_packed_to_planar!(
    rgb_to_ycbcr_u8_ac4_to_p3,
    u8,
    AC4,
    3,
    nppiRGBToYCbCr_8u_AC4P3R_Ctx
);
impl_color_convert_planar_to_packed!(
    ycbcr_to_rgb_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToRGB_8u_P3C3R_Ctx
);
impl_color_convert_planar_to_packed_constant_alpha!(
    ycbcr_to_rgb_u8_p3_to_c4,
    u8,
    3,
    nppiYCbCrToRGB_8u_P3C4R_Ctx
);
impl_color_convert_packed_to_planar!(
    bgr_to_ycbcr_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiBGRToYCbCr_8u_C3P3R_Ctx
);
impl_color_convert_different_layout!(
    bgr_to_ycbcr422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiBGRToYCbCr422_8u_C3C2R_Ctx
);
impl_color_convert_different_layout!(
    bgr_to_ycbcr422_u8_ac4_to_c2,
    u8,
    AC4,
    C2,
    nppiBGRToYCbCr422_8u_AC4C2R_Ctx
);
impl_color_convert_packed_to_planar!(
    bgr_to_ycbcr_u8_ac4_to_p3,
    u8,
    AC4,
    3,
    nppiBGRToYCbCr_8u_AC4P3R_Ctx
);
impl_color_convert_packed_to_planar!(
    bgr_to_ycbcr_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiBGRToYCbCr_8u_AC4P4R_Ctx
);
impl_color_convert_planar_to_packed!(
    ycbcr_to_bgr_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToBGR_8u_P3C3R_Ctx
);
impl_color_convert_different_layout!(
    ycbcr422_to_bgr_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYCbCr422ToBGR_8u_C2C3R_Ctx
);
impl_color_convert_different_layout_constant_alpha!(
    ycbcr422_to_bgr_u8_c2_to_c4,
    u8,
    C2,
    nppiYCbCr422ToBGR_8u_C2C4R_Ctx
);
impl_color_convert_planar_to_packed_constant_alpha!(
    ycbcr_to_bgr_u8_p3_to_c4,
    u8,
    3,
    nppiYCbCrToBGR_8u_P3C4R_Ctx
);
impl_color_convert_batch_same_layout!(
    ycbcr_to_bgr_batch_u8_c3,
    u8,
    C3,
    nppiYCbCrToBGRBatch_8u_C3R_Ctx
);
impl_color_convert_batch_same_layout_advanced!(
    ycbcr_to_bgr_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYCbCrToBGRBatch_8u_C3R_Advanced_Ctx
);
impl_color_convert_batch_planar_to_packed!(
    ycbcr_to_bgr_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToBGRBatch_8u_P3C3R_Ctx
);
impl_color_convert_batch_planar_to_packed_advanced!(
    ycbcr_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYCbCrToBGRBatch_8u_P3C3R_Advanced_Ctx
);
impl_subsampled_color_convert_batch_planar_to_packed!(
    ycbcr422_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr422ToBGRBatch_8u_P3C3R_Ctx,
    2,
    1
);
impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    ycbcr422_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);
impl_subsampled_color_convert_batch_planar_to_packed!(
    ycbcr420_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYCbCr420ToBGRBatch_8u_P3C3R_Ctx,
    2,
    2
);
impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    ycbcr420_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);
impl_color_convert_planar_to_packed!(
    ycbcr_to_bgr_709csc_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx
);
impl_color_convert_planar_to_packed_constant_alpha!(
    ycbcr_to_bgr_709csc_u8_p3_to_c4,
    u8,
    3,
    nppiYCbCrToBGR_709CSC_8u_P3C4R_Ctx
);
impl_color_convert_same_layout!(rgb_to_ycc_u8_c3, u8, C3, nppiRGBToYCC_8u_C3R_Ctx);
impl_color_convert_same_layout!(rgb_to_ycc_u8_ac4, u8, AC4, nppiRGBToYCC_8u_AC4R_Ctx);
impl_color_convert_same_layout!(ycc_to_rgb_u8_c3, u8, C3, nppiYCCToRGB_8u_C3R_Ctx);
impl_color_convert_same_layout!(ycc_to_rgb_u8_ac4, u8, AC4, nppiYCCToRGB_8u_AC4R_Ctx);
impl_generic_color_convert_same_layout!(RgbToYccC3, rgb_to_ycc, rgb_to_ycc_c3, C3, [
    u8 => rgb_to_ycc_u8_c3,
]);
impl_generic_color_convert_same_layout!(RgbToYccAC4, rgb_to_ycc, rgb_to_ycc_ac4, AC4, [
    u8 => rgb_to_ycc_u8_ac4,
]);
impl_generic_color_convert_same_layout!(YccToRgbC3, ycc_to_rgb, ycc_to_rgb_c3, C3, [
    u8 => ycc_to_rgb_u8_c3,
]);
impl_generic_color_convert_same_layout!(YccToRgbAC4, ycc_to_rgb, ycc_to_rgb_ac4, AC4, [
    u8 => ycc_to_rgb_u8_ac4,
]);
