use super::*;

pub(crate) fn copy_u8_c3_to_p3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut PlanarImageViewMut<'_, u8, 3>,
) -> Result<()> {
    validate_same_size(source.size(), destination.planes()[0].size())?;

    let destination_planes = destination
        .planes_mut()
        .each_mut()
        .map(ImageViewMut::as_mut_ptr);

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

pub(crate) fn copy_u8_c4_to_p4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut PlanarImageViewMut<'_, u8, 4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.planes()[0].size())?;

    let destination_planes = destination
        .planes_mut()
        .each_mut()
        .map(ImageViewMut::as_mut_ptr);

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

pub(crate) fn copy_u8_p3_to_c3(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, u8, 3>,
    destination: &mut ImageViewMut<'_, u8, C3>,
) -> Result<()> {
    validate_same_size(source.planes()[0].size(), destination.size())?;

    let source_planes = source.planes().map(|plane| plane.as_ptr());

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

pub(crate) fn copy_u8_p4_to_c4(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, u8, 4>,
    destination: &mut ImageViewMut<'_, u8, C4>,
) -> Result<()> {
    validate_same_size(source.planes()[0].size(), destination.size())?;

    let source_planes = source.planes().map(|plane| plane.as_ptr());

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
