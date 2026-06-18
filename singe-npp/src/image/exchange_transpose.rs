use super::*;

pub(crate) fn transpose_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C1>,
) -> Result<()> {
    validate_transpose_size(source.size(), destination.size())?;

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

pub(crate) fn transpose_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageViewMut<'_, u8, C3>,
) -> Result<()> {
    validate_transpose_size(source.size(), destination.size())?;

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

pub(crate) fn transpose_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, u8, C4>,
) -> Result<()> {
    validate_transpose_size(source.size(), destination.size())?;

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

impl_transpose!(transpose_u16_c1, u16, C1, nppiTranspose_16u_C1R_Ctx);
impl_transpose!(transpose_u16_c3, u16, C3, nppiTranspose_16u_C3R_Ctx);
impl_transpose!(transpose_u16_c4, u16, C4, nppiTranspose_16u_C4R_Ctx);
impl_transpose!(transpose_i16_c1, i16, C1, nppiTranspose_16s_C1R_Ctx);
impl_transpose!(transpose_i16_c3, i16, C3, nppiTranspose_16s_C3R_Ctx);
impl_transpose!(transpose_i16_c4, i16, C4, nppiTranspose_16s_C4R_Ctx);
impl_transpose!(transpose_i32_c1, i32, C1, nppiTranspose_32s_C1R_Ctx);
impl_transpose!(transpose_i32_c3, i32, C3, nppiTranspose_32s_C3R_Ctx);
impl_transpose!(transpose_i32_c4, i32, C4, nppiTranspose_32s_C4R_Ctx);
impl_transpose!(transpose_f32_c1, f32, C1, nppiTranspose_32f_C1R_Ctx);
impl_transpose!(transpose_f32_c3, f32, C3, nppiTranspose_32f_C3R_Ctx);
impl_transpose!(transpose_f32_c4, f32, C4, nppiTranspose_32f_C4R_Ctx);

impl_generic_transpose_operation!(TransposeC1, transpose, transpose_c1, C1, [
    u8 => transpose_u8_c1,
    u16 => transpose_u16_c1,
    i16 => transpose_i16_c1,
    i32 => transpose_i32_c1,
    f32 => transpose_f32_c1,
]);
impl_generic_transpose_operation!(TransposeC3, transpose, transpose_c3, C3, [
    u8 => transpose_u8_c3,
    u16 => transpose_u16_c3,
    i16 => transpose_i16_c3,
    i32 => transpose_i32_c3,
    f32 => transpose_f32_c3,
]);
impl_generic_transpose_operation!(TransposeC4, transpose, transpose_c4, C4, [
    u8 => transpose_u8_c4,
    u16 => transpose_u16_c4,
    i16 => transpose_i16_c4,
    i32 => transpose_i32_c4,
    f32 => transpose_f32_c4,
]);
