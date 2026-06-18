use super::*;

pub(crate) fn convert_u8_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiConvert_8u32f_C1R_Ctx(
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

pub(crate) fn convert_u8_to_f32_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageViewMut<'_, f32, C3>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiConvert_8u32f_C3R_Ctx(
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

pub(crate) fn convert_u8_to_f32_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, f32, C4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiConvert_8u32f_C4R_Ctx(
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

pub(crate) fn convert_u8_to_f32_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageViewMut<'_, f32, AC4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiConvert_8u32f_AC4R_Ctx(
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

impl_convert!(convert_u8_to_u16_c1, u8, u16, C1, nppiConvert_8u16u_C1R_Ctx);
impl_convert!(convert_u8_to_u16_c3, u8, u16, C3, nppiConvert_8u16u_C3R_Ctx);
impl_convert!(convert_u8_to_u16_c4, u8, u16, C4, nppiConvert_8u16u_C4R_Ctx);
impl_convert!(
    convert_u8_to_u16_ac4,
    u8,
    u16,
    AC4,
    nppiConvert_8u16u_AC4R_Ctx
);
impl_convert!(convert_u8_to_i16_c1, u8, i16, C1, nppiConvert_8u16s_C1R_Ctx);
impl_convert!(convert_u8_to_i16_c3, u8, i16, C3, nppiConvert_8u16s_C3R_Ctx);
impl_convert!(convert_u8_to_i16_c4, u8, i16, C4, nppiConvert_8u16s_C4R_Ctx);
impl_convert!(
    convert_u8_to_i16_ac4,
    u8,
    i16,
    AC4,
    nppiConvert_8u16s_AC4R_Ctx
);
impl_convert!(convert_u8_to_i32_c1, u8, i32, C1, nppiConvert_8u32s_C1R_Ctx);
impl_convert!(convert_u8_to_i32_c3, u8, i32, C3, nppiConvert_8u32s_C3R_Ctx);
impl_convert!(convert_u8_to_i32_c4, u8, i32, C4, nppiConvert_8u32s_C4R_Ctx);
impl_convert!(
    convert_u8_to_i32_ac4,
    u8,
    i32,
    AC4,
    nppiConvert_8u32s_AC4R_Ctx
);
impl_convert!(convert_u16_to_u8_c1, u16, u8, C1, nppiConvert_16u8u_C1R_Ctx);
impl_convert!(convert_u16_to_u8_c3, u16, u8, C3, nppiConvert_16u8u_C3R_Ctx);
impl_convert!(convert_u16_to_u8_c4, u16, u8, C4, nppiConvert_16u8u_C4R_Ctx);
impl_convert!(
    convert_u16_to_u8_ac4,
    u16,
    u8,
    AC4,
    nppiConvert_16u8u_AC4R_Ctx
);
impl_convert!(convert_i16_to_u8_c1, i16, u8, C1, nppiConvert_16s8u_C1R_Ctx);
impl_convert!(convert_i16_to_u8_c3, i16, u8, C3, nppiConvert_16s8u_C3R_Ctx);
impl_convert!(convert_i16_to_u8_c4, i16, u8, C4, nppiConvert_16s8u_C4R_Ctx);
impl_convert!(
    convert_i16_to_u8_ac4,
    i16,
    u8,
    AC4,
    nppiConvert_16s8u_AC4R_Ctx
);
impl_convert!(convert_i32_to_u8_c1, i32, u8, C1, nppiConvert_32s8u_C1R_Ctx);
impl_convert!(convert_i32_to_u8_c3, i32, u8, C3, nppiConvert_32s8u_C3R_Ctx);
impl_convert!(convert_i32_to_u8_c4, i32, u8, C4, nppiConvert_32s8u_C4R_Ctx);
impl_convert!(
    convert_i32_to_u8_ac4,
    i32,
    u8,
    AC4,
    nppiConvert_32s8u_AC4R_Ctx
);
impl_convert!(
    convert_u16_to_i32_c1,
    u16,
    i32,
    C1,
    nppiConvert_16u32s_C1R_Ctx
);
impl_convert!(
    convert_u16_to_i32_c3,
    u16,
    i32,
    C3,
    nppiConvert_16u32s_C3R_Ctx
);
impl_convert!(
    convert_u16_to_i32_c4,
    u16,
    i32,
    C4,
    nppiConvert_16u32s_C4R_Ctx
);
impl_convert!(
    convert_u16_to_i32_ac4,
    u16,
    i32,
    AC4,
    nppiConvert_16u32s_AC4R_Ctx
);
impl_convert!(
    convert_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiConvert_16u32f_C1R_Ctx
);
impl_convert!(
    convert_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiConvert_16u32f_C3R_Ctx
);
impl_convert!(
    convert_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiConvert_16u32f_C4R_Ctx
);
impl_convert!(
    convert_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiConvert_16u32f_AC4R_Ctx
);
impl_convert!(
    convert_i16_to_i32_c1,
    i16,
    i32,
    C1,
    nppiConvert_16s32s_C1R_Ctx
);
impl_convert!(
    convert_i16_to_i32_c3,
    i16,
    i32,
    C3,
    nppiConvert_16s32s_C3R_Ctx
);
impl_convert!(
    convert_i16_to_i32_c4,
    i16,
    i32,
    C4,
    nppiConvert_16s32s_C4R_Ctx
);
impl_convert!(
    convert_i16_to_i32_ac4,
    i16,
    i32,
    AC4,
    nppiConvert_16s32s_AC4R_Ctx
);
impl_convert!(
    convert_i16_to_f32_c1,
    i16,
    f32,
    C1,
    nppiConvert_16s32f_C1R_Ctx
);
impl_convert!(
    convert_i16_to_f32_c3,
    i16,
    f32,
    C3,
    nppiConvert_16s32f_C3R_Ctx
);
impl_convert!(
    convert_i16_to_f32_c4,
    i16,
    f32,
    C4,
    nppiConvert_16s32f_C4R_Ctx
);
impl_convert!(
    convert_i16_to_f32_ac4,
    i16,
    f32,
    AC4,
    nppiConvert_16s32f_AC4R_Ctx
);
impl_convert!(
    convert_f16_to_f32_c1,
    f16,
    f32,
    C1,
    nppiConvert_16f32f_C1R_Ctx
);
impl_convert!(
    convert_f16_to_f32_c3,
    f16,
    f32,
    C3,
    nppiConvert_16f32f_C3R_Ctx
);
impl_convert!(
    convert_f16_to_f32_c4,
    f16,
    f32,
    C4,
    nppiConvert_16f32f_C4R_Ctx
);
impl_convert!(
    convert_f16_to_f32_ac4,
    f16,
    f32,
    AC4,
    nppiConvert_16f32f_AC4R_Ctx
);
impl_convert!(
    convert_u16_to_u32_c1,
    u16,
    u32,
    C1,
    nppiConvert_16u32u_C1R_Ctx
);
impl_convert_round!(
    convert_f32_to_u16_c1,
    f32,
    u16,
    C1,
    nppiConvert_32f16u_C1R_Ctx
);
impl_convert_round!(
    convert_f32_to_u16_c3,
    f32,
    u16,
    C3,
    nppiConvert_32f16u_C3R_Ctx
);
impl_convert_round!(
    convert_f32_to_u16_c4,
    f32,
    u16,
    C4,
    nppiConvert_32f16u_C4R_Ctx
);
impl_convert_round!(
    convert_f32_to_u16_ac4,
    f32,
    u16,
    AC4,
    nppiConvert_32f16u_AC4R_Ctx
);
impl_convert_round!(
    convert_f32_to_i16_c1,
    f32,
    i16,
    C1,
    nppiConvert_32f16s_C1R_Ctx
);
impl_convert_round!(
    convert_f32_to_i16_c3,
    f32,
    i16,
    C3,
    nppiConvert_32f16s_C3R_Ctx
);
impl_convert_round!(
    convert_f32_to_i16_c4,
    f32,
    i16,
    C4,
    nppiConvert_32f16s_C4R_Ctx
);
impl_convert_round!(
    convert_f32_to_i16_ac4,
    f32,
    i16,
    AC4,
    nppiConvert_32f16s_AC4R_Ctx
);
impl_convert_round!(
    convert_f32_to_f16_c1,
    f32,
    f16,
    C1,
    nppiConvert_32f16f_C1R_Ctx
);
impl_convert_round!(
    convert_f32_to_f16_c3,
    f32,
    f16,
    C3,
    nppiConvert_32f16f_C3R_Ctx
);
impl_convert_round!(
    convert_f32_to_f16_c4,
    f32,
    f16,
    C4,
    nppiConvert_32f16f_C4R_Ctx
);
impl_convert_round!(
    convert_f32_to_f16_ac4,
    f32,
    f16,
    AC4,
    nppiConvert_32f16f_AC4R_Ctx
);
impl_generic_convert_to_operation!(
    ConvertToU16C1,
    convert_to_u16_c1,
    u16,
    C1,
    [
        u8 => convert_u8_to_u16_c1,
        i8 => convert_i8_to_u16_c1,
        i16 => convert_i16_to_u16_c1,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToU16C3,
    convert_to_u16_c3,
    u16,
    C3,
    [u8 => convert_u8_to_u16_c3]
);
impl_generic_convert_to_operation!(
    ConvertToU16C4,
    convert_to_u16_c4,
    u16,
    C4,
    [u8 => convert_u8_to_u16_c4]
);
impl_generic_convert_to_operation!(
    ConvertToU16AC4,
    convert_to_u16_ac4,
    u16,
    AC4,
    [u8 => convert_u8_to_u16_ac4]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU16C1,
    convert_round_to_u16_c1,
    u16,
    C1,
    [f32 => convert_f32_to_u16_c1]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU16C3,
    convert_round_to_u16_c3,
    u16,
    C3,
    [f32 => convert_f32_to_u16_c3]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU16C4,
    convert_round_to_u16_c4,
    u16,
    C4,
    [f32 => convert_f32_to_u16_c4]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU16AC4,
    convert_round_to_u16_ac4,
    u16,
    AC4,
    [f32 => convert_f32_to_u16_ac4]
);
impl_generic_convert_to_operation!(
    ConvertToI16C1,
    convert_to_i16_c1,
    i16,
    C1,
    [
        u8 => convert_u8_to_i16_c1,
        i8 => convert_i8_to_i16_c1,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToI16C3,
    convert_to_i16_c3,
    i16,
    C3,
    [u8 => convert_u8_to_i16_c3]
);
impl_generic_convert_to_operation!(
    ConvertToI16C4,
    convert_to_i16_c4,
    i16,
    C4,
    [u8 => convert_u8_to_i16_c4]
);
impl_generic_convert_to_operation!(
    ConvertToI16AC4,
    convert_to_i16_ac4,
    i16,
    AC4,
    [u8 => convert_u8_to_i16_ac4]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToI16C1,
    convert_round_to_i16_c1,
    i16,
    C1,
    [f32 => convert_f32_to_i16_c1]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToI16C3,
    convert_round_to_i16_c3,
    i16,
    C3,
    [f32 => convert_f32_to_i16_c3]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToI16C4,
    convert_round_to_i16_c4,
    i16,
    C4,
    [f32 => convert_f32_to_i16_c4]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToI16AC4,
    convert_round_to_i16_ac4,
    i16,
    AC4,
    [f32 => convert_f32_to_i16_ac4]
);
