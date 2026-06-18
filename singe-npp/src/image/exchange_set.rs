use super::*;

pub(crate) fn set_u8_c1(
    stream_context: &StreamContext,
    value: u8,
    destination: &mut ImageViewMut<'_, u8, C1>,
) -> Result<()> {
    unsafe {
        try_ffi!(sys::nppiSet_8u_C1R_Ctx(
            value,
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_u8_c2(
    stream_context: &StreamContext,
    value: [u8; 2],
    destination: &mut ImageViewMut<'_, u8, C2>,
) -> Result<()> {
    unsafe {
        try_ffi!(sys::nppiSet_8u_C2R_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_u8_c3(
    stream_context: &StreamContext,
    value: [u8; 3],
    destination: &mut ImageViewMut<'_, u8, C3>,
) -> Result<()> {
    unsafe {
        try_ffi!(sys::nppiSet_8u_C3R_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_u8_c4(
    stream_context: &StreamContext,
    value: [u8; 4],
    destination: &mut ImageViewMut<'_, u8, C4>,
) -> Result<()> {
    unsafe {
        try_ffi!(sys::nppiSet_8u_C4R_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_u8_ac4(
    stream_context: &StreamContext,
    value: [u8; 3],
    destination: &mut ImageViewMut<'_, u8, AC4>,
) -> Result<()> {
    unsafe {
        try_ffi!(sys::nppiSet_8u_AC4R_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_masked_u8_c1(
    stream_context: &StreamContext,
    value: u8,
    destination: &mut ImageViewMut<'_, u8, C1>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_mask_size(destination.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiSet_8u_C1MR_Ctx(
            value,
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_masked_u8_c3(
    stream_context: &StreamContext,
    value: [u8; 3],
    destination: &mut ImageViewMut<'_, u8, C3>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_mask_size(destination.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiSet_8u_C3MR_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_masked_u8_c4(
    stream_context: &StreamContext,
    value: [u8; 4],
    destination: &mut ImageViewMut<'_, u8, C4>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_mask_size(destination.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiSet_8u_C4MR_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_masked_u8_ac4(
    stream_context: &StreamContext,
    value: [u8; 3],
    destination: &mut ImageViewMut<'_, u8, AC4>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_mask_size(destination.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiSet_8u_AC4MR_Ctx(
            value.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_set!(set_i8_c1, i8, i8, C1, nppiSet_8s_C1R_Ctx, |value: &i8| {
    *value
});
impl_set!(
    set_i8_c2,
    i8,
    [i8; 2],
    C2,
    nppiSet_8s_C2R_Ctx,
    |value: &[i8; 2]| value.as_ptr()
);
impl_set!(
    set_i8_c3,
    i8,
    [i8; 3],
    C3,
    nppiSet_8s_C3R_Ctx,
    |value: &[i8; 3]| value.as_ptr()
);
impl_set!(
    set_i8_c4,
    i8,
    [i8; 4],
    C4,
    nppiSet_8s_C4R_Ctx,
    |value: &[i8; 4]| value.as_ptr()
);
impl_set!(
    set_i8_ac4,
    i8,
    [i8; 3],
    AC4,
    nppiSet_8s_AC4R_Ctx,
    |value: &[i8; 3]| value.as_ptr()
);
impl_set!(
    set_u16_c1,
    u16,
    u16,
    C1,
    nppiSet_16u_C1R_Ctx,
    |value: &u16| *value
);
impl_set!(
    set_u16_c2,
    u16,
    [u16; 2],
    C2,
    nppiSet_16u_C2R_Ctx,
    |value: &[u16; 2]| value.as_ptr()
);
impl_set!(
    set_u16_c3,
    u16,
    [u16; 3],
    C3,
    nppiSet_16u_C3R_Ctx,
    |value: &[u16; 3]| value.as_ptr()
);
impl_set!(
    set_u16_c4,
    u16,
    [u16; 4],
    C4,
    nppiSet_16u_C4R_Ctx,
    |value: &[u16; 4]| value.as_ptr()
);
impl_set!(
    set_u16_ac4,
    u16,
    [u16; 3],
    AC4,
    nppiSet_16u_AC4R_Ctx,
    |value: &[u16; 3]| value.as_ptr()
);
impl_set!(
    set_f16_c1,
    f16,
    f32,
    C1,
    nppiSet_16f_C1R_Ctx,
    |value: &f32| *value
);
impl_set!(
    set_f16_c2,
    f16,
    [f32; 2],
    C2,
    nppiSet_16f_C2R_Ctx,
    |value: &[f32; 2]| value.as_ptr()
);
impl_set!(
    set_f16_c3,
    f16,
    [f32; 3],
    C3,
    nppiSet_16f_C3R_Ctx,
    |value: &[f32; 3]| value.as_ptr()
);
impl_set!(
    set_f16_c4,
    f16,
    [f32; 4],
    C4,
    nppiSet_16f_C4R_Ctx,
    |value: &[f32; 4]| value.as_ptr()
);
impl_set!(
    set_i16_c1,
    i16,
    i16,
    C1,
    nppiSet_16s_C1R_Ctx,
    |value: &i16| *value
);
impl_set!(
    set_i16_c2,
    i16,
    [i16; 2],
    C2,
    nppiSet_16s_C2R_Ctx,
    |value: &[i16; 2]| value.as_ptr()
);
impl_set!(
    set_i16_c3,
    i16,
    [i16; 3],
    C3,
    nppiSet_16s_C3R_Ctx,
    |value: &[i16; 3]| value.as_ptr()
);
impl_set!(
    set_i16_c4,
    i16,
    [i16; 4],
    C4,
    nppiSet_16s_C4R_Ctx,
    |value: &[i16; 4]| value.as_ptr()
);
impl_set!(
    set_i16_ac4,
    i16,
    [i16; 3],
    AC4,
    nppiSet_16s_AC4R_Ctx,
    |value: &[i16; 3]| value.as_ptr()
);
impl_set!(
    set_i32_c1,
    i32,
    i32,
    C1,
    nppiSet_32s_C1R_Ctx,
    |value: &i32| *value
);
impl_set!(
    set_i32_c2,
    i32,
    [i32; 2],
    C2,
    nppiSet_32s_C2R_Ctx,
    |value: &[i32; 2]| value.as_ptr()
);
impl_set!(
    set_i32_c3,
    i32,
    [i32; 3],
    C3,
    nppiSet_32s_C3R_Ctx,
    |value: &[i32; 3]| value.as_ptr()
);
impl_set!(
    set_i32_c4,
    i32,
    [i32; 4],
    C4,
    nppiSet_32s_C4R_Ctx,
    |value: &[i32; 4]| value.as_ptr()
);
impl_set!(
    set_i32_ac4,
    i32,
    [i32; 3],
    AC4,
    nppiSet_32s_AC4R_Ctx,
    |value: &[i32; 3]| value.as_ptr()
);
impl_set!(
    set_u32_c1,
    u32,
    u32,
    C1,
    nppiSet_32u_C1R_Ctx,
    |value: &u32| *value
);
impl_set!(
    set_u32_c2,
    u32,
    [u32; 2],
    C2,
    nppiSet_32u_C2R_Ctx,
    |value: &[u32; 2]| value.as_ptr()
);
impl_set!(
    set_u32_c3,
    u32,
    [u32; 3],
    C3,
    nppiSet_32u_C3R_Ctx,
    |value: &[u32; 3]| value.as_ptr()
);
impl_set!(
    set_u32_c4,
    u32,
    [u32; 4],
    C4,
    nppiSet_32u_C4R_Ctx,
    |value: &[u32; 4]| value.as_ptr()
);
impl_set!(
    set_u32_ac4,
    u32,
    [u32; 3],
    AC4,
    nppiSet_32u_AC4R_Ctx,
    |value: &[u32; 3]| value.as_ptr()
);
impl_set!(
    set_i16_complex_c1,
    ComplexI16,
    ComplexI16,
    C1,
    nppiSet_16sc_C1R_Ctx,
    |value: &ComplexI16| (*value).into_npp()
);
impl_set!(
    set_i16_complex_c2,
    ComplexI16,
    [ComplexI16; 2],
    C2,
    nppiSet_16sc_C2R_Ctx,
    |value: &[ComplexI16; 2]| value.as_ptr().cast()
);
impl_set!(
    set_i16_complex_c3,
    ComplexI16,
    [ComplexI16; 3],
    C3,
    nppiSet_16sc_C3R_Ctx,
    |value: &[ComplexI16; 3]| value.as_ptr().cast()
);
impl_set!(
    set_i16_complex_c4,
    ComplexI16,
    [ComplexI16; 4],
    C4,
    nppiSet_16sc_C4R_Ctx,
    |value: &[ComplexI16; 4]| value.as_ptr().cast()
);
impl_set!(
    set_i16_complex_ac4,
    ComplexI16,
    [ComplexI16; 3],
    AC4,
    nppiSet_16sc_AC4R_Ctx,
    |value: &[ComplexI16; 3]| value.as_ptr().cast()
);
impl_set!(
    set_f32_complex_c1,
    Complex32,
    Complex32,
    C1,
    nppiSet_32fc_C1R_Ctx,
    |value: &Complex32| (*value).into_npp()
);
impl_set!(
    set_f32_complex_c2,
    Complex32,
    [Complex32; 2],
    C2,
    nppiSet_32fc_C2R_Ctx,
    |value: &[Complex32; 2]| value.as_ptr().cast()
);
impl_set!(
    set_f32_complex_c3,
    Complex32,
    [Complex32; 3],
    C3,
    nppiSet_32fc_C3R_Ctx,
    |value: &[Complex32; 3]| value.as_ptr().cast()
);
impl_set!(
    set_f32_complex_c4,
    Complex32,
    [Complex32; 4],
    C4,
    nppiSet_32fc_C4R_Ctx,
    |value: &[Complex32; 4]| value.as_ptr().cast()
);
impl_set!(
    set_f32_complex_ac4,
    Complex32,
    [Complex32; 3],
    AC4,
    nppiSet_32fc_AC4R_Ctx,
    |value: &[Complex32; 3]| value.as_ptr().cast()
);
impl_set!(
    set_i32_complex_c1,
    ComplexI32,
    ComplexI32,
    C1,
    nppiSet_32sc_C1R_Ctx,
    |value: &ComplexI32| (*value).into_npp()
);
impl_set!(
    set_i32_complex_c2,
    ComplexI32,
    [ComplexI32; 2],
    C2,
    nppiSet_32sc_C2R_Ctx,
    |value: &[ComplexI32; 2]| value.as_ptr().cast()
);
impl_set!(
    set_i32_complex_c3,
    ComplexI32,
    [ComplexI32; 3],
    C3,
    nppiSet_32sc_C3R_Ctx,
    |value: &[ComplexI32; 3]| value.as_ptr().cast()
);
impl_set!(
    set_i32_complex_c4,
    ComplexI32,
    [ComplexI32; 4],
    C4,
    nppiSet_32sc_C4R_Ctx,
    |value: &[ComplexI32; 4]| value.as_ptr().cast()
);
impl_set!(
    set_i32_complex_ac4,
    ComplexI32,
    [ComplexI32; 3],
    AC4,
    nppiSet_32sc_AC4R_Ctx,
    |value: &[ComplexI32; 3]| value.as_ptr().cast()
);
impl_set!(
    set_f32_c1,
    f32,
    f32,
    C1,
    nppiSet_32f_C1R_Ctx,
    |value: &f32| *value
);
impl_set!(
    set_f32_c2,
    f32,
    [f32; 2],
    C2,
    nppiSet_32f_C2R_Ctx,
    |value: &[f32; 2]| value.as_ptr()
);
impl_set!(
    set_f32_c3,
    f32,
    [f32; 3],
    C3,
    nppiSet_32f_C3R_Ctx,
    |value: &[f32; 3]| value.as_ptr()
);
impl_set!(
    set_f32_c4,
    f32,
    [f32; 4],
    C4,
    nppiSet_32f_C4R_Ctx,
    |value: &[f32; 4]| value.as_ptr()
);
impl_set!(
    set_f32_ac4,
    f32,
    [f32; 3],
    AC4,
    nppiSet_32f_AC4R_Ctx,
    |value: &[f32; 3]| value.as_ptr()
);

impl_set_masked!(
    set_masked_u16_c1,
    u16,
    u16,
    C1,
    nppiSet_16u_C1MR_Ctx,
    |value: &u16| *value
);
impl_set_masked!(
    set_masked_u16_c3,
    u16,
    [u16; 3],
    C3,
    nppiSet_16u_C3MR_Ctx,
    |value: &[u16; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_u16_c4,
    u16,
    [u16; 4],
    C4,
    nppiSet_16u_C4MR_Ctx,
    |value: &[u16; 4]| value.as_ptr()
);
impl_set_masked!(
    set_masked_u16_ac4,
    u16,
    [u16; 3],
    AC4,
    nppiSet_16u_AC4MR_Ctx,
    |value: &[u16; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_i16_c1,
    i16,
    i16,
    C1,
    nppiSet_16s_C1MR_Ctx,
    |value: &i16| *value
);
impl_set_masked!(
    set_masked_i16_c3,
    i16,
    [i16; 3],
    C3,
    nppiSet_16s_C3MR_Ctx,
    |value: &[i16; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_i16_c4,
    i16,
    [i16; 4],
    C4,
    nppiSet_16s_C4MR_Ctx,
    |value: &[i16; 4]| value.as_ptr()
);
impl_set_masked!(
    set_masked_i16_ac4,
    i16,
    [i16; 3],
    AC4,
    nppiSet_16s_AC4MR_Ctx,
    |value: &[i16; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_i32_c1,
    i32,
    i32,
    C1,
    nppiSet_32s_C1MR_Ctx,
    |value: &i32| *value
);
impl_set_masked!(
    set_masked_i32_c3,
    i32,
    [i32; 3],
    C3,
    nppiSet_32s_C3MR_Ctx,
    |value: &[i32; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_i32_c4,
    i32,
    [i32; 4],
    C4,
    nppiSet_32s_C4MR_Ctx,
    |value: &[i32; 4]| value.as_ptr()
);
impl_set_masked!(
    set_masked_i32_ac4,
    i32,
    [i32; 3],
    AC4,
    nppiSet_32s_AC4MR_Ctx,
    |value: &[i32; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_f32_c1,
    f32,
    f32,
    C1,
    nppiSet_32f_C1MR_Ctx,
    |value: &f32| *value
);
impl_set_masked!(
    set_masked_f32_c3,
    f32,
    [f32; 3],
    C3,
    nppiSet_32f_C3MR_Ctx,
    |value: &[f32; 3]| value.as_ptr()
);
impl_set_masked!(
    set_masked_f32_c4,
    f32,
    [f32; 4],
    C4,
    nppiSet_32f_C4MR_Ctx,
    |value: &[f32; 4]| value.as_ptr()
);
impl_set_masked!(
    set_masked_f32_ac4,
    f32,
    [f32; 3],
    AC4,
    nppiSet_32f_AC4MR_Ctx,
    |value: &[f32; 3]| value.as_ptr()
);
impl_generic_set_operation!(
    SetC1,
    set_c1,
    C1,
    [
        u8 => (u8, set_u8_c1),
        i8 => (i8, set_i8_c1),
        u16 => (u16, set_u16_c1),
        f16 => (f32, set_f16_c1),
        i16 => (i16, set_i16_c1),
        i32 => (i32, set_i32_c1),
        u32 => (u32, set_u32_c1),
        ComplexI16 => (ComplexI16, set_i16_complex_c1),
        Complex32 => (Complex32, set_f32_complex_c1),
        ComplexI32 => (ComplexI32, set_i32_complex_c1),
        f32 => (f32, set_f32_c1),
    ]
);
impl_generic_set_operation!(
    SetC2,
    set_c2,
    C2,
    [
        u8 => ([u8; 2], set_u8_c2),
        i8 => ([i8; 2], set_i8_c2),
        u16 => ([u16; 2], set_u16_c2),
        f16 => ([f32; 2], set_f16_c2),
        i16 => ([i16; 2], set_i16_c2),
        i32 => ([i32; 2], set_i32_c2),
        u32 => ([u32; 2], set_u32_c2),
        ComplexI16 => ([ComplexI16; 2], set_i16_complex_c2),
        Complex32 => ([Complex32; 2], set_f32_complex_c2),
        ComplexI32 => ([ComplexI32; 2], set_i32_complex_c2),
        f32 => ([f32; 2], set_f32_c2),
    ]
);
impl_generic_set_operation!(
    SetC3,
    set_c3,
    C3,
    [
        u8 => ([u8; 3], set_u8_c3),
        i8 => ([i8; 3], set_i8_c3),
        u16 => ([u16; 3], set_u16_c3),
        f16 => ([f32; 3], set_f16_c3),
        i16 => ([i16; 3], set_i16_c3),
        i32 => ([i32; 3], set_i32_c3),
        u32 => ([u32; 3], set_u32_c3),
        ComplexI16 => ([ComplexI16; 3], set_i16_complex_c3),
        Complex32 => ([Complex32; 3], set_f32_complex_c3),
        ComplexI32 => ([ComplexI32; 3], set_i32_complex_c3),
        f32 => ([f32; 3], set_f32_c3),
    ]
);
impl_generic_set_operation!(
    SetC4,
    set_c4,
    C4,
    [
        u8 => ([u8; 4], set_u8_c4),
        i8 => ([i8; 4], set_i8_c4),
        u16 => ([u16; 4], set_u16_c4),
        f16 => ([f32; 4], set_f16_c4),
        i16 => ([i16; 4], set_i16_c4),
        i32 => ([i32; 4], set_i32_c4),
        u32 => ([u32; 4], set_u32_c4),
        ComplexI16 => ([ComplexI16; 4], set_i16_complex_c4),
        Complex32 => ([Complex32; 4], set_f32_complex_c4),
        ComplexI32 => ([ComplexI32; 4], set_i32_complex_c4),
        f32 => ([f32; 4], set_f32_c4),
    ]
);
impl_generic_set_operation!(
    SetAC4,
    set_ac4,
    AC4,
    [
        u8 => ([u8; 3], set_u8_ac4),
        i8 => ([i8; 3], set_i8_ac4),
        u16 => ([u16; 3], set_u16_ac4),
        i16 => ([i16; 3], set_i16_ac4),
        i32 => ([i32; 3], set_i32_ac4),
        u32 => ([u32; 3], set_u32_ac4),
        ComplexI16 => ([ComplexI16; 3], set_i16_complex_ac4),
        Complex32 => ([Complex32; 3], set_f32_complex_ac4),
        ComplexI32 => ([ComplexI32; 3], set_i32_complex_ac4),
        f32 => ([f32; 3], set_f32_ac4),
    ]
);
impl_generic_set_masked_operation!(
    SetMaskedC1,
    set_masked_c1,
    C1,
    [
        u8 => (u8, set_masked_u8_c1),
        u16 => (u16, set_masked_u16_c1),
        i16 => (i16, set_masked_i16_c1),
        i32 => (i32, set_masked_i32_c1),
        f32 => (f32, set_masked_f32_c1),
    ]
);
impl_generic_set_masked_operation!(
    SetMaskedC3,
    set_masked_c3,
    C3,
    [
        u8 => ([u8; 3], set_masked_u8_c3),
        u16 => ([u16; 3], set_masked_u16_c3),
        i16 => ([i16; 3], set_masked_i16_c3),
        i32 => ([i32; 3], set_masked_i32_c3),
        f32 => ([f32; 3], set_masked_f32_c3),
    ]
);
impl_generic_set_masked_operation!(
    SetMaskedC4,
    set_masked_c4,
    C4,
    [
        u8 => ([u8; 4], set_masked_u8_c4),
        u16 => ([u16; 4], set_masked_u16_c4),
        i16 => ([i16; 4], set_masked_i16_c4),
        i32 => ([i32; 4], set_masked_i32_c4),
        f32 => ([f32; 4], set_masked_f32_c4),
    ]
);
impl_generic_set_masked_operation!(
    SetMaskedAC4,
    set_masked_ac4,
    AC4,
    [
        u8 => ([u8; 3], set_masked_u8_ac4),
        u16 => ([u16; 3], set_masked_u16_ac4),
        i16 => ([i16; 3], set_masked_i16_ac4),
        i32 => ([i32; 3], set_masked_i32_ac4),
        f32 => ([f32; 3], set_masked_f32_ac4),
    ]
);
