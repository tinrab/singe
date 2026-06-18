use super::*;

pub(crate) fn copy_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C1>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C1R_Ctx(
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

pub(crate) fn copy_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageViewMut<'_, u8, C3>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3R_Ctx(
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

pub(crate) fn copy_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, u8, C4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4R_Ctx(
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

pub(crate) fn copy_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageViewMut<'_, u8, AC4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_AC4R_Ctx(
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

impl_copy!(copy_i8_c1, i8, C1, nppiCopy_8s_C1R_Ctx);
impl_copy!(copy_i8_c2, i8, C2, nppiCopy_8s_C2R_Ctx);
impl_copy!(copy_i8_c3, i8, C3, nppiCopy_8s_C3R_Ctx);
impl_copy!(copy_i8_c4, i8, C4, nppiCopy_8s_C4R_Ctx);
impl_copy!(copy_i8_ac4, i8, AC4, nppiCopy_8s_AC4R_Ctx);
impl_copy!(copy_u16_c1, u16, C1, nppiCopy_16u_C1R_Ctx);
impl_copy!(copy_u16_c3, u16, C3, nppiCopy_16u_C3R_Ctx);
impl_copy!(copy_u16_c4, u16, C4, nppiCopy_16u_C4R_Ctx);
impl_copy!(copy_u16_ac4, u16, AC4, nppiCopy_16u_AC4R_Ctx);
impl_copy!(copy_i16_c1, i16, C1, nppiCopy_16s_C1R_Ctx);
impl_copy!(copy_i16_c3, i16, C3, nppiCopy_16s_C3R_Ctx);
impl_copy!(copy_i16_c4, i16, C4, nppiCopy_16s_C4R_Ctx);
impl_copy!(copy_i16_ac4, i16, AC4, nppiCopy_16s_AC4R_Ctx);
impl_copy!(copy_u32_c1, u32, C1, nppiCopy_32s_C1R_Ctx);
impl_copy!(copy_u32_ac4, u32, AC4, nppiCopy_32s_AC4R_Ctx);
impl_copy!(copy_i32_c1, i32, C1, nppiCopy_32s_C1R_Ctx);
impl_copy!(copy_i32_c3, i32, C3, nppiCopy_32s_C3R_Ctx);
impl_copy!(copy_i32_c4, i32, C4, nppiCopy_32s_C4R_Ctx);
impl_copy!(copy_i32_ac4, i32, AC4, nppiCopy_32s_AC4R_Ctx);
impl_copy!(copy_f16_c1, f16, C1, nppiCopy_16f_C1R_Ctx);
impl_copy!(copy_f16_c3, f16, C3, nppiCopy_16f_C3R_Ctx);
impl_copy!(copy_f16_c4, f16, C4, nppiCopy_16f_C4R_Ctx);
impl_copy!(copy_f32_c1, f32, C1, nppiCopy_32f_C1R_Ctx);
impl_copy!(copy_f32_c3, f32, C3, nppiCopy_32f_C3R_Ctx);
impl_copy!(copy_f32_c4, f32, C4, nppiCopy_32f_C4R_Ctx);
impl_copy!(copy_f32_ac4, f32, AC4, nppiCopy_32f_AC4R_Ctx);
impl_copy!(copy_i16_complex_c1, ComplexI16, C1, nppiCopy_16sc_C1R_Ctx);
impl_copy!(copy_i16_complex_c2, ComplexI16, C2, nppiCopy_16sc_C2R_Ctx);
impl_copy!(copy_i16_complex_c3, ComplexI16, C3, nppiCopy_16sc_C3R_Ctx);
impl_copy!(copy_i16_complex_c4, ComplexI16, C4, nppiCopy_16sc_C4R_Ctx);
impl_copy!(
    copy_i16_complex_ac4,
    ComplexI16,
    AC4,
    nppiCopy_16sc_AC4R_Ctx
);
impl_copy!(copy_f32_complex_c1, Complex32, C1, nppiCopy_32fc_C1R_Ctx);
impl_copy!(copy_f32_complex_c2, Complex32, C2, nppiCopy_32fc_C2R_Ctx);
impl_copy!(copy_f32_complex_c3, Complex32, C3, nppiCopy_32fc_C3R_Ctx);
impl_copy!(copy_f32_complex_c4, Complex32, C4, nppiCopy_32fc_C4R_Ctx);
impl_copy!(copy_f32_complex_ac4, Complex32, AC4, nppiCopy_32fc_AC4R_Ctx);
impl_copy!(copy_i32_complex_c1, ComplexI32, C1, nppiCopy_32sc_C1R_Ctx);
impl_copy!(copy_i32_complex_c2, ComplexI32, C2, nppiCopy_32sc_C2R_Ctx);
impl_copy!(copy_i32_complex_c3, ComplexI32, C3, nppiCopy_32sc_C3R_Ctx);
impl_copy!(copy_i32_complex_c4, ComplexI32, C4, nppiCopy_32sc_C4R_Ctx);
impl_copy!(
    copy_i32_complex_ac4,
    ComplexI32,
    AC4,
    nppiCopy_32sc_AC4R_Ctx
);

impl_generic_copy_operation!(CopyC1, copy, copy_c1, C1, [
    u8 => copy_u8_c1,
    i8 => copy_i8_c1,
    u16 => copy_u16_c1,
    i16 => copy_i16_c1,
    u32 => copy_u32_c1,
    i32 => copy_i32_c1,
    f16 => copy_f16_c1,
    f32 => copy_f32_c1,
    ComplexI16 => copy_i16_complex_c1,
    Complex32 => copy_f32_complex_c1,
    ComplexI32 => copy_i32_complex_c1,
]);
impl_generic_copy_operation!(CopyC2, copy, copy_c2, C2, [
    i8 => copy_i8_c2,
    ComplexI16 => copy_i16_complex_c2,
    Complex32 => copy_f32_complex_c2,
    ComplexI32 => copy_i32_complex_c2,
]);
impl_generic_copy_operation!(CopyC3, copy, copy_c3, C3, [
    u8 => copy_u8_c3,
    i8 => copy_i8_c3,
    u16 => copy_u16_c3,
    i16 => copy_i16_c3,
    i32 => copy_i32_c3,
    f16 => copy_f16_c3,
    f32 => copy_f32_c3,
    ComplexI16 => copy_i16_complex_c3,
    Complex32 => copy_f32_complex_c3,
    ComplexI32 => copy_i32_complex_c3,
]);
impl_generic_copy_operation!(CopyC4, copy, copy_c4, C4, [
    u8 => copy_u8_c4,
    i8 => copy_i8_c4,
    u16 => copy_u16_c4,
    i16 => copy_i16_c4,
    i32 => copy_i32_c4,
    f16 => copy_f16_c4,
    f32 => copy_f32_c4,
    ComplexI16 => copy_i16_complex_c4,
    Complex32 => copy_f32_complex_c4,
    ComplexI32 => copy_i32_complex_c4,
]);
impl_generic_copy_operation!(CopyAc4, copy, copy_ac4, AC4, [
    u8 => copy_u8_ac4,
    i8 => copy_i8_ac4,
    u16 => copy_u16_ac4,
    i16 => copy_i16_ac4,
    u32 => copy_u32_ac4,
    i32 => copy_i32_ac4,
    f32 => copy_f32_ac4,
    ComplexI16 => copy_i16_complex_ac4,
    Complex32 => copy_f32_complex_ac4,
    ComplexI32 => copy_i32_complex_ac4,
]);

pub(crate) fn copy_masked_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C1>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_mask_size(source.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C1MR_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_masked_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageViewMut<'_, u8, C3>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_mask_size(source.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3MR_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_masked_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, u8, C4>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_mask_size(source.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4MR_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_masked_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageViewMut<'_, u8, AC4>,
    mask: &MaskView<'_>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_mask_size(source.size(), mask.size())?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_AC4MR_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            mask.as_ptr().cast(),
            mask.step(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_masked!(copy_masked_u16_c1, u16, C1, nppiCopy_16u_C1MR_Ctx);
impl_copy_masked!(copy_masked_u16_c3, u16, C3, nppiCopy_16u_C3MR_Ctx);
impl_copy_masked!(copy_masked_u16_c4, u16, C4, nppiCopy_16u_C4MR_Ctx);
impl_copy_masked!(copy_masked_u16_ac4, u16, AC4, nppiCopy_16u_AC4MR_Ctx);
impl_copy_masked!(copy_masked_i16_c1, i16, C1, nppiCopy_16s_C1MR_Ctx);
impl_copy_masked!(copy_masked_i16_c3, i16, C3, nppiCopy_16s_C3MR_Ctx);
impl_copy_masked!(copy_masked_i16_c4, i16, C4, nppiCopy_16s_C4MR_Ctx);
impl_copy_masked!(copy_masked_i16_ac4, i16, AC4, nppiCopy_16s_AC4MR_Ctx);
impl_copy_masked!(copy_masked_i32_c1, i32, C1, nppiCopy_32s_C1MR_Ctx);
impl_copy_masked!(copy_masked_i32_c3, i32, C3, nppiCopy_32s_C3MR_Ctx);
impl_copy_masked!(copy_masked_i32_c4, i32, C4, nppiCopy_32s_C4MR_Ctx);
impl_copy_masked!(copy_masked_i32_ac4, i32, AC4, nppiCopy_32s_AC4MR_Ctx);
impl_copy_masked!(copy_masked_f32_c1, f32, C1, nppiCopy_32f_C1MR_Ctx);
impl_copy_masked!(copy_masked_f32_c3, f32, C3, nppiCopy_32f_C3MR_Ctx);
impl_copy_masked!(copy_masked_f32_c4, f32, C4, nppiCopy_32f_C4MR_Ctx);
impl_copy_masked!(copy_masked_f32_ac4, f32, AC4, nppiCopy_32f_AC4MR_Ctx);

impl_generic_copy_masked_operation!(CopyMaskedC1, copy_masked, copy_masked_c1, C1, [
    u8 => copy_masked_u8_c1,
    u16 => copy_masked_u16_c1,
    i16 => copy_masked_i16_c1,
    i32 => copy_masked_i32_c1,
    f32 => copy_masked_f32_c1,
]);
impl_generic_copy_masked_operation!(CopyMaskedC3, copy_masked, copy_masked_c3, C3, [
    u8 => copy_masked_u8_c3,
    u16 => copy_masked_u16_c3,
    i16 => copy_masked_i16_c3,
    i32 => copy_masked_i32_c3,
    f32 => copy_masked_f32_c3,
]);
impl_generic_copy_masked_operation!(CopyMaskedC4, copy_masked, copy_masked_c4, C4, [
    u8 => copy_masked_u8_c4,
    u16 => copy_masked_u16_c4,
    i16 => copy_masked_i16_c4,
    i32 => copy_masked_i32_c4,
    f32 => copy_masked_f32_c4,
]);
impl_generic_copy_masked_operation!(CopyMaskedAc4, copy_masked, copy_masked_ac4, AC4, [
    u8 => copy_masked_u8_ac4,
    u16 => copy_masked_u16_ac4,
    i16 => copy_masked_i16_ac4,
    i32 => copy_masked_i32_ac4,
    f32 => copy_masked_f32_ac4,
]);
