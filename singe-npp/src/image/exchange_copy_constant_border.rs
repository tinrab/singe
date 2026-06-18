use super::*;

pub(crate) fn copy_constant_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C1>,
    top: usize,
    left: usize,
    value: u8,
) -> Result<()> {
    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    unsafe {
        try_ffi!(sys::nppiCopyConstBorder_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            value,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_constant_border_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageViewMut<'_, u8, C3>,
    top: usize,
    left: usize,
    value: [u8; 3],
) -> Result<()> {
    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    unsafe {
        try_ffi!(sys::nppiCopyConstBorder_8u_C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            value.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_constant_border_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, u8, C4>,
    top: usize,
    left: usize,
    value: [u8; 4],
) -> Result<()> {
    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    unsafe {
        try_ffi!(sys::nppiCopyConstBorder_8u_C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            value.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_constant_border_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageViewMut<'_, u8, AC4>,
    top: usize,
    left: usize,
    value: [u8; 3],
) -> Result<()> {
    let (top, left) = validate_border_placement(source.size(), destination.size(), top, left)?;

    unsafe {
        try_ffi!(sys::nppiCopyConstBorder_8u_AC4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            top,
            left,
            value.as_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_constant_border_scalar!(
    copy_constant_border_u16_c1,
    u16,
    nppiCopyConstBorder_16u_C1R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_u16_c3,
    u16,
    C3,
    [u16; 3],
    nppiCopyConstBorder_16u_C3R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_u16_c4,
    u16,
    C4,
    [u16; 4],
    nppiCopyConstBorder_16u_C4R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_u16_ac4,
    u16,
    AC4,
    [u16; 3],
    nppiCopyConstBorder_16u_AC4R_Ctx
);
impl_copy_constant_border_scalar!(
    copy_constant_border_i16_c1,
    i16,
    nppiCopyConstBorder_16s_C1R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_i16_c3,
    i16,
    C3,
    [i16; 3],
    nppiCopyConstBorder_16s_C3R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_i16_c4,
    i16,
    C4,
    [i16; 4],
    nppiCopyConstBorder_16s_C4R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_i16_ac4,
    i16,
    AC4,
    [i16; 3],
    nppiCopyConstBorder_16s_AC4R_Ctx
);
impl_copy_constant_border_scalar!(
    copy_constant_border_i32_c1,
    i32,
    nppiCopyConstBorder_32s_C1R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_i32_c3,
    i32,
    C3,
    [i32; 3],
    nppiCopyConstBorder_32s_C3R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_i32_c4,
    i32,
    C4,
    [i32; 4],
    nppiCopyConstBorder_32s_C4R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_i32_ac4,
    i32,
    AC4,
    [i32; 3],
    nppiCopyConstBorder_32s_AC4R_Ctx
);
impl_copy_constant_border_scalar!(
    copy_constant_border_f32_c1,
    f32,
    nppiCopyConstBorder_32f_C1R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_f32_c3,
    f32,
    C3,
    [f32; 3],
    nppiCopyConstBorder_32f_C3R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_f32_c4,
    f32,
    C4,
    [f32; 4],
    nppiCopyConstBorder_32f_C4R_Ctx
);
impl_copy_constant_border_packed!(
    copy_constant_border_f32_ac4,
    f32,
    AC4,
    [f32; 3],
    nppiCopyConstBorder_32f_AC4R_Ctx
);
impl_generic_copy_constant_border_operation!(
    CopyConstantBorderC1,
    copy_constant_border_c1,
    C1,
    [
        u8 => (u8, copy_constant_border_u8_c1),
        u16 => (u16, copy_constant_border_u16_c1),
        i16 => (i16, copy_constant_border_i16_c1),
        i32 => (i32, copy_constant_border_i32_c1),
        f32 => (f32, copy_constant_border_f32_c1),
    ]
);
impl_generic_copy_constant_border_operation!(
    CopyConstantBorderC3,
    copy_constant_border_c3,
    C3,
    [
        u8 => ([u8; 3], copy_constant_border_u8_c3),
        u16 => ([u16; 3], copy_constant_border_u16_c3),
        i16 => ([i16; 3], copy_constant_border_i16_c3),
        i32 => ([i32; 3], copy_constant_border_i32_c3),
        f32 => ([f32; 3], copy_constant_border_f32_c3),
    ]
);
impl_generic_copy_constant_border_operation!(
    CopyConstantBorderC4,
    copy_constant_border_c4,
    C4,
    [
        u8 => ([u8; 4], copy_constant_border_u8_c4),
        u16 => ([u16; 4], copy_constant_border_u16_c4),
        i16 => ([i16; 4], copy_constant_border_i16_c4),
        i32 => ([i32; 4], copy_constant_border_i32_c4),
        f32 => ([f32; 4], copy_constant_border_f32_c4),
    ]
);
impl_generic_copy_constant_border_operation!(
    CopyConstantBorderAC4,
    copy_constant_border_ac4,
    AC4,
    [
        u8 => ([u8; 3], copy_constant_border_u8_ac4),
        u16 => ([u16; 3], copy_constant_border_u16_ac4),
        i16 => ([i16; 3], copy_constant_border_i16_ac4),
        i32 => ([i32; 3], copy_constant_border_i32_ac4),
        f32 => ([f32; 3], copy_constant_border_f32_ac4),
    ]
);
