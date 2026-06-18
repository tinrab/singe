use super::*;

pub(crate) fn lookup_table_trilinear_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, u8, C4>,
    values: &DeviceMemory<u32>,
    levels: [&[u8]; 3],
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    let (mut level_pointers, mut level_counts) =
        lookup_table_trilinear_descriptors(values.len(), levels)?;

    unsafe {
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
        ))?;
    }
    Ok(())
}

pub(crate) fn lookup_table_trilinear_u8_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, AC4>,
    destination: &mut ImageViewMut<'_, u8, AC4>,
    values: &DeviceMemory<u32>,
    levels: [&[u8]; 3],
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    let (mut level_pointers, mut level_counts) =
        lookup_table_trilinear_descriptors(values.len(), levels)?;

    unsafe {
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
        ))?;
    }
    Ok(())
}

pub(crate) fn lookup_table_trilinear_u8_ac4_in_place(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, u8, AC4>,
    values: &DeviceMemory<u32>,
    levels: [&[u8]; 3],
) -> Result<()> {
    let (mut level_pointers, mut level_counts) =
        lookup_table_trilinear_descriptors(values.len(), levels)?;

    unsafe {
        try_ffi!(sys::nppiLUT_Trilinear_8u_AC4IR_Ctx(
            image.as_mut_ptr().cast(),
            image.step(),
            image.size().into(),
            values.as_ptr().cast_mut(),
            level_pointers.as_mut_ptr().cast(),
            level_counts.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
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
