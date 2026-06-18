use super::*;

pub(crate) fn extract_channel_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    destination: &mut ImageViewMut<'_, u8, C1>,
    channel: usize,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3C1R_Ctx(
            source.as_ptr().add(channel),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn extract_channel_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    destination: &mut ImageViewMut<'_, u8, C1>,
    channel: usize,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4C1R_Ctx(
            source.as_ptr().add(channel),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_extract_channel!(extract_channel_u16_c3, u16, C3, 3, nppiCopy_16u_C3C1R_Ctx);
impl_extract_channel!(extract_channel_u16_c4, u16, C4, 4, nppiCopy_16u_C4C1R_Ctx);
impl_extract_channel!(extract_channel_i16_c3, i16, C3, 3, nppiCopy_16s_C3C1R_Ctx);
impl_extract_channel!(extract_channel_i16_c4, i16, C4, 4, nppiCopy_16s_C4C1R_Ctx);
impl_extract_channel!(extract_channel_i32_c3, i32, C3, 3, nppiCopy_32s_C3C1R_Ctx);
impl_extract_channel!(extract_channel_i32_c4, i32, C4, 4, nppiCopy_32s_C4C1R_Ctx);
impl_extract_channel!(extract_channel_f32_c2, f32, C2, 2, nppiCopy_32f_C2C1R_Ctx);
impl_extract_channel!(extract_channel_f32_c3, f32, C3, 3, nppiCopy_32f_C3C1R_Ctx);
impl_extract_channel!(extract_channel_f32_c4, f32, C4, 4, nppiCopy_32f_C4C1R_Ctx);
impl_generic_extract_channel_operation!(
    ExtractChannelC2,
    extract_channel_c2,
    C2,
    [f32 => extract_channel_f32_c2]
);
impl_generic_extract_channel_operation!(
    ExtractChannelC3,
    extract_channel_c3,
    C3,
    [
        u8 => extract_channel_u8_c3,
        u16 => extract_channel_u16_c3,
        i16 => extract_channel_i16_c3,
        i32 => extract_channel_i32_c3,
        f32 => extract_channel_f32_c3,
    ]
);
impl_generic_extract_channel_operation!(
    ExtractChannelC4,
    extract_channel_c4,
    C4,
    [
        u8 => extract_channel_u8_c4,
        u16 => extract_channel_u16_c4,
        i16 => extract_channel_i16_c4,
        i32 => extract_channel_i32_c4,
        f32 => extract_channel_f32_c4,
    ]
);

pub(crate) fn set_channel_u8_c3(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, u8, C3>,
    value: u8,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiSet_8u_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_u8_c4(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, u8, C4>,
    value: u8,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiSet_8u_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_u16_c3(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, u16, C3>,
    value: u16,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiSet_16u_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_u16_c4(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, u16, C4>,
    value: u16,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiSet_16u_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_i16_c3(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, i16, C3>,
    value: i16,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiSet_16s_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_i16_c4(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, i16, C4>,
    value: i16,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiSet_16s_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_i32_c3(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, i32, C3>,
    value: i32,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiSet_32s_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_i32_c4(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, i32, C4>,
    value: i32,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiSet_32s_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_f32_c3(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, f32, C3>,
    value: f32,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiSet_32f_C3CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn set_channel_f32_c4(
    stream_context: &StreamContext,
    destination: &mut ImageViewMut<'_, f32, C4>,
    value: f32,
    channel: usize,
) -> Result<()> {
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiSet_32f_C4CR_Ctx(
            value,
            destination.as_mut_ptr().add(channel),
            destination.step(),
            destination.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}
impl_generic_set_channel_operation!(
    SetChannelC3,
    set_channel_c3,
    C3,
    [
        u8 => set_channel_u8_c3,
        u16 => set_channel_u16_c3,
        i16 => set_channel_i16_c3,
        i32 => set_channel_i32_c3,
        f32 => set_channel_f32_c3,
    ]
);
impl_generic_set_channel_operation!(
    SetChannelC4,
    set_channel_c4,
    C4,
    [
        u8 => set_channel_u8_c4,
        u16 => set_channel_u16_c4,
        i16 => set_channel_i16_c4,
        i32 => set_channel_i32_c4,
        f32 => set_channel_f32_c4,
    ]
);

pub(crate) fn insert_channel_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C3>,
    channel: usize,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C1C3R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().add(channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn insert_channel_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C4>,
    channel: usize,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C1C4R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().add(channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_insert_channel!(insert_channel_u16_c3, u16, C3, 3, nppiCopy_16u_C1C3R_Ctx);
impl_insert_channel!(insert_channel_u16_c4, u16, C4, 4, nppiCopy_16u_C1C4R_Ctx);
impl_insert_channel!(insert_channel_i16_c3, i16, C3, 3, nppiCopy_16s_C1C3R_Ctx);
impl_insert_channel!(insert_channel_i16_c4, i16, C4, 4, nppiCopy_16s_C1C4R_Ctx);
impl_insert_channel!(insert_channel_i32_c3, i32, C3, 3, nppiCopy_32s_C1C3R_Ctx);
impl_insert_channel!(insert_channel_i32_c4, i32, C4, 4, nppiCopy_32s_C1C4R_Ctx);
impl_insert_channel!(insert_channel_f32_c2, f32, C2, 2, nppiCopy_32f_C1C2R_Ctx);
impl_insert_channel!(insert_channel_f32_c3, f32, C3, 3, nppiCopy_32f_C1C3R_Ctx);
impl_insert_channel!(insert_channel_f32_c4, f32, C4, 4, nppiCopy_32f_C1C4R_Ctx);
impl_generic_insert_channel_operation!(
    InsertChannelC2,
    insert_channel_c2,
    C2,
    [f32 => insert_channel_f32_c2]
);
impl_generic_insert_channel_operation!(
    InsertChannelC3,
    insert_channel_c3,
    C3,
    [
        u8 => insert_channel_u8_c3,
        u16 => insert_channel_u16_c3,
        i16 => insert_channel_i16_c3,
        i32 => insert_channel_i32_c3,
        f32 => insert_channel_f32_c3,
    ]
);
impl_generic_insert_channel_operation!(
    InsertChannelC4,
    insert_channel_c4,
    C4,
    [
        u8 => insert_channel_u8_c4,
        u16 => insert_channel_u16_c4,
        i16 => insert_channel_i16_c4,
        i32 => insert_channel_i32_c4,
        f32 => insert_channel_f32_c4,
    ]
);

pub(crate) fn duplicate_channel_u8_c1_to_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C3>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiDup_8u_C1C3R_Ctx(
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

pub(crate) fn duplicate_channel_u8_c1_to_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiDup_8u_C1C4R_Ctx(
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

pub(crate) fn duplicate_channel_u8_c1_to_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, AC4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiDup_8u_C1AC4R_Ctx(
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

pub(crate) fn duplicate_channel_u16_c1_to_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageViewMut<'_, u16, C3>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiDup_16u_C1C3R_Ctx(
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

pub(crate) fn duplicate_channel_u16_c1_to_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageViewMut<'_, u16, C4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiDup_16u_C1C4R_Ctx(
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

pub(crate) fn duplicate_channel_u16_c1_to_ac4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageViewMut<'_, u16, AC4>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

    unsafe {
        try_ffi!(sys::nppiDup_16u_C1AC4R_Ctx(
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

macro_rules! impl_duplicate_channel {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
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
    };
}

macro_rules! impl_generic_duplicate_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination)
        }
    };
}

impl_duplicate_channel!(
    duplicate_channel_i16_c1_to_c3,
    i16,
    C3,
    nppiDup_16s_C1C3R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_i16_c1_to_c4,
    i16,
    C4,
    nppiDup_16s_C1C4R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_i16_c1_to_ac4,
    i16,
    AC4,
    nppiDup_16s_C1AC4R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_i32_c1_to_c3,
    i32,
    C3,
    nppiDup_32s_C1C3R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_i32_c1_to_c4,
    i32,
    C4,
    nppiDup_32s_C1C4R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_i32_c1_to_ac4,
    i32,
    AC4,
    nppiDup_32s_C1AC4R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_f32_c1_to_c3,
    f32,
    C3,
    nppiDup_32f_C1C3R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_f32_c1_to_c4,
    f32,
    C4,
    nppiDup_32f_C1C4R_Ctx
);
impl_duplicate_channel!(
    duplicate_channel_f32_c1_to_ac4,
    f32,
    AC4,
    nppiDup_32f_C1AC4R_Ctx
);
impl_generic_duplicate_channel_operation!(
    DuplicateChannelC1ToC3,
    duplicate_channel_c1_to_c3,
    C3,
    [
        u8 => duplicate_channel_u8_c1_to_c3,
        u16 => duplicate_channel_u16_c1_to_c3,
        i16 => duplicate_channel_i16_c1_to_c3,
        i32 => duplicate_channel_i32_c1_to_c3,
        f32 => duplicate_channel_f32_c1_to_c3,
    ]
);
impl_generic_duplicate_channel_operation!(
    DuplicateChannelC1ToC4,
    duplicate_channel_c1_to_c4,
    C4,
    [
        u8 => duplicate_channel_u8_c1_to_c4,
        u16 => duplicate_channel_u16_c1_to_c4,
        i16 => duplicate_channel_i16_c1_to_c4,
        i32 => duplicate_channel_i32_c1_to_c4,
        f32 => duplicate_channel_f32_c1_to_c4,
    ]
);
impl_generic_duplicate_channel_operation!(
    DuplicateChannelC1ToAC4,
    duplicate_channel_c1_to_ac4,
    AC4,
    [
        u8 => duplicate_channel_u8_c1_to_ac4,
        u16 => duplicate_channel_u16_c1_to_ac4,
        i16 => duplicate_channel_i16_c1_to_ac4,
        i32 => duplicate_channel_i32_c1_to_ac4,
        f32 => duplicate_channel_f32_c1_to_ac4,
    ]
);

pub(crate) fn copy_channel_u8_c3(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_channel: usize,
    destination: &mut ImageViewMut<'_, u8, C3>,
    destination_channel: usize,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(source_channel, 3)?;
    validate_channel_index(destination_channel, 3)?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C3CR_Ctx(
            source.as_ptr().add(source_channel),
            source.step(),
            destination.as_mut_ptr().add(destination_channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn copy_channel_u8_c4(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C4>,
    source_channel: usize,
    destination: &mut ImageViewMut<'_, u8, C4>,
    destination_channel: usize,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;
    validate_channel_index(source_channel, 4)?;
    validate_channel_index(destination_channel, 4)?;

    unsafe {
        try_ffi!(sys::nppiCopy_8u_C4CR_Ctx(
            source.as_ptr().add(source_channel),
            source.step(),
            destination.as_mut_ptr().add(destination_channel),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

impl_copy_channel!(copy_channel_u16_c3, u16, C3, 3, nppiCopy_16u_C3CR_Ctx);
impl_copy_channel!(copy_channel_u16_c4, u16, C4, 4, nppiCopy_16u_C4CR_Ctx);
impl_copy_channel!(copy_channel_i16_c3, i16, C3, 3, nppiCopy_16s_C3CR_Ctx);
impl_copy_channel!(copy_channel_i16_c4, i16, C4, 4, nppiCopy_16s_C4CR_Ctx);
impl_copy_channel!(copy_channel_i32_c3, i32, C3, 3, nppiCopy_32s_C3CR_Ctx);
impl_copy_channel!(copy_channel_i32_c4, i32, C4, 4, nppiCopy_32s_C4CR_Ctx);
impl_copy_channel!(copy_channel_f32_c3, f32, C3, 3, nppiCopy_32f_C3CR_Ctx);
impl_copy_channel!(copy_channel_f32_c4, f32, C4, 4, nppiCopy_32f_C4CR_Ctx);
impl_generic_copy_channel_operation!(
    CopyChannelC3,
    copy_channel_c3,
    C3,
    [
        u8 => copy_channel_u8_c3,
        u16 => copy_channel_u16_c3,
        i16 => copy_channel_i16_c3,
        i32 => copy_channel_i32_c3,
        f32 => copy_channel_f32_c3,
    ]
);
impl_generic_copy_channel_operation!(
    CopyChannelC4,
    copy_channel_c4,
    C4,
    [
        u8 => copy_channel_u8_c4,
        u16 => copy_channel_u16_c4,
        i16 => copy_channel_i16_c4,
        i32 => copy_channel_i32_c4,
        f32 => copy_channel_f32_c4,
    ]
);
