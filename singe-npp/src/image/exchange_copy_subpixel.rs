use super::*;

pub(crate) fn copy_subpixel_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C1>,
    x_shift: f32,
    y_shift: f32,
) -> Result<()> {
    validate_subpixel_copy(source.size(), destination.size(), x_shift, y_shift)?;

    unsafe {
        try_ffi!(sys::nppiCopySubpix_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            x_shift,
            y_shift,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

macro_rules! impl_copy_subpixel {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            x_shift: f32,
            y_shift: f32,
        ) -> Result<()> {
            validate_subpixel_copy(source.size(), destination.size(), x_shift, y_shift)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    x_shift,
                    y_shift,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_subpixel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                x_shift: f32,
                y_shift: f32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    x_shift: f32,
                    y_shift: f32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, x_shift, y_shift)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            x_shift: f32,
            y_shift: f32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, x_shift, y_shift)
        }
    };
}

impl_copy_subpixel!(copy_subpixel_u8_c3, u8, C3, nppiCopySubpix_8u_C3R_Ctx);
impl_copy_subpixel!(copy_subpixel_u8_c4, u8, C4, nppiCopySubpix_8u_C4R_Ctx);
impl_copy_subpixel!(copy_subpixel_u8_ac4, u8, AC4, nppiCopySubpix_8u_AC4R_Ctx);
impl_copy_subpixel!(copy_subpixel_u16_c1, u16, C1, nppiCopySubpix_16u_C1R_Ctx);
impl_copy_subpixel!(copy_subpixel_u16_c3, u16, C3, nppiCopySubpix_16u_C3R_Ctx);
impl_copy_subpixel!(copy_subpixel_u16_c4, u16, C4, nppiCopySubpix_16u_C4R_Ctx);
impl_copy_subpixel!(copy_subpixel_u16_ac4, u16, AC4, nppiCopySubpix_16u_AC4R_Ctx);
impl_copy_subpixel!(copy_subpixel_i16_c1, i16, C1, nppiCopySubpix_16s_C1R_Ctx);
impl_copy_subpixel!(copy_subpixel_i16_c3, i16, C3, nppiCopySubpix_16s_C3R_Ctx);
impl_copy_subpixel!(copy_subpixel_i16_c4, i16, C4, nppiCopySubpix_16s_C4R_Ctx);
impl_copy_subpixel!(copy_subpixel_i16_ac4, i16, AC4, nppiCopySubpix_16s_AC4R_Ctx);
impl_copy_subpixel!(copy_subpixel_i32_c1, i32, C1, nppiCopySubpix_32s_C1R_Ctx);
impl_copy_subpixel!(copy_subpixel_i32_c3, i32, C3, nppiCopySubpix_32s_C3R_Ctx);
impl_copy_subpixel!(copy_subpixel_i32_c4, i32, C4, nppiCopySubpix_32s_C4R_Ctx);
impl_copy_subpixel!(copy_subpixel_i32_ac4, i32, AC4, nppiCopySubpix_32s_AC4R_Ctx);
impl_copy_subpixel!(copy_subpixel_f32_c1, f32, C1, nppiCopySubpix_32f_C1R_Ctx);
impl_copy_subpixel!(copy_subpixel_f32_c3, f32, C3, nppiCopySubpix_32f_C3R_Ctx);
impl_copy_subpixel!(copy_subpixel_f32_c4, f32, C4, nppiCopySubpix_32f_C4R_Ctx);
impl_copy_subpixel!(copy_subpixel_f32_ac4, f32, AC4, nppiCopySubpix_32f_AC4R_Ctx);
impl_generic_copy_subpixel_operation!(
    CopySubpixelC1,
    copy_subpixel_c1,
    C1,
    [
        u8 => copy_subpixel_u8_c1,
        u16 => copy_subpixel_u16_c1,
        i16 => copy_subpixel_i16_c1,
        i32 => copy_subpixel_i32_c1,
        f32 => copy_subpixel_f32_c1,
    ]
);
impl_generic_copy_subpixel_operation!(
    CopySubpixelC3,
    copy_subpixel_c3,
    C3,
    [
        u8 => copy_subpixel_u8_c3,
        u16 => copy_subpixel_u16_c3,
        i16 => copy_subpixel_i16_c3,
        i32 => copy_subpixel_i32_c3,
        f32 => copy_subpixel_f32_c3,
    ]
);
impl_generic_copy_subpixel_operation!(
    CopySubpixelC4,
    copy_subpixel_c4,
    C4,
    [
        u8 => copy_subpixel_u8_c4,
        u16 => copy_subpixel_u16_c4,
        i16 => copy_subpixel_i16_c4,
        i32 => copy_subpixel_i32_c4,
        f32 => copy_subpixel_f32_c4,
    ]
);
impl_generic_copy_subpixel_operation!(
    CopySubpixelAC4,
    copy_subpixel_ac4,
    AC4,
    [
        u8 => copy_subpixel_u8_ac4,
        u16 => copy_subpixel_u16_ac4,
        i16 => copy_subpixel_i16_ac4,
        i32 => copy_subpixel_i32_ac4,
        f32 => copy_subpixel_f32_ac4,
    ]
);
