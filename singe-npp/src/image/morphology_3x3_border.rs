use super::*;

macro_rules! impl_morphology_3x3_border {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_3x3_border {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, $layout>,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source,
                source_offset,
                destination,
                border_type,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                }
            }
        )*
    };
}

impl_morphology_3x3_border!(
    dilate_3x3_border_u8_c1,
    u8,
    C1,
    nppiDilate3x3Border_8u_C1R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u8_c3,
    u8,
    C3,
    nppiDilate3x3Border_8u_C3R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u8_c4,
    u8,
    C4,
    nppiDilate3x3Border_8u_C4R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u8_ac4,
    u8,
    AC4,
    nppiDilate3x3Border_8u_AC4R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u16_c1,
    u16,
    C1,
    nppiDilate3x3Border_16u_C1R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u16_c3,
    u16,
    C3,
    nppiDilate3x3Border_16u_C3R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u16_c4,
    u16,
    C4,
    nppiDilate3x3Border_16u_C4R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_u16_ac4,
    u16,
    AC4,
    nppiDilate3x3Border_16u_AC4R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_f32_c1,
    f32,
    C1,
    nppiDilate3x3Border_32f_C1R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_f32_c3,
    f32,
    C3,
    nppiDilate3x3Border_32f_C3R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_f32_c4,
    f32,
    C4,
    nppiDilate3x3Border_32f_C4R_Ctx
);
impl_morphology_3x3_border!(
    dilate_3x3_border_f32_ac4,
    f32,
    AC4,
    nppiDilate3x3Border_32f_AC4R_Ctx
);

impl_morphology_3x3_border!(
    erode_3x3_border_u8_c1,
    u8,
    C1,
    nppiErode3x3Border_8u_C1R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u8_c3,
    u8,
    C3,
    nppiErode3x3Border_8u_C3R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u8_c4,
    u8,
    C4,
    nppiErode3x3Border_8u_C4R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u8_ac4,
    u8,
    AC4,
    nppiErode3x3Border_8u_AC4R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u16_c1,
    u16,
    C1,
    nppiErode3x3Border_16u_C1R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u16_c3,
    u16,
    C3,
    nppiErode3x3Border_16u_C3R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u16_c4,
    u16,
    C4,
    nppiErode3x3Border_16u_C4R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_u16_ac4,
    u16,
    AC4,
    nppiErode3x3Border_16u_AC4R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_f32_c1,
    f32,
    C1,
    nppiErode3x3Border_32f_C1R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_f32_c3,
    f32,
    C3,
    nppiErode3x3Border_32f_C3R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_f32_c4,
    f32,
    C4,
    nppiErode3x3Border_32f_C4R_Ctx
);
impl_morphology_3x3_border!(
    erode_3x3_border_f32_ac4,
    f32,
    AC4,
    nppiErode3x3Border_32f_AC4R_Ctx
);

impl_generic_morphology_3x3_border!(
    Dilate3x3BorderC1,
    dilate_3x3_border,
    dilate_3x3_border_c1,
    C1,
    [
        u8 => dilate_3x3_border_u8_c1,
        u16 => dilate_3x3_border_u16_c1,
        f32 => dilate_3x3_border_f32_c1,
    ]
);
impl_generic_morphology_3x3_border!(
    Dilate3x3BorderC3,
    dilate_3x3_border,
    dilate_3x3_border_c3,
    C3,
    [
        u8 => dilate_3x3_border_u8_c3,
        u16 => dilate_3x3_border_u16_c3,
        f32 => dilate_3x3_border_f32_c3,
    ]
);
impl_generic_morphology_3x3_border!(
    Dilate3x3BorderC4,
    dilate_3x3_border,
    dilate_3x3_border_c4,
    C4,
    [
        u8 => dilate_3x3_border_u8_c4,
        u16 => dilate_3x3_border_u16_c4,
        f32 => dilate_3x3_border_f32_c4,
    ]
);
impl_generic_morphology_3x3_border!(
    Dilate3x3BorderAc4,
    dilate_3x3_border,
    dilate_3x3_border_ac4,
    AC4,
    [
        u8 => dilate_3x3_border_u8_ac4,
        u16 => dilate_3x3_border_u16_ac4,
        f32 => dilate_3x3_border_f32_ac4,
    ]
);
impl_generic_morphology_3x3_border!(
    Erode3x3BorderC1,
    erode_3x3_border,
    erode_3x3_border_c1,
    C1,
    [
        u8 => erode_3x3_border_u8_c1,
        u16 => erode_3x3_border_u16_c1,
        f32 => erode_3x3_border_f32_c1,
    ]
);
impl_generic_morphology_3x3_border!(
    Erode3x3BorderC3,
    erode_3x3_border,
    erode_3x3_border_c3,
    C3,
    [
        u8 => erode_3x3_border_u8_c3,
        u16 => erode_3x3_border_u16_c3,
        f32 => erode_3x3_border_f32_c3,
    ]
);
impl_generic_morphology_3x3_border!(
    Erode3x3BorderC4,
    erode_3x3_border,
    erode_3x3_border_c4,
    C4,
    [
        u8 => erode_3x3_border_u8_c4,
        u16 => erode_3x3_border_u16_c4,
        f32 => erode_3x3_border_f32_c4,
    ]
);
impl_generic_morphology_3x3_border!(
    Erode3x3BorderAc4,
    erode_3x3_border,
    erode_3x3_border_ac4,
    AC4,
    [
        u8 => erode_3x3_border_u8_ac4,
        u16 => erode_3x3_border_u16_ac4,
        f32 => erode_3x3_border_f32_ac4,
    ]
);
