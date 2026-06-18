use super::*;

macro_rules! impl_morphology_3x3 {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
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
    };
}

macro_rules! impl_generic_morphology_3x3 {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

impl_morphology_3x3!(dilate_3x3_u8_c1, u8, C1, nppiDilate3x3_8u_C1R_Ctx);
impl_morphology_3x3!(dilate_3x3_u8_c3, u8, C3, nppiDilate3x3_8u_C3R_Ctx);
impl_morphology_3x3!(dilate_3x3_u8_c4, u8, C4, nppiDilate3x3_8u_C4R_Ctx);
impl_morphology_3x3!(dilate_3x3_u8_ac4, u8, AC4, nppiDilate3x3_8u_AC4R_Ctx);
impl_morphology_3x3!(dilate_3x3_u16_c1, u16, C1, nppiDilate3x3_16u_C1R_Ctx);
impl_morphology_3x3!(dilate_3x3_u16_c3, u16, C3, nppiDilate3x3_16u_C3R_Ctx);
impl_morphology_3x3!(dilate_3x3_u16_c4, u16, C4, nppiDilate3x3_16u_C4R_Ctx);
impl_morphology_3x3!(dilate_3x3_u16_ac4, u16, AC4, nppiDilate3x3_16u_AC4R_Ctx);
impl_morphology_3x3!(dilate_3x3_f32_c1, f32, C1, nppiDilate3x3_32f_C1R_Ctx);
impl_morphology_3x3!(dilate_3x3_f32_c3, f32, C3, nppiDilate3x3_32f_C3R_Ctx);
impl_morphology_3x3!(dilate_3x3_f32_c4, f32, C4, nppiDilate3x3_32f_C4R_Ctx);
impl_morphology_3x3!(dilate_3x3_f32_ac4, f32, AC4, nppiDilate3x3_32f_AC4R_Ctx);
impl_morphology_3x3!(dilate_3x3_f64_c1, f64, C1, nppiDilate3x3_64f_C1R_Ctx);

impl_morphology_3x3!(erode_3x3_u8_c1, u8, C1, nppiErode3x3_8u_C1R_Ctx);
impl_morphology_3x3!(erode_3x3_u8_c3, u8, C3, nppiErode3x3_8u_C3R_Ctx);
impl_morphology_3x3!(erode_3x3_u8_c4, u8, C4, nppiErode3x3_8u_C4R_Ctx);
impl_morphology_3x3!(erode_3x3_u8_ac4, u8, AC4, nppiErode3x3_8u_AC4R_Ctx);
impl_morphology_3x3!(erode_3x3_u16_c1, u16, C1, nppiErode3x3_16u_C1R_Ctx);
impl_morphology_3x3!(erode_3x3_u16_c3, u16, C3, nppiErode3x3_16u_C3R_Ctx);
impl_morphology_3x3!(erode_3x3_u16_c4, u16, C4, nppiErode3x3_16u_C4R_Ctx);
impl_morphology_3x3!(erode_3x3_u16_ac4, u16, AC4, nppiErode3x3_16u_AC4R_Ctx);
impl_morphology_3x3!(erode_3x3_f32_c1, f32, C1, nppiErode3x3_32f_C1R_Ctx);
impl_morphology_3x3!(erode_3x3_f32_c3, f32, C3, nppiErode3x3_32f_C3R_Ctx);
impl_morphology_3x3!(erode_3x3_f32_c4, f32, C4, nppiErode3x3_32f_C4R_Ctx);
impl_morphology_3x3!(erode_3x3_f32_ac4, f32, AC4, nppiErode3x3_32f_AC4R_Ctx);
impl_morphology_3x3!(erode_3x3_f64_c1, f64, C1, nppiErode3x3_64f_C1R_Ctx);

impl_generic_morphology_3x3!(Dilate3x3C1, dilate_3x3, dilate_3x3_c1, C1, [
    u8 => dilate_3x3_u8_c1,
    u16 => dilate_3x3_u16_c1,
    f32 => dilate_3x3_f32_c1,
    f64 => dilate_3x3_f64_c1,
]);
impl_generic_morphology_3x3!(Dilate3x3C3, dilate_3x3, dilate_3x3_c3, C3, [
    u8 => dilate_3x3_u8_c3,
    u16 => dilate_3x3_u16_c3,
    f32 => dilate_3x3_f32_c3,
]);
impl_generic_morphology_3x3!(Dilate3x3C4, dilate_3x3, dilate_3x3_c4, C4, [
    u8 => dilate_3x3_u8_c4,
    u16 => dilate_3x3_u16_c4,
    f32 => dilate_3x3_f32_c4,
]);
impl_generic_morphology_3x3!(Dilate3x3Ac4, dilate_3x3, dilate_3x3_ac4, AC4, [
    u8 => dilate_3x3_u8_ac4,
    u16 => dilate_3x3_u16_ac4,
    f32 => dilate_3x3_f32_ac4,
]);
impl_generic_morphology_3x3!(Erode3x3C1, erode_3x3, erode_3x3_c1, C1, [
    u8 => erode_3x3_u8_c1,
    u16 => erode_3x3_u16_c1,
    f32 => erode_3x3_f32_c1,
    f64 => erode_3x3_f64_c1,
]);
impl_generic_morphology_3x3!(Erode3x3C3, erode_3x3, erode_3x3_c3, C3, [
    u8 => erode_3x3_u8_c3,
    u16 => erode_3x3_u16_c3,
    f32 => erode_3x3_f32_c3,
]);
impl_generic_morphology_3x3!(Erode3x3C4, erode_3x3, erode_3x3_c4, C4, [
    u8 => erode_3x3_u8_c4,
    u16 => erode_3x3_u16_c4,
    f32 => erode_3x3_f32_c4,
]);
impl_generic_morphology_3x3!(Erode3x3Ac4, erode_3x3, erode_3x3_ac4, AC4, [
    u8 => erode_3x3_u8_ac4,
    u16 => erode_3x3_u16_ac4,
    f32 => erode_3x3_f32_ac4,
]);
