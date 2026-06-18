use super::*;

macro_rules! impl_morphology_mask {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_mask(mask, mask_size, anchor)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_mask {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, mask, mask_size, anchor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, mask, mask_size, anchor)
                }
            }
        )*
    };
}

impl_morphology_mask!(dilate_u8_c1, u8, C1, nppiDilate_8u_C1R_Ctx);
impl_morphology_mask!(dilate_u8_c3, u8, C3, nppiDilate_8u_C3R_Ctx);
impl_morphology_mask!(dilate_u8_c4, u8, C4, nppiDilate_8u_C4R_Ctx);
impl_morphology_mask!(dilate_u8_ac4, u8, AC4, nppiDilate_8u_AC4R_Ctx);
impl_morphology_mask!(dilate_u16_c1, u16, C1, nppiDilate_16u_C1R_Ctx);
impl_morphology_mask!(dilate_u16_c3, u16, C3, nppiDilate_16u_C3R_Ctx);
impl_morphology_mask!(dilate_u16_c4, u16, C4, nppiDilate_16u_C4R_Ctx);
impl_morphology_mask!(dilate_u16_ac4, u16, AC4, nppiDilate_16u_AC4R_Ctx);
impl_morphology_mask!(dilate_f32_c1, f32, C1, nppiDilate_32f_C1R_Ctx);
impl_morphology_mask!(dilate_f32_c3, f32, C3, nppiDilate_32f_C3R_Ctx);
impl_morphology_mask!(dilate_f32_c4, f32, C4, nppiDilate_32f_C4R_Ctx);
impl_morphology_mask!(dilate_f32_ac4, f32, AC4, nppiDilate_32f_AC4R_Ctx);

impl_morphology_mask!(erode_u8_c1, u8, C1, nppiErode_8u_C1R_Ctx);
impl_morphology_mask!(erode_u8_c3, u8, C3, nppiErode_8u_C3R_Ctx);
impl_morphology_mask!(erode_u8_c4, u8, C4, nppiErode_8u_C4R_Ctx);
impl_morphology_mask!(erode_u8_ac4, u8, AC4, nppiErode_8u_AC4R_Ctx);
impl_morphology_mask!(erode_u16_c1, u16, C1, nppiErode_16u_C1R_Ctx);
impl_morphology_mask!(erode_u16_c3, u16, C3, nppiErode_16u_C3R_Ctx);
impl_morphology_mask!(erode_u16_c4, u16, C4, nppiErode_16u_C4R_Ctx);
impl_morphology_mask!(erode_u16_ac4, u16, AC4, nppiErode_16u_AC4R_Ctx);
impl_morphology_mask!(erode_f32_c1, f32, C1, nppiErode_32f_C1R_Ctx);
impl_morphology_mask!(erode_f32_c3, f32, C3, nppiErode_32f_C3R_Ctx);
impl_morphology_mask!(erode_f32_c4, f32, C4, nppiErode_32f_C4R_Ctx);
impl_morphology_mask!(erode_f32_ac4, f32, AC4, nppiErode_32f_AC4R_Ctx);

impl_generic_morphology_mask!(DilateC1, dilate, dilate_c1, C1, [
    u8 => dilate_u8_c1,
    u16 => dilate_u16_c1,
    f32 => dilate_f32_c1,
]);
impl_generic_morphology_mask!(DilateC3, dilate, dilate_c3, C3, [
    u8 => dilate_u8_c3,
    u16 => dilate_u16_c3,
    f32 => dilate_f32_c3,
]);
impl_generic_morphology_mask!(DilateC4, dilate, dilate_c4, C4, [
    u8 => dilate_u8_c4,
    u16 => dilate_u16_c4,
    f32 => dilate_f32_c4,
]);
impl_generic_morphology_mask!(DilateAc4, dilate, dilate_ac4, AC4, [
    u8 => dilate_u8_ac4,
    u16 => dilate_u16_ac4,
    f32 => dilate_f32_ac4,
]);
impl_generic_morphology_mask!(ErodeC1, erode, erode_c1, C1, [
    u8 => erode_u8_c1,
    u16 => erode_u16_c1,
    f32 => erode_f32_c1,
]);
impl_generic_morphology_mask!(ErodeC3, erode, erode_c3, C3, [
    u8 => erode_u8_c3,
    u16 => erode_u16_c3,
    f32 => erode_f32_c3,
]);
impl_generic_morphology_mask!(ErodeC4, erode, erode_c4, C4, [
    u8 => erode_u8_c4,
    u16 => erode_u16_c4,
    f32 => erode_f32_c4,
]);
impl_generic_morphology_mask!(ErodeAc4, erode, erode_ac4, AC4, [
    u8 => erode_u8_ac4,
    u16 => erode_u16_ac4,
    f32 => erode_f32_ac4,
]);
