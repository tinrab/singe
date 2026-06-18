use super::*;

macro_rules! impl_morphology_mask_border {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_mask(mask, mask_size, anchor)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_mask_border {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
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
                mask,
                mask_size,
                anchor,
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
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                }
            }
        )*
    };
}

impl_morphology_mask_border!(dilate_border_u8_c1, u8, C1, nppiDilateBorder_8u_C1R_Ctx);
impl_morphology_mask_border!(dilate_border_u8_c3, u8, C3, nppiDilateBorder_8u_C3R_Ctx);
impl_morphology_mask_border!(dilate_border_u8_c4, u8, C4, nppiDilateBorder_8u_C4R_Ctx);
impl_morphology_mask_border!(dilate_border_u8_ac4, u8, AC4, nppiDilateBorder_8u_AC4R_Ctx);
impl_morphology_mask_border!(dilate_border_u16_c1, u16, C1, nppiDilateBorder_16u_C1R_Ctx);
impl_morphology_mask_border!(dilate_border_u16_c3, u16, C3, nppiDilateBorder_16u_C3R_Ctx);
impl_morphology_mask_border!(dilate_border_u16_c4, u16, C4, nppiDilateBorder_16u_C4R_Ctx);
impl_morphology_mask_border!(
    dilate_border_u16_ac4,
    u16,
    AC4,
    nppiDilateBorder_16u_AC4R_Ctx
);
impl_morphology_mask_border!(dilate_border_f32_c1, f32, C1, nppiDilateBorder_32f_C1R_Ctx);
impl_morphology_mask_border!(dilate_border_f32_c3, f32, C3, nppiDilateBorder_32f_C3R_Ctx);
impl_morphology_mask_border!(dilate_border_f32_c4, f32, C4, nppiDilateBorder_32f_C4R_Ctx);
impl_morphology_mask_border!(
    dilate_border_f32_ac4,
    f32,
    AC4,
    nppiDilateBorder_32f_AC4R_Ctx
);

impl_morphology_mask_border!(erode_border_u8_c1, u8, C1, nppiErodeBorder_8u_C1R_Ctx);
impl_morphology_mask_border!(erode_border_u8_c3, u8, C3, nppiErodeBorder_8u_C3R_Ctx);
impl_morphology_mask_border!(erode_border_u8_c4, u8, C4, nppiErodeBorder_8u_C4R_Ctx);
impl_morphology_mask_border!(erode_border_u8_ac4, u8, AC4, nppiErodeBorder_8u_AC4R_Ctx);
impl_morphology_mask_border!(erode_border_u16_c1, u16, C1, nppiErodeBorder_16u_C1R_Ctx);
impl_morphology_mask_border!(erode_border_u16_c3, u16, C3, nppiErodeBorder_16u_C3R_Ctx);
impl_morphology_mask_border!(erode_border_u16_c4, u16, C4, nppiErodeBorder_16u_C4R_Ctx);
impl_morphology_mask_border!(erode_border_u16_ac4, u16, AC4, nppiErodeBorder_16u_AC4R_Ctx);
impl_morphology_mask_border!(erode_border_f32_c1, f32, C1, nppiErodeBorder_32f_C1R_Ctx);
impl_morphology_mask_border!(erode_border_f32_c3, f32, C3, nppiErodeBorder_32f_C3R_Ctx);
impl_morphology_mask_border!(erode_border_f32_c4, f32, C4, nppiErodeBorder_32f_C4R_Ctx);
impl_morphology_mask_border!(erode_border_f32_ac4, f32, AC4, nppiErodeBorder_32f_AC4R_Ctx);

impl_generic_morphology_mask_border!(DilateBorderC1, dilate_border, dilate_border_c1, C1, [
    u8 => dilate_border_u8_c1,
    u16 => dilate_border_u16_c1,
    f32 => dilate_border_f32_c1,
]);
impl_generic_morphology_mask_border!(DilateBorderC3, dilate_border, dilate_border_c3, C3, [
    u8 => dilate_border_u8_c3,
    u16 => dilate_border_u16_c3,
    f32 => dilate_border_f32_c3,
]);
impl_generic_morphology_mask_border!(DilateBorderC4, dilate_border, dilate_border_c4, C4, [
    u8 => dilate_border_u8_c4,
    u16 => dilate_border_u16_c4,
    f32 => dilate_border_f32_c4,
]);
impl_generic_morphology_mask_border!(DilateBorderAc4, dilate_border, dilate_border_ac4, AC4, [
    u8 => dilate_border_u8_ac4,
    u16 => dilate_border_u16_ac4,
    f32 => dilate_border_f32_ac4,
]);
impl_generic_morphology_mask_border!(ErodeBorderC1, erode_border, erode_border_c1, C1, [
    u8 => erode_border_u8_c1,
    u16 => erode_border_u16_c1,
    f32 => erode_border_f32_c1,
]);
impl_generic_morphology_mask_border!(ErodeBorderC3, erode_border, erode_border_c3, C3, [
    u8 => erode_border_u8_c3,
    u16 => erode_border_u16_c3,
    f32 => erode_border_f32_c3,
]);
impl_generic_morphology_mask_border!(ErodeBorderC4, erode_border, erode_border_c4, C4, [
    u8 => erode_border_u8_c4,
    u16 => erode_border_u16_c4,
    f32 => erode_border_f32_c4,
]);
impl_generic_morphology_mask_border!(ErodeBorderAc4, erode_border, erode_border_ac4, AC4, [
    u8 => erode_border_u8_ac4,
    u16 => erode_border_u16_ac4,
    f32 => erode_border_f32_ac4,
]);

macro_rules! impl_gray_morphology_mask_border {
    ($name:ident, $ty:ty, $mask_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            mask: &[$mask_ty],
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_mask_shape(mask, mask_size, anchor)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_gray_morphology_mask_border {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty, $mask_ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            type Mask: Copy;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, C1>,
                mask: &[Self::Mask],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, C1>,
            mask: &[T::Mask],
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                source,
                source_offset,
                destination,
                mask,
                mask_size,
                anchor,
                border_type,
            )
        }

        $(
            impl $trait for $ty {
                type Mask = $mask_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    source_offset: Point,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    mask: &[Self::Mask],
                    mask_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                }
            }
        )*
    };
}

impl_gray_morphology_mask_border!(
    gray_dilate_border_u8_c1,
    u8,
    i32,
    nppiGrayDilateBorder_8u_C1R_Ctx
);
impl_gray_morphology_mask_border!(
    gray_dilate_border_f32_c1,
    f32,
    f32,
    nppiGrayDilateBorder_32f_C1R_Ctx
);
impl_gray_morphology_mask_border!(
    gray_erode_border_u8_c1,
    u8,
    i32,
    nppiGrayErodeBorder_8u_C1R_Ctx
);
impl_gray_morphology_mask_border!(
    gray_erode_border_f32_c1,
    f32,
    f32,
    nppiGrayErodeBorder_32f_C1R_Ctx
);

impl_generic_gray_morphology_mask_border!(GrayDilateBorderC1, gray_dilate_border, gray_dilate_border_c1, [
    u8, i32 => gray_dilate_border_u8_c1,
    f32, f32 => gray_dilate_border_f32_c1,
]);
impl_generic_gray_morphology_mask_border!(GrayErodeBorderC1, gray_erode_border, gray_erode_border_c1, [
    u8, i32 => gray_erode_border_u8_c1,
    f32, f32 => gray_erode_border_f32_c1,
]);
