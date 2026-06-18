use super::*;

macro_rules! impl_accumulate_square {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_square_masked {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            mask: &MaskView<'_>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            validate_same_size(source.size(), mask.size())?;
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_product {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $source_ty, C1>,
            source2: &ImageView<'_, $source_ty, C1>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source1.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_product_masked {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $source_ty, C1>,
            source2: &ImageView<'_, $source_ty, C1>,
            mask: &MaskView<'_>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), mask.size())?;
            validate_same_size(source1.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source1.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_weighted {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
            alpha: f32,
        ) -> Result<()> {
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_weighted_masked {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            mask: &MaskView<'_>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
            alpha: f32,
        ) -> Result<()> {
            validate_same_size(source.size(), mask.size())?;
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_accumulate_square!(
    accumulate_square_u8_to_f32_c1,
    u8,
    f32,
    nppiAddSquare_8u32f_C1IR_Ctx
);
impl_accumulate_square_masked!(
    accumulate_square_u8_to_f32_c1_masked,
    u8,
    f32,
    nppiAddSquare_8u32f_C1IMR_Ctx
);
impl_accumulate_square!(
    accumulate_square_u16_to_f32_c1,
    u16,
    f32,
    nppiAddSquare_16u32f_C1IR_Ctx
);
impl_accumulate_square_masked!(
    accumulate_square_u16_to_f32_c1_masked,
    u16,
    f32,
    nppiAddSquare_16u32f_C1IMR_Ctx
);
impl_accumulate_square!(
    accumulate_square_f32_c1,
    f32,
    f32,
    nppiAddSquare_32f_C1IR_Ctx
);
impl_accumulate_square_masked!(
    accumulate_square_f32_c1_masked,
    f32,
    f32,
    nppiAddSquare_32f_C1IMR_Ctx
);

impl_accumulate_product!(
    accumulate_product_u8_to_f32_c1,
    u8,
    f32,
    nppiAddProduct_8u32f_C1IR_Ctx
);
impl_accumulate_product_masked!(
    accumulate_product_u8_to_f32_c1_masked,
    u8,
    f32,
    nppiAddProduct_8u32f_C1IMR_Ctx
);
impl_accumulate_product!(
    accumulate_product_u16_to_f32_c1,
    u16,
    f32,
    nppiAddProduct_16u32f_C1IR_Ctx
);
impl_accumulate_product_masked!(
    accumulate_product_u16_to_f32_c1_masked,
    u16,
    f32,
    nppiAddProduct_16u32f_C1IMR_Ctx
);
impl_accumulate_product!(
    accumulate_product_f32_c1,
    f32,
    f32,
    nppiAddProduct_32f_C1IR_Ctx
);
impl_accumulate_product_masked!(
    accumulate_product_f32_c1_masked,
    f32,
    f32,
    nppiAddProduct_32f_C1IMR_Ctx
);
impl_accumulate_product!(
    accumulate_product_f16_c1,
    f16,
    f16,
    nppiAddProduct_16f_C1IR_Ctx
);

impl_accumulate_weighted!(
    accumulate_weighted_u8_to_f32_c1,
    u8,
    f32,
    nppiAddWeighted_8u32f_C1IR_Ctx
);
impl_accumulate_weighted_masked!(
    accumulate_weighted_u8_to_f32_c1_masked,
    u8,
    f32,
    nppiAddWeighted_8u32f_C1IMR_Ctx
);
impl_accumulate_weighted!(
    accumulate_weighted_u16_to_f32_c1,
    u16,
    f32,
    nppiAddWeighted_16u32f_C1IR_Ctx
);
impl_accumulate_weighted_masked!(
    accumulate_weighted_u16_to_f32_c1_masked,
    u16,
    f32,
    nppiAddWeighted_16u32f_C1IMR_Ctx
);
impl_accumulate_weighted!(
    accumulate_weighted_f32_c1,
    f32,
    f32,
    nppiAddWeighted_32f_C1IR_Ctx
);
impl_accumulate_weighted_masked!(
    accumulate_weighted_f32_c1_masked,
    f32,
    f32,
    nppiAddWeighted_32f_C1IMR_Ctx
);
