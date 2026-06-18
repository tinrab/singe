use super::*;

macro_rules! impl_alpha_comp_constant {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            alpha1: $ty,
            source2: &ImageView<'_, $ty, $layout>,
            alpha2: $ty,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            operation: AlphaOperation,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    alpha1,
                    source2.as_ptr().cast(),
                    source2.step(),
                    alpha2,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_alpha_comp_constant!(alpha_comp_constant_u8_c1, u8, C1, nppiAlphaCompC_8u_C1R_Ctx);
impl_alpha_comp_constant!(alpha_comp_constant_u8_c3, u8, C3, nppiAlphaCompC_8u_C3R_Ctx);
impl_alpha_comp_constant!(alpha_comp_constant_u8_c4, u8, C4, nppiAlphaCompC_8u_C4R_Ctx);
impl_alpha_comp_constant!(
    alpha_comp_constant_u8_ac4,
    u8,
    AC4,
    nppiAlphaCompC_8u_AC4R_Ctx
);
impl_alpha_comp_constant!(alpha_comp_constant_i8_c1, i8, C1, nppiAlphaCompC_8s_C1R_Ctx);
impl_alpha_comp_constant!(
    alpha_comp_constant_u16_c1,
    u16,
    C1,
    nppiAlphaCompC_16u_C1R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_u16_c3,
    u16,
    C3,
    nppiAlphaCompC_16u_C3R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_u16_c4,
    u16,
    C4,
    nppiAlphaCompC_16u_C4R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_u16_ac4,
    u16,
    AC4,
    nppiAlphaCompC_16u_AC4R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_i16_c1,
    i16,
    C1,
    nppiAlphaCompC_16s_C1R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_u32_c1,
    u32,
    C1,
    nppiAlphaCompC_32u_C1R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_i32_c1,
    i32,
    C1,
    nppiAlphaCompC_32s_C1R_Ctx
);
impl_alpha_comp_constant!(
    alpha_comp_constant_f32_c1,
    f32,
    C1,
    nppiAlphaCompC_32f_C1R_Ctx
);

macro_rules! impl_alpha_comp {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            operation: AlphaOperation,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_alpha_comp!(alpha_comp_u8_ac1, u8, C1, nppiAlphaComp_8u_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_u8_ac4, u8, AC4, nppiAlphaComp_8u_AC4R_Ctx);
impl_alpha_comp!(alpha_comp_i8_ac1, i8, C1, nppiAlphaComp_8s_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_u16_ac1, u16, C1, nppiAlphaComp_16u_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_u16_ac4, u16, AC4, nppiAlphaComp_16u_AC4R_Ctx);
impl_alpha_comp!(alpha_comp_i16_ac1, i16, C1, nppiAlphaComp_16s_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_u32_ac1, u32, C1, nppiAlphaComp_32u_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_u32_ac4, u32, AC4, nppiAlphaComp_32u_AC4R_Ctx);
impl_alpha_comp!(alpha_comp_i32_ac1, i32, C1, nppiAlphaComp_32s_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_i32_ac4, i32, AC4, nppiAlphaComp_32s_AC4R_Ctx);
impl_alpha_comp!(alpha_comp_f32_ac1, f32, C1, nppiAlphaComp_32f_AC1R_Ctx);
impl_alpha_comp!(alpha_comp_f32_ac4, f32, AC4, nppiAlphaComp_32f_AC4R_Ctx);

macro_rules! impl_comp_color_key_scalar {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, u8, C1>,
            source2: &ImageView<'_, u8, C1>,
            color_key: u8,
            destination: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    color_key,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_comp_color_key_array {
    ($name:ident, $layout:ty, $channels:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, u8, $layout>,
            source2: &ImageView<'_, u8, $layout>,
            color_key: [u8; $channels],
            destination: &mut ImageViewMut<'_, u8, $layout>,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            let mut color_key = color_key;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    color_key.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_alpha_comp_color_key {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, u8, AC4>,
            alpha1: u8,
            source2: &ImageView<'_, u8, AC4>,
            alpha2: u8,
            color_key: [u8; 4],
            destination: &mut ImageViewMut<'_, u8, AC4>,
            operation: AlphaOperation,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            let mut color_key = color_key;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    alpha1,
                    source2.as_ptr().cast(),
                    source2.step(),
                    alpha2,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    color_key.as_mut_ptr().cast(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_comp_color_key_scalar!(comp_color_key_u8_c1, nppiCompColorKey_8u_C1R_Ctx);
impl_comp_color_key_array!(comp_color_key_u8_c3, C3, 3, nppiCompColorKey_8u_C3R_Ctx);
impl_comp_color_key_array!(comp_color_key_u8_c4, C4, 4, nppiCompColorKey_8u_C4R_Ctx);
impl_alpha_comp_color_key!(
    alpha_comp_color_key_u8_ac4,
    nppiAlphaCompColorKey_8u_AC4R_Ctx
);

macro_rules! impl_alpha_premultiply_constant {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            alpha: $ty,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    alpha,
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

macro_rules! impl_alpha_premultiply_constant_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            alpha: $ty,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    alpha,
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_alpha_premultiply {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, AC4>,
            destination: &mut ImageViewMut<'_, $ty, AC4>,
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

macro_rules! impl_alpha_premultiply_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $ty, AC4>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_c1,
    u8,
    C1,
    nppiAlphaPremulC_8u_C1R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_c3,
    u8,
    C3,
    nppiAlphaPremulC_8u_C3R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_c4,
    u8,
    C4,
    nppiAlphaPremulC_8u_C4R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_ac4,
    u8,
    AC4,
    nppiAlphaPremulC_8u_AC4R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_c1,
    u16,
    C1,
    nppiAlphaPremulC_16u_C1R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_c3,
    u16,
    C3,
    nppiAlphaPremulC_16u_C3R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_c4,
    u16,
    C4,
    nppiAlphaPremulC_16u_C4R_Ctx
);
impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_ac4,
    u16,
    AC4,
    nppiAlphaPremulC_16u_AC4R_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_c1_in_place,
    u8,
    C1,
    nppiAlphaPremulC_8u_C1IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_c3_in_place,
    u8,
    C3,
    nppiAlphaPremulC_8u_C3IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_c4_in_place,
    u8,
    C4,
    nppiAlphaPremulC_8u_C4IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_ac4_in_place,
    u8,
    AC4,
    nppiAlphaPremulC_8u_AC4IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_c1_in_place,
    u16,
    C1,
    nppiAlphaPremulC_16u_C1IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_c3_in_place,
    u16,
    C3,
    nppiAlphaPremulC_16u_C3IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_c4_in_place,
    u16,
    C4,
    nppiAlphaPremulC_16u_C4IR_Ctx
);
impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_ac4_in_place,
    u16,
    AC4,
    nppiAlphaPremulC_16u_AC4IR_Ctx
);

impl_alpha_premultiply!(alpha_premultiply_u8_ac4, u8, nppiAlphaPremul_8u_AC4R_Ctx);
impl_alpha_premultiply!(alpha_premultiply_u16_ac4, u16, nppiAlphaPremul_16u_AC4R_Ctx);
impl_alpha_premultiply_in_place!(
    alpha_premultiply_u8_ac4_in_place,
    u8,
    nppiAlphaPremul_8u_AC4IR_Ctx
);
impl_alpha_premultiply_in_place!(
    alpha_premultiply_u16_ac4_in_place,
    u16,
    nppiAlphaPremul_16u_AC4IR_Ctx
);

impl_generic_alpha_comp_constant!(
    AlphaCompConstantC1,
    alpha_comp_constant,
    alpha_comp_constant_c1,
    C1,
    [
        u8 => alpha_comp_constant_u8_c1,
        i8 => alpha_comp_constant_i8_c1,
        u16 => alpha_comp_constant_u16_c1,
        i16 => alpha_comp_constant_i16_c1,
        u32 => alpha_comp_constant_u32_c1,
        i32 => alpha_comp_constant_i32_c1,
        f32 => alpha_comp_constant_f32_c1,
    ]
);
impl_generic_alpha_comp_constant!(
    AlphaCompConstantC3,
    alpha_comp_constant,
    alpha_comp_constant_c3,
    C3,
    [
        u8 => alpha_comp_constant_u8_c3,
        u16 => alpha_comp_constant_u16_c3,
    ]
);
impl_generic_alpha_comp_constant!(
    AlphaCompConstantC4,
    alpha_comp_constant,
    alpha_comp_constant_c4,
    C4,
    [
        u8 => alpha_comp_constant_u8_c4,
        u16 => alpha_comp_constant_u16_c4,
    ]
);
impl_generic_alpha_comp_constant!(
    AlphaCompConstantAc4,
    alpha_comp_constant,
    alpha_comp_constant_ac4,
    AC4,
    [
        u8 => alpha_comp_constant_u8_ac4,
        u16 => alpha_comp_constant_u16_ac4,
    ]
);

impl_generic_alpha_comp!(
    AlphaCompAc1,
    alpha_comp,
    alpha_comp_ac1,
    C1,
    [
        u8 => alpha_comp_u8_ac1,
        i8 => alpha_comp_i8_ac1,
        u16 => alpha_comp_u16_ac1,
        i16 => alpha_comp_i16_ac1,
        u32 => alpha_comp_u32_ac1,
        i32 => alpha_comp_i32_ac1,
        f32 => alpha_comp_f32_ac1,
    ]
);
impl_generic_alpha_comp!(
    AlphaCompAc4,
    alpha_comp,
    alpha_comp_ac4,
    AC4,
    [
        u8 => alpha_comp_u8_ac4,
        u16 => alpha_comp_u16_ac4,
        u32 => alpha_comp_u32_ac4,
        i32 => alpha_comp_i32_ac4,
        f32 => alpha_comp_f32_ac4,
    ]
);

impl_generic_constant_scalar_operation!(
    AlphaPremultiplyConstantC1,
    alpha_premultiply_constant,
    alpha_premultiply_constant_c1,
    C1,
    [
        u8, u8 => alpha_premultiply_constant_u8_c1,
        u16, u16 => alpha_premultiply_constant_u16_c1,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    AlphaPremultiplyConstantC1InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_c1_in_place,
    C1,
    [
        u8, u8 => alpha_premultiply_constant_u8_c1_in_place,
        u16, u16 => alpha_premultiply_constant_u16_c1_in_place,
    ]
);
impl_generic_constant_scalar_operation!(
    AlphaPremultiplyConstantC3,
    alpha_premultiply_constant,
    alpha_premultiply_constant_c3,
    C3,
    [
        u8, u8 => alpha_premultiply_constant_u8_c3,
        u16, u16 => alpha_premultiply_constant_u16_c3,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    AlphaPremultiplyConstantC3InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_c3_in_place,
    C3,
    [
        u8, u8 => alpha_premultiply_constant_u8_c3_in_place,
        u16, u16 => alpha_premultiply_constant_u16_c3_in_place,
    ]
);
impl_generic_constant_scalar_operation!(
    AlphaPremultiplyConstantC4,
    alpha_premultiply_constant,
    alpha_premultiply_constant_c4,
    C4,
    [
        u8, u8 => alpha_premultiply_constant_u8_c4,
        u16, u16 => alpha_premultiply_constant_u16_c4,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    AlphaPremultiplyConstantC4InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_c4_in_place,
    C4,
    [
        u8, u8 => alpha_premultiply_constant_u8_c4_in_place,
        u16, u16 => alpha_premultiply_constant_u16_c4_in_place,
    ]
);
impl_generic_constant_scalar_operation!(
    AlphaPremultiplyConstantAc4,
    alpha_premultiply_constant,
    alpha_premultiply_constant_ac4,
    AC4,
    [
        u8, u8 => alpha_premultiply_constant_u8_ac4,
        u16, u16 => alpha_premultiply_constant_u16_ac4,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    AlphaPremultiplyConstantAc4InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_ac4_in_place,
    AC4,
    [
        u8, u8 => alpha_premultiply_constant_u8_ac4_in_place,
        u16, u16 => alpha_premultiply_constant_u16_ac4_in_place,
    ]
);

impl_generic_unary_operation!(
    AlphaPremultiplyAc4,
    alpha_premultiply,
    alpha_premultiply_ac4,
    AC4,
    [
        u8 => alpha_premultiply_u8_ac4,
        u16 => alpha_premultiply_u16_ac4,
    ]
);
impl_generic_unary_operation_in_place!(
    AlphaPremultiplyAc4InPlace,
    alpha_premultiply_in_place,
    alpha_premultiply_ac4_in_place,
    AC4,
    [
        u8 => alpha_premultiply_u8_ac4_in_place,
        u16 => alpha_premultiply_u16_ac4_in_place,
    ]
);
