use super::*;

impl_scale_to_higher!(scale_u8_to_u16_c1, u8, u16, C1, nppiScale_8u16u_C1R_Ctx);
impl_scale_to_higher!(scale_u8_to_u16_c3, u8, u16, C3, nppiScale_8u16u_C3R_Ctx);
impl_scale_to_higher!(scale_u8_to_u16_c4, u8, u16, C4, nppiScale_8u16u_C4R_Ctx);
impl_scale_to_higher!(scale_u8_to_u16_ac4, u8, u16, AC4, nppiScale_8u16u_AC4R_Ctx);
impl_scale_to_higher!(scale_u8_to_i16_c1, u8, i16, C1, nppiScale_8u16s_C1R_Ctx);
impl_scale_to_higher!(scale_u8_to_i16_c3, u8, i16, C3, nppiScale_8u16s_C3R_Ctx);
impl_scale_to_higher!(scale_u8_to_i16_c4, u8, i16, C4, nppiScale_8u16s_C4R_Ctx);
impl_scale_to_higher!(scale_u8_to_i16_ac4, u8, i16, AC4, nppiScale_8u16s_AC4R_Ctx);
impl_scale_to_higher!(scale_u8_to_i32_c1, u8, i32, C1, nppiScale_8u32s_C1R_Ctx);
impl_scale_to_higher!(scale_u8_to_i32_c3, u8, i32, C3, nppiScale_8u32s_C3R_Ctx);
impl_scale_to_higher!(scale_u8_to_i32_c4, u8, i32, C4, nppiScale_8u32s_C4R_Ctx);
impl_scale_to_higher!(scale_u8_to_i32_ac4, u8, i32, AC4, nppiScale_8u32s_AC4R_Ctx);
impl_scale_to_higher_range!(scale_u8_to_f32_c1, u8, C1, nppiScale_8u32f_C1R_Ctx);
impl_scale_to_higher_range!(scale_u8_to_f32_c3, u8, C3, nppiScale_8u32f_C3R_Ctx);
impl_scale_to_higher_range!(scale_u8_to_f32_c4, u8, C4, nppiScale_8u32f_C4R_Ctx);
impl_scale_to_higher_range!(scale_u8_to_f32_ac4, u8, AC4, nppiScale_8u32f_AC4R_Ctx);
impl_scale_to_lower_hint!(scale_u16_to_u8_c1, u16, C1, nppiScale_16u8u_C1R_Ctx);
impl_scale_to_lower_hint!(scale_u16_to_u8_c3, u16, C3, nppiScale_16u8u_C3R_Ctx);
impl_scale_to_lower_hint!(scale_u16_to_u8_c4, u16, C4, nppiScale_16u8u_C4R_Ctx);
impl_scale_to_lower_hint!(scale_u16_to_u8_ac4, u16, AC4, nppiScale_16u8u_AC4R_Ctx);
impl_scale_to_lower_hint!(scale_i16_to_u8_c1, i16, C1, nppiScale_16s8u_C1R_Ctx);
impl_scale_to_lower_hint!(scale_i16_to_u8_c3, i16, C3, nppiScale_16s8u_C3R_Ctx);
impl_scale_to_lower_hint!(scale_i16_to_u8_c4, i16, C4, nppiScale_16s8u_C4R_Ctx);
impl_scale_to_lower_hint!(scale_i16_to_u8_ac4, i16, AC4, nppiScale_16s8u_AC4R_Ctx);
impl_scale_to_lower_hint!(scale_i32_to_u8_c1, i32, C1, nppiScale_32s8u_C1R_Ctx);
impl_scale_to_lower_hint!(scale_i32_to_u8_c3, i32, C3, nppiScale_32s8u_C3R_Ctx);
impl_scale_to_lower_hint!(scale_i32_to_u8_c4, i32, C4, nppiScale_32s8u_C4R_Ctx);
impl_scale_to_lower_hint!(scale_i32_to_u8_ac4, i32, AC4, nppiScale_32s8u_AC4R_Ctx);
impl_scale_to_lower_range!(scale_f32_to_u8_c1, C1, nppiScale_32f8u_C1R_Ctx);
impl_scale_to_lower_range!(scale_f32_to_u8_c3, C3, nppiScale_32f8u_C3R_Ctx);
impl_scale_to_lower_range!(scale_f32_to_u8_c4, C4, nppiScale_32f8u_C4R_Ctx);
impl_scale_to_lower_range!(scale_f32_to_u8_ac4, AC4, nppiScale_32f8u_AC4R_Ctx);
impl_generic_scale_to_higher_operation!(
    ScaleToU16C1,
    scale_to_u16_c1,
    C1,
    [u8 => (u16, scale_u8_to_u16_c1)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToU16C3,
    scale_to_u16_c3,
    C3,
    [u8 => (u16, scale_u8_to_u16_c3)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToU16C4,
    scale_to_u16_c4,
    C4,
    [u8 => (u16, scale_u8_to_u16_c4)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToU16AC4,
    scale_to_u16_ac4,
    AC4,
    [u8 => (u16, scale_u8_to_u16_ac4)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI16C1,
    scale_to_i16_c1,
    C1,
    [u8 => (i16, scale_u8_to_i16_c1)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI16C3,
    scale_to_i16_c3,
    C3,
    [u8 => (i16, scale_u8_to_i16_c3)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI16C4,
    scale_to_i16_c4,
    C4,
    [u8 => (i16, scale_u8_to_i16_c4)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI16AC4,
    scale_to_i16_ac4,
    AC4,
    [u8 => (i16, scale_u8_to_i16_ac4)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI32C1,
    scale_to_i32_c1,
    C1,
    [u8 => (i32, scale_u8_to_i32_c1)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI32C3,
    scale_to_i32_c3,
    C3,
    [u8 => (i32, scale_u8_to_i32_c3)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI32C4,
    scale_to_i32_c4,
    C4,
    [u8 => (i32, scale_u8_to_i32_c4)]
);
impl_generic_scale_to_higher_operation!(
    ScaleToI32AC4,
    scale_to_i32_ac4,
    AC4,
    [u8 => (i32, scale_u8_to_i32_ac4)]
);
impl_generic_scale_to_f32_operation!(
    ScaleToF32C1,
    scale_to_f32_c1,
    C1,
    [u8 => scale_u8_to_f32_c1]
);
impl_generic_scale_to_f32_operation!(
    ScaleToF32C3,
    scale_to_f32_c3,
    C3,
    [u8 => scale_u8_to_f32_c3]
);
impl_generic_scale_to_f32_operation!(
    ScaleToF32C4,
    scale_to_f32_c4,
    C4,
    [u8 => scale_u8_to_f32_c4]
);
impl_generic_scale_to_f32_operation!(
    ScaleToF32AC4,
    scale_to_f32_ac4,
    AC4,
    [u8 => scale_u8_to_f32_ac4]
);
impl_generic_scale_to_u8_hint_operation!(
    ScaleToU8HintC1,
    scale_to_u8_hint_c1,
    C1,
    [
        u16 => scale_u16_to_u8_c1,
        i16 => scale_i16_to_u8_c1,
        i32 => scale_i32_to_u8_c1,
    ]
);
impl_generic_scale_to_u8_hint_operation!(
    ScaleToU8HintC3,
    scale_to_u8_hint_c3,
    C3,
    [
        u16 => scale_u16_to_u8_c3,
        i16 => scale_i16_to_u8_c3,
        i32 => scale_i32_to_u8_c3,
    ]
);
impl_generic_scale_to_u8_hint_operation!(
    ScaleToU8HintC4,
    scale_to_u8_hint_c4,
    C4,
    [
        u16 => scale_u16_to_u8_c4,
        i16 => scale_i16_to_u8_c4,
        i32 => scale_i32_to_u8_c4,
    ]
);
impl_generic_scale_to_u8_hint_operation!(
    ScaleToU8HintAC4,
    scale_to_u8_hint_ac4,
    AC4,
    [
        u16 => scale_u16_to_u8_ac4,
        i16 => scale_i16_to_u8_ac4,
        i32 => scale_i32_to_u8_ac4,
    ]
);
impl_generic_scale_to_u8_range_operation!(
    ScaleToU8RangeC1,
    scale_to_u8_range_c1,
    C1,
    [f32 => scale_f32_to_u8_c1]
);
impl_generic_scale_to_u8_range_operation!(
    ScaleToU8RangeC3,
    scale_to_u8_range_c3,
    C3,
    [f32 => scale_f32_to_u8_c3]
);
impl_generic_scale_to_u8_range_operation!(
    ScaleToU8RangeC4,
    scale_to_u8_range_c4,
    C4,
    [f32 => scale_f32_to_u8_c4]
);
impl_generic_scale_to_u8_range_operation!(
    ScaleToU8RangeAC4,
    scale_to_u8_range_ac4,
    AC4,
    [f32 => scale_f32_to_u8_ac4]
);
