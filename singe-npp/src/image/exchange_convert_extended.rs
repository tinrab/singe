use super::*;

impl_convert!(
    convert_i16_to_u16_c1,
    i16,
    u16,
    C1,
    nppiConvert_16s16u_C1Rs_Ctx
);
impl_convert!(
    convert_i16_to_u32_c1,
    i16,
    u32,
    C1,
    nppiConvert_16s32u_C1Rs_Ctx
);
impl_convert!(
    convert_i32_to_u32_c1,
    i32,
    u32,
    C1,
    nppiConvert_32s32u_C1Rs_Ctx
);
impl_convert_scaled_round!(
    convert_u16_to_i8_scaled_c1,
    u16,
    i8,
    C1,
    nppiConvert_16u8s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_i16_to_i8_scaled_c1,
    i16,
    i8,
    C1,
    nppiConvert_16s8s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_u16_to_i16_scaled_c1,
    u16,
    i16,
    C1,
    nppiConvert_16u16s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_u32_to_u8_scaled_c1,
    u32,
    u8,
    C1,
    nppiConvert_32u8u_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_u32_to_i8_scaled_c1,
    u32,
    i8,
    C1,
    nppiConvert_32u8s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_u32_to_u16_scaled_c1,
    u32,
    u16,
    C1,
    nppiConvert_32u16u_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_u32_to_i16_scaled_c1,
    u32,
    i16,
    C1,
    nppiConvert_32u16s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_i32_to_u16_scaled_c1,
    i32,
    u16,
    C1,
    nppiConvert_32s16u_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_i32_to_i16_scaled_c1,
    i32,
    i16,
    C1,
    nppiConvert_32s16s_C1RSfs_Ctx
);
impl_convert_round!(convert_f32_to_u8_c1, f32, u8, C1, nppiConvert_32f8u_C1R_Ctx);
impl_convert_round!(convert_f32_to_u8_c3, f32, u8, C3, nppiConvert_32f8u_C3R_Ctx);
impl_convert_round!(convert_f32_to_u8_c4, f32, u8, C4, nppiConvert_32f8u_C4R_Ctx);
impl_convert_round!(
    convert_f32_to_u8_ac4,
    f32,
    u8,
    AC4,
    nppiConvert_32f8u_AC4R_Ctx
);
impl_generic_convert_to_operation!(
    ConvertToU8C1,
    convert_to_u8_c1,
    u8,
    C1,
    [
        i8 => convert_i8_to_u8_c1,
        u16 => convert_u16_to_u8_c1,
        i16 => convert_i16_to_u8_c1,
        i32 => convert_i32_to_u8_c1,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToU8C3,
    convert_to_u8_c3,
    u8,
    C3,
    [
        u16 => convert_u16_to_u8_c3,
        i16 => convert_i16_to_u8_c3,
        i32 => convert_i32_to_u8_c3,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToU8C4,
    convert_to_u8_c4,
    u8,
    C4,
    [
        u16 => convert_u16_to_u8_c4,
        i16 => convert_i16_to_u8_c4,
        i32 => convert_i32_to_u8_c4,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToU8AC4,
    convert_to_u8_ac4,
    u8,
    AC4,
    [
        u16 => convert_u16_to_u8_ac4,
        i16 => convert_i16_to_u8_ac4,
        i32 => convert_i32_to_u8_ac4,
    ]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU8C1,
    convert_round_to_u8_c1,
    u8,
    C1,
    [f32 => convert_f32_to_u8_c1]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU8C3,
    convert_round_to_u8_c3,
    u8,
    C3,
    [f32 => convert_f32_to_u8_c3]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU8C4,
    convert_round_to_u8_c4,
    u8,
    C4,
    [f32 => convert_f32_to_u8_c4]
);
impl_generic_convert_round_to_operation!(
    ConvertRoundToU8AC4,
    convert_round_to_u8_ac4,
    u8,
    AC4,
    [f32 => convert_f32_to_u8_ac4]
);
impl_convert_round!(convert_f32_to_i8_c1, f32, i8, C1, nppiConvert_32f8s_C1R_Ctx);
impl_convert_round!(convert_f32_to_i8_c3, f32, i8, C3, nppiConvert_32f8s_C3R_Ctx);
impl_convert_round!(convert_f32_to_i8_c4, f32, i8, C4, nppiConvert_32f8s_C4R_Ctx);
impl_convert_round!(
    convert_f32_to_i8_ac4,
    f32,
    i8,
    AC4,
    nppiConvert_32f8s_AC4R_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_u8_scaled_c1,
    f32,
    u8,
    C1,
    nppiConvert_32f8u_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_i8_scaled_c1,
    f32,
    i8,
    C1,
    nppiConvert_32f8s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_u16_scaled_c1,
    f32,
    u16,
    C1,
    nppiConvert_32f16u_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_i16_scaled_c1,
    f32,
    i16,
    C1,
    nppiConvert_32f16s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_u32_scaled_c1,
    f32,
    u32,
    C1,
    nppiConvert_32f32u_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_f32_to_i32_scaled_c1,
    f32,
    i32,
    C1,
    nppiConvert_32f32s_C1RSfs_Ctx
);
impl_convert_scaled_round!(
    convert_u8_to_i8_scaled_c1,
    u8,
    i8,
    C1,
    nppiConvert_8u8s_C1RSfs_Ctx
);
impl_generic_convert_scaled_round_to_operation!(
    ConvertScaledRoundToI8C1,
    convert_scaled_round_to_i8_c1,
    i8,
    C1,
    [
        u8 => convert_u8_to_i8_scaled_c1,
        u16 => convert_u16_to_i8_scaled_c1,
        i16 => convert_i16_to_i8_scaled_c1,
        u32 => convert_u32_to_i8_scaled_c1,
        f32 => convert_f32_to_i8_scaled_c1,
    ]
);
impl_generic_convert_scaled_round_to_operation!(
    ConvertScaledRoundToU8C1,
    convert_scaled_round_to_u8_c1,
    u8,
    C1,
    [
        u32 => convert_u32_to_u8_scaled_c1,
        f32 => convert_f32_to_u8_scaled_c1,
    ]
);
impl_generic_convert_scaled_round_to_operation!(
    ConvertScaledRoundToI16C1,
    convert_scaled_round_to_i16_c1,
    i16,
    C1,
    [
        u16 => convert_u16_to_i16_scaled_c1,
        u32 => convert_u32_to_i16_scaled_c1,
        i32 => convert_i32_to_i16_scaled_c1,
        f32 => convert_f32_to_i16_scaled_c1,
    ]
);
impl_generic_convert_scaled_round_to_operation!(
    ConvertScaledRoundToU16C1,
    convert_scaled_round_to_u16_c1,
    u16,
    C1,
    [
        u32 => convert_u32_to_u16_scaled_c1,
        i32 => convert_i32_to_u16_scaled_c1,
        f32 => convert_f32_to_u16_scaled_c1,
    ]
);
impl_generic_convert_scaled_round_to_operation!(
    ConvertScaledRoundToI32C1,
    convert_scaled_round_to_i32_c1,
    i32,
    C1,
    [
        u32 => convert_u32_to_i32_scaled_c1,
        f32 => convert_f32_to_i32_scaled_c1,
    ]
);
impl_generic_convert_scaled_round_to_operation!(
    ConvertScaledRoundToU32C1,
    convert_scaled_round_to_u32_c1,
    u32,
    C1,
    [f32 => convert_f32_to_u32_scaled_c1]
);
impl_convert!(convert_i8_to_u8_c1, i8, u8, C1, nppiConvert_8s8u_C1Rs_Ctx);
impl_convert!(
    convert_i8_to_u16_c1,
    i8,
    u16,
    C1,
    nppiConvert_8s16u_C1Rs_Ctx
);
impl_convert!(convert_i8_to_i16_c1, i8, i16, C1, nppiConvert_8s16s_C1R_Ctx);
impl_convert!(
    convert_i8_to_u32_c1,
    i8,
    u32,
    C1,
    nppiConvert_8s32u_C1Rs_Ctx
);
impl_convert!(convert_i8_to_i32_c1, i8, i32, C1, nppiConvert_8s32s_C1R_Ctx);
impl_convert!(convert_i8_to_i32_c3, i8, i32, C3, nppiConvert_8s32s_C3R_Ctx);
impl_convert!(convert_i8_to_i32_c4, i8, i32, C4, nppiConvert_8s32s_C4R_Ctx);
impl_convert!(
    convert_i8_to_i32_ac4,
    i8,
    i32,
    AC4,
    nppiConvert_8s32s_AC4R_Ctx
);
impl_convert!(convert_i8_to_f32_c1, i8, f32, C1, nppiConvert_8s32f_C1R_Ctx);
impl_convert!(convert_i8_to_f32_c3, i8, f32, C3, nppiConvert_8s32f_C3R_Ctx);
impl_convert!(convert_i8_to_f32_c4, i8, f32, C4, nppiConvert_8s32f_C4R_Ctx);
impl_convert!(
    convert_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiConvert_8s32f_AC4R_Ctx
);
impl_convert!(convert_i32_to_i8_c1, i32, i8, C1, nppiConvert_32s8s_C1R_Ctx);
impl_convert!(convert_i32_to_i8_c3, i32, i8, C3, nppiConvert_32s8s_C3R_Ctx);
impl_convert!(convert_i32_to_i8_c4, i32, i8, C4, nppiConvert_32s8s_C4R_Ctx);
impl_convert!(
    convert_i32_to_i8_ac4,
    i32,
    i8,
    AC4,
    nppiConvert_32s8s_AC4R_Ctx
);
impl_convert!(
    convert_i32_to_f32_c1,
    i32,
    f32,
    C1,
    nppiConvert_32s32f_C1R_Ctx
);
impl_convert!(
    convert_u32_to_f32_c1,
    u32,
    f32,
    C1,
    nppiConvert_32u32f_C1R_Ctx
);
impl_generic_convert_to_operation!(
    ConvertToF32C1,
    convert_to_f32_c1,
    f32,
    C1,
    [
        u8 => convert_u8_to_f32_c1,
        i8 => convert_i8_to_f32_c1,
        u16 => convert_u16_to_f32_c1,
        i16 => convert_i16_to_f32_c1,
        f16 => convert_f16_to_f32_c1,
        i32 => convert_i32_to_f32_c1,
        u32 => convert_u32_to_f32_c1,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToF32C3,
    convert_to_f32_c3,
    f32,
    C3,
    [
        u8 => convert_u8_to_f32_c3,
        i8 => convert_i8_to_f32_c3,
        u16 => convert_u16_to_f32_c3,
        i16 => convert_i16_to_f32_c3,
        f16 => convert_f16_to_f32_c3,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToF32C4,
    convert_to_f32_c4,
    f32,
    C4,
    [
        u8 => convert_u8_to_f32_c4,
        i8 => convert_i8_to_f32_c4,
        u16 => convert_u16_to_f32_c4,
        i16 => convert_i16_to_f32_c4,
        f16 => convert_f16_to_f32_c4,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToF32AC4,
    convert_to_f32_ac4,
    f32,
    AC4,
    [
        u8 => convert_u8_to_f32_ac4,
        i8 => convert_i8_to_f32_ac4,
        u16 => convert_u16_to_f32_ac4,
        i16 => convert_i16_to_f32_ac4,
        f16 => convert_f16_to_f32_ac4,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToI32C1,
    convert_to_i32_c1,
    i32,
    C1,
    [
        u8 => convert_u8_to_i32_c1,
        i8 => convert_i8_to_i32_c1,
        u16 => convert_u16_to_i32_c1,
        i16 => convert_i16_to_i32_c1,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToI32C3,
    convert_to_i32_c3,
    i32,
    C3,
    [
        u8 => convert_u8_to_i32_c3,
        i8 => convert_i8_to_i32_c3,
        u16 => convert_u16_to_i32_c3,
        i16 => convert_i16_to_i32_c3,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToI32C4,
    convert_to_i32_c4,
    i32,
    C4,
    [
        u8 => convert_u8_to_i32_c4,
        i8 => convert_i8_to_i32_c4,
        u16 => convert_u16_to_i32_c4,
        i16 => convert_i16_to_i32_c4,
    ]
);
impl_generic_convert_to_operation!(
    ConvertToI32AC4,
    convert_to_i32_ac4,
    i32,
    AC4,
    [
        u8 => convert_u8_to_i32_ac4,
        i8 => convert_i8_to_i32_ac4,
        u16 => convert_u16_to_i32_ac4,
        i16 => convert_i16_to_i32_ac4,
    ]
);
impl_convert_scaled_round!(
    convert_u32_to_i32_scaled_c1,
    u32,
    i32,
    C1,
    nppiConvert_32u32s_C1RSfs_Ctx
);
