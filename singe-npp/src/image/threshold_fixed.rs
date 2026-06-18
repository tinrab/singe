use super::*;

impl_threshold_fixed!(threshold_greater_u8_c1, u8, nppiThreshold_GT_8u_C1R_Ctx);
impl_threshold_fixed!(threshold_greater_u16_c1, u16, nppiThreshold_GT_16u_C1R_Ctx);
impl_threshold_fixed!(threshold_greater_i16_c1, i16, nppiThreshold_GT_16s_C1R_Ctx);
impl_threshold_fixed!(threshold_greater_f32_c1, f32, nppiThreshold_GT_32f_C1R_Ctx);
impl_threshold_fixed_in_place!(
    threshold_greater_u8_c1_in_place,
    u8,
    nppiThreshold_GT_8u_C1IR_Ctx
);
impl_threshold_fixed_in_place!(
    threshold_greater_u16_c1_in_place,
    u16,
    nppiThreshold_GT_16u_C1IR_Ctx
);
impl_threshold_fixed_in_place!(
    threshold_greater_i16_c1_in_place,
    i16,
    nppiThreshold_GT_16s_C1IR_Ctx
);
impl_threshold_fixed_in_place!(
    threshold_greater_f32_c1_in_place,
    f32,
    nppiThreshold_GT_32f_C1IR_Ctx
);
impl_threshold_fixed!(threshold_less_u8_c1, u8, nppiThreshold_LT_8u_C1R_Ctx);
impl_threshold_fixed!(threshold_less_u16_c1, u16, nppiThreshold_LT_16u_C1R_Ctx);
impl_threshold_fixed!(threshold_less_i16_c1, i16, nppiThreshold_LT_16s_C1R_Ctx);
impl_threshold_fixed!(threshold_less_f32_c1, f32, nppiThreshold_LT_32f_C1R_Ctx);
impl_threshold_fixed_in_place!(
    threshold_less_u8_c1_in_place,
    u8,
    nppiThreshold_LT_8u_C1IR_Ctx
);
impl_threshold_fixed_in_place!(
    threshold_less_u16_c1_in_place,
    u16,
    nppiThreshold_LT_16u_C1IR_Ctx
);
impl_threshold_fixed_in_place!(
    threshold_less_i16_c1_in_place,
    i16,
    nppiThreshold_LT_16s_C1IR_Ctx
);
impl_threshold_fixed_in_place!(
    threshold_less_f32_c1_in_place,
    f32,
    nppiThreshold_LT_32f_C1IR_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_u8_c3,
    u8,
    C3,
    3,
    nppiThreshold_GT_8u_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_u8_ac4,
    u8,
    AC4,
    3,
    nppiThreshold_GT_8u_AC4R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_u16_c3,
    u16,
    C3,
    3,
    nppiThreshold_GT_16u_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_u16_ac4,
    u16,
    AC4,
    3,
    nppiThreshold_GT_16u_AC4R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_i16_c3,
    i16,
    C3,
    3,
    nppiThreshold_GT_16s_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_i16_ac4,
    i16,
    AC4,
    3,
    nppiThreshold_GT_16s_AC4R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_f32_c3,
    f32,
    C3,
    3,
    nppiThreshold_GT_32f_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_greater_f32_ac4,
    f32,
    AC4,
    3,
    nppiThreshold_GT_32f_AC4R_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_u8_c3_in_place,
    u8,
    C3,
    3,
    nppiThreshold_GT_8u_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_u8_ac4_in_place,
    u8,
    AC4,
    3,
    nppiThreshold_GT_8u_AC4IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiThreshold_GT_16u_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_u16_ac4_in_place,
    u16,
    AC4,
    3,
    nppiThreshold_GT_16u_AC4IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiThreshold_GT_16s_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_i16_ac4_in_place,
    i16,
    AC4,
    3,
    nppiThreshold_GT_16s_AC4IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_f32_c3_in_place,
    f32,
    C3,
    3,
    nppiThreshold_GT_32f_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_greater_f32_ac4_in_place,
    f32,
    AC4,
    3,
    nppiThreshold_GT_32f_AC4IR_Ctx
);
impl_threshold_fixed_packed!(threshold_less_u8_c3, u8, C3, 3, nppiThreshold_LT_8u_C3R_Ctx);
impl_threshold_fixed_packed!(
    threshold_less_u8_ac4,
    u8,
    AC4,
    3,
    nppiThreshold_LT_8u_AC4R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_less_u16_c3,
    u16,
    C3,
    3,
    nppiThreshold_LT_16u_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_less_u16_ac4,
    u16,
    AC4,
    3,
    nppiThreshold_LT_16u_AC4R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_less_i16_c3,
    i16,
    C3,
    3,
    nppiThreshold_LT_16s_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_less_i16_ac4,
    i16,
    AC4,
    3,
    nppiThreshold_LT_16s_AC4R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_less_f32_c3,
    f32,
    C3,
    3,
    nppiThreshold_LT_32f_C3R_Ctx
);
impl_threshold_fixed_packed!(
    threshold_less_f32_ac4,
    f32,
    AC4,
    3,
    nppiThreshold_LT_32f_AC4R_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_u8_c3_in_place,
    u8,
    C3,
    3,
    nppiThreshold_LT_8u_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_u8_ac4_in_place,
    u8,
    AC4,
    3,
    nppiThreshold_LT_8u_AC4IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiThreshold_LT_16u_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_u16_ac4_in_place,
    u16,
    AC4,
    3,
    nppiThreshold_LT_16u_AC4IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiThreshold_LT_16s_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_i16_ac4_in_place,
    i16,
    AC4,
    3,
    nppiThreshold_LT_16s_AC4IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_f32_c3_in_place,
    f32,
    C3,
    3,
    nppiThreshold_LT_32f_C3IR_Ctx
);
impl_threshold_fixed_packed_in_place!(
    threshold_less_f32_ac4_in_place,
    f32,
    AC4,
    3,
    nppiThreshold_LT_32f_AC4IR_Ctx
);

impl_generic_threshold_fixed_scalar!(
    ThresholdGreaterC1,
    threshold_greater,
    threshold_greater_c1,
    [
        u8 => threshold_greater_u8_c1,
        u16 => threshold_greater_u16_c1,
        i16 => threshold_greater_i16_c1,
        f32 => threshold_greater_f32_c1,
    ]
);
impl_generic_threshold_fixed_scalar_in_place!(
    ThresholdGreaterC1InPlace,
    threshold_greater_in_place,
    threshold_greater_c1_in_place,
    [
        u8 => threshold_greater_u8_c1_in_place,
        u16 => threshold_greater_u16_c1_in_place,
        i16 => threshold_greater_i16_c1_in_place,
        f32 => threshold_greater_f32_c1_in_place,
    ]
);
impl_generic_threshold_fixed_array!(
    ThresholdGreaterC3,
    threshold_greater,
    threshold_greater_c3,
    C3,
    3,
    [
        u8 => threshold_greater_u8_c3,
        u16 => threshold_greater_u16_c3,
        i16 => threshold_greater_i16_c3,
        f32 => threshold_greater_f32_c3,
    ]
);
impl_generic_threshold_fixed_array_in_place!(
    ThresholdGreaterC3InPlace,
    threshold_greater_in_place,
    threshold_greater_c3_in_place,
    C3,
    3,
    [
        u8 => threshold_greater_u8_c3_in_place,
        u16 => threshold_greater_u16_c3_in_place,
        i16 => threshold_greater_i16_c3_in_place,
        f32 => threshold_greater_f32_c3_in_place,
    ]
);
impl_generic_threshold_fixed_array!(
    ThresholdGreaterAc4,
    threshold_greater,
    threshold_greater_ac4,
    AC4,
    3,
    [
        u8 => threshold_greater_u8_ac4,
        u16 => threshold_greater_u16_ac4,
        i16 => threshold_greater_i16_ac4,
        f32 => threshold_greater_f32_ac4,
    ]
);
impl_generic_threshold_fixed_array_in_place!(
    ThresholdGreaterAc4InPlace,
    threshold_greater_in_place,
    threshold_greater_ac4_in_place,
    AC4,
    3,
    [
        u8 => threshold_greater_u8_ac4_in_place,
        u16 => threshold_greater_u16_ac4_in_place,
        i16 => threshold_greater_i16_ac4_in_place,
        f32 => threshold_greater_f32_ac4_in_place,
    ]
);
impl_generic_threshold_fixed_scalar!(ThresholdLessC1, threshold_less, threshold_less_c1, [
    u8 => threshold_less_u8_c1,
    u16 => threshold_less_u16_c1,
    i16 => threshold_less_i16_c1,
    f32 => threshold_less_f32_c1,
]);
impl_generic_threshold_fixed_scalar_in_place!(
    ThresholdLessC1InPlace,
    threshold_less_in_place,
    threshold_less_c1_in_place,
    [
        u8 => threshold_less_u8_c1_in_place,
        u16 => threshold_less_u16_c1_in_place,
        i16 => threshold_less_i16_c1_in_place,
        f32 => threshold_less_f32_c1_in_place,
    ]
);
impl_generic_threshold_fixed_array!(
    ThresholdLessC3,
    threshold_less,
    threshold_less_c3,
    C3,
    3,
    [
        u8 => threshold_less_u8_c3,
        u16 => threshold_less_u16_c3,
        i16 => threshold_less_i16_c3,
        f32 => threshold_less_f32_c3,
    ]
);
impl_generic_threshold_fixed_array_in_place!(
    ThresholdLessC3InPlace,
    threshold_less_in_place,
    threshold_less_c3_in_place,
    C3,
    3,
    [
        u8 => threshold_less_u8_c3_in_place,
        u16 => threshold_less_u16_c3_in_place,
        i16 => threshold_less_i16_c3_in_place,
        f32 => threshold_less_f32_c3_in_place,
    ]
);
impl_generic_threshold_fixed_array!(
    ThresholdLessAc4,
    threshold_less,
    threshold_less_ac4,
    AC4,
    3,
    [
        u8 => threshold_less_u8_ac4,
        u16 => threshold_less_u16_ac4,
        i16 => threshold_less_i16_ac4,
        f32 => threshold_less_f32_ac4,
    ]
);
impl_generic_threshold_fixed_array_in_place!(
    ThresholdLessAc4InPlace,
    threshold_less_in_place,
    threshold_less_ac4_in_place,
    AC4,
    3,
    [
        u8 => threshold_less_u8_ac4_in_place,
        u16 => threshold_less_u16_ac4_in_place,
        i16 => threshold_less_i16_ac4_in_place,
        f32 => threshold_less_f32_ac4_in_place,
    ]
);
