use super::*;

impl_threshold_c1!(threshold_u8_c1, u8, nppiThreshold_8u_C1R_Ctx);
impl_threshold_c1!(threshold_u16_c1, u16, nppiThreshold_16u_C1R_Ctx);
impl_threshold_c1!(threshold_i16_c1, i16, nppiThreshold_16s_C1R_Ctx);
impl_threshold_c1!(threshold_f32_c1, f32, nppiThreshold_32f_C1R_Ctx);
impl_threshold_c1_in_place!(threshold_u8_c1_in_place, u8, nppiThreshold_8u_C1IR_Ctx);
impl_threshold_c1_in_place!(threshold_u16_c1_in_place, u16, nppiThreshold_16u_C1IR_Ctx);
impl_threshold_c1_in_place!(threshold_i16_c1_in_place, i16, nppiThreshold_16s_C1IR_Ctx);
impl_threshold_c1_in_place!(threshold_f32_c1_in_place, f32, nppiThreshold_32f_C1IR_Ctx);
impl_threshold_packed!(threshold_u8_c3, u8, C3, 3, nppiThreshold_8u_C3R_Ctx);
impl_threshold_packed!(threshold_u8_ac4, u8, AC4, 3, nppiThreshold_8u_AC4R_Ctx);
impl_threshold_packed!(threshold_u16_c3, u16, C3, 3, nppiThreshold_16u_C3R_Ctx);
impl_threshold_packed!(threshold_u16_ac4, u16, AC4, 3, nppiThreshold_16u_AC4R_Ctx);
impl_threshold_packed!(threshold_i16_c3, i16, C3, 3, nppiThreshold_16s_C3R_Ctx);
impl_threshold_packed!(threshold_i16_ac4, i16, AC4, 3, nppiThreshold_16s_AC4R_Ctx);
impl_threshold_packed!(threshold_f32_c3, f32, C3, 3, nppiThreshold_32f_C3R_Ctx);
impl_threshold_packed!(threshold_f32_ac4, f32, AC4, 3, nppiThreshold_32f_AC4R_Ctx);
impl_threshold_packed_in_place!(
    threshold_u8_c3_in_place,
    u8,
    C3,
    3,
    nppiThreshold_8u_C3IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_u8_ac4_in_place,
    u8,
    AC4,
    3,
    nppiThreshold_8u_AC4IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiThreshold_16u_C3IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_u16_ac4_in_place,
    u16,
    AC4,
    3,
    nppiThreshold_16u_AC4IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiThreshold_16s_C3IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_i16_ac4_in_place,
    i16,
    AC4,
    3,
    nppiThreshold_16s_AC4IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_f32_c3_in_place,
    f32,
    C3,
    3,
    nppiThreshold_32f_C3IR_Ctx
);
impl_threshold_packed_in_place!(
    threshold_f32_ac4_in_place,
    f32,
    AC4,
    3,
    nppiThreshold_32f_AC4IR_Ctx
);

impl_generic_threshold_scalar!(ThresholdC1, threshold, threshold_c1, [
    u8 => threshold_u8_c1,
    u16 => threshold_u16_c1,
    i16 => threshold_i16_c1,
    f32 => threshold_f32_c1,
]);
impl_generic_threshold_scalar_in_place!(
    ThresholdC1InPlace,
    threshold_in_place,
    threshold_c1_in_place,
    [
        u8 => threshold_u8_c1_in_place,
        u16 => threshold_u16_c1_in_place,
        i16 => threshold_i16_c1_in_place,
        f32 => threshold_f32_c1_in_place,
    ]
);
impl_generic_threshold_array!(ThresholdC3, threshold, threshold_c3, C3, 3, [
    u8 => threshold_u8_c3,
    u16 => threshold_u16_c3,
    i16 => threshold_i16_c3,
    f32 => threshold_f32_c3,
]);
impl_generic_threshold_array_in_place!(
    ThresholdC3InPlace,
    threshold_in_place,
    threshold_c3_in_place,
    C3,
    3,
    [
        u8 => threshold_u8_c3_in_place,
        u16 => threshold_u16_c3_in_place,
        i16 => threshold_i16_c3_in_place,
        f32 => threshold_f32_c3_in_place,
    ]
);
impl_generic_threshold_array!(ThresholdAc4, threshold, threshold_ac4, AC4, 3, [
    u8 => threshold_u8_ac4,
    u16 => threshold_u16_ac4,
    i16 => threshold_i16_ac4,
    f32 => threshold_f32_ac4,
]);
impl_generic_threshold_array_in_place!(
    ThresholdAc4InPlace,
    threshold_in_place,
    threshold_ac4_in_place,
    AC4,
    3,
    [
        u8 => threshold_u8_ac4_in_place,
        u16 => threshold_u16_ac4_in_place,
        i16 => threshold_i16_ac4_in_place,
        f32 => threshold_f32_ac4_in_place,
    ]
);
