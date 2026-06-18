use super::*;

impl_unary_operation!(absolute_i16_c1, i16, C1, nppiAbs_16s_C1R_Ctx);
impl_unary_operation_in_place!(absolute_i16_c1_in_place, i16, C1, nppiAbs_16s_C1IR_Ctx);
impl_unary_operation!(absolute_i16_c3, i16, C3, nppiAbs_16s_C3R_Ctx);
impl_unary_operation_in_place!(absolute_i16_c3_in_place, i16, C3, nppiAbs_16s_C3IR_Ctx);
impl_unary_operation!(absolute_i16_ac4, i16, AC4, nppiAbs_16s_AC4R_Ctx);
impl_unary_operation_in_place!(absolute_i16_ac4_in_place, i16, AC4, nppiAbs_16s_AC4IR_Ctx);
impl_unary_operation!(absolute_i16_c4, i16, C4, nppiAbs_16s_C4R_Ctx);
impl_unary_operation_in_place!(absolute_i16_c4_in_place, i16, C4, nppiAbs_16s_C4IR_Ctx);
impl_unary_operation!(absolute_f16_c1, f16, C1, nppiAbs_16f_C1R_Ctx);
impl_unary_operation_in_place!(absolute_f16_c1_in_place, f16, C1, nppiAbs_16f_C1IR_Ctx);
impl_unary_operation!(absolute_f16_c3, f16, C3, nppiAbs_16f_C3R_Ctx);
impl_unary_operation_in_place!(absolute_f16_c3_in_place, f16, C3, nppiAbs_16f_C3IR_Ctx);
impl_unary_operation!(absolute_f16_c4, f16, C4, nppiAbs_16f_C4R_Ctx);
impl_unary_operation_in_place!(absolute_f16_c4_in_place, f16, C4, nppiAbs_16f_C4IR_Ctx);
impl_unary_operation!(absolute_f32_c1, f32, C1, nppiAbs_32f_C1R_Ctx);
impl_unary_operation_in_place!(absolute_f32_c1_in_place, f32, C1, nppiAbs_32f_C1IR_Ctx);
impl_unary_operation!(absolute_f32_c3, f32, C3, nppiAbs_32f_C3R_Ctx);
impl_unary_operation_in_place!(absolute_f32_c3_in_place, f32, C3, nppiAbs_32f_C3IR_Ctx);
impl_unary_operation!(absolute_f32_ac4, f32, AC4, nppiAbs_32f_AC4R_Ctx);
impl_unary_operation_in_place!(absolute_f32_ac4_in_place, f32, AC4, nppiAbs_32f_AC4IR_Ctx);
impl_unary_operation!(absolute_f32_c4, f32, C4, nppiAbs_32f_C4R_Ctx);
impl_unary_operation_in_place!(absolute_f32_c4_in_place, f32, C4, nppiAbs_32f_C4IR_Ctx);

impl_binary!(absolute_difference_u8_c1, u8, C1, nppiAbsDiff_8u_C1R_Ctx);
impl_binary!(absolute_difference_u8_c3, u8, C3, nppiAbsDiff_8u_C3R_Ctx);
impl_binary!(absolute_difference_u8_c4, u8, C4, nppiAbsDiff_8u_C4R_Ctx);
impl_binary!(absolute_difference_u16_c1, u16, C1, nppiAbsDiff_16u_C1R_Ctx);
impl_binary!(absolute_difference_f16_c1, f16, C1, nppiAbsDiff_16f_C1R_Ctx);
impl_binary!(absolute_difference_f32_c1, f32, C1, nppiAbsDiff_32f_C1R_Ctx);

impl_unary_operation_scaled!(square_u8_c1, u8, C1, nppiSqr_8u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u8_c1_in_place, u8, C1, nppiSqr_8u_C1IRSfs_Ctx);
impl_unary_operation_scaled!(square_u8_c3, u8, C3, nppiSqr_8u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u8_c3_in_place, u8, C3, nppiSqr_8u_C3IRSfs_Ctx);
impl_unary_operation_scaled!(square_u8_ac4, u8, AC4, nppiSqr_8u_AC4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u8_ac4_in_place, u8, AC4, nppiSqr_8u_AC4IRSfs_Ctx);
impl_unary_operation_scaled!(square_u8_c4, u8, C4, nppiSqr_8u_C4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u8_c4_in_place, u8, C4, nppiSqr_8u_C4IRSfs_Ctx);
impl_unary_operation_scaled!(square_u16_c1, u16, C1, nppiSqr_16u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u16_c1_in_place, u16, C1, nppiSqr_16u_C1IRSfs_Ctx);
impl_unary_operation_scaled!(square_u16_c3, u16, C3, nppiSqr_16u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u16_c3_in_place, u16, C3, nppiSqr_16u_C3IRSfs_Ctx);
impl_unary_operation_scaled!(square_u16_ac4, u16, AC4, nppiSqr_16u_AC4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u16_ac4_in_place, u16, AC4, nppiSqr_16u_AC4IRSfs_Ctx);
impl_unary_operation_scaled!(square_u16_c4, u16, C4, nppiSqr_16u_C4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_u16_c4_in_place, u16, C4, nppiSqr_16u_C4IRSfs_Ctx);
impl_unary_operation_scaled!(square_i16_c1, i16, C1, nppiSqr_16s_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_i16_c1_in_place, i16, C1, nppiSqr_16s_C1IRSfs_Ctx);
impl_unary_operation_scaled!(square_i16_c3, i16, C3, nppiSqr_16s_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_i16_c3_in_place, i16, C3, nppiSqr_16s_C3IRSfs_Ctx);
impl_unary_operation_scaled!(square_i16_ac4, i16, AC4, nppiSqr_16s_AC4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_i16_ac4_in_place, i16, AC4, nppiSqr_16s_AC4IRSfs_Ctx);
impl_unary_operation_scaled!(square_i16_c4, i16, C4, nppiSqr_16s_C4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_i16_c4_in_place, i16, C4, nppiSqr_16s_C4IRSfs_Ctx);
impl_unary_operation!(square_f16_c1, f16, C1, nppiSqr_16f_C1R_Ctx);
impl_unary_operation_in_place!(square_f16_c1_in_place, f16, C1, nppiSqr_16f_C1IR_Ctx);
impl_unary_operation!(square_f16_c3, f16, C3, nppiSqr_16f_C3R_Ctx);
impl_unary_operation_in_place!(square_f16_c3_in_place, f16, C3, nppiSqr_16f_C3IR_Ctx);
impl_unary_operation!(square_f16_c4, f16, C4, nppiSqr_16f_C4R_Ctx);
impl_unary_operation_in_place!(square_f16_c4_in_place, f16, C4, nppiSqr_16f_C4IR_Ctx);
impl_unary_operation!(square_f32_c1, f32, C1, nppiSqr_32f_C1R_Ctx);
impl_unary_operation_in_place!(square_f32_c1_in_place, f32, C1, nppiSqr_32f_C1IR_Ctx);
impl_unary_operation!(square_f32_c3, f32, C3, nppiSqr_32f_C3R_Ctx);
impl_unary_operation_in_place!(square_f32_c3_in_place, f32, C3, nppiSqr_32f_C3IR_Ctx);
impl_unary_operation!(square_f32_ac4, f32, AC4, nppiSqr_32f_AC4R_Ctx);
impl_unary_operation_in_place!(square_f32_ac4_in_place, f32, AC4, nppiSqr_32f_AC4IR_Ctx);
impl_unary_operation!(square_f32_c4, f32, C4, nppiSqr_32f_C4R_Ctx);
impl_unary_operation_in_place!(square_f32_c4_in_place, f32, C4, nppiSqr_32f_C4IR_Ctx);

impl_unary_operation_scaled!(square_root_u8_c1, u8, C1, nppiSqrt_8u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_root_u8_c1_in_place, u8, C1, nppiSqrt_8u_C1IRSfs_Ctx);
impl_unary_operation_scaled!(square_root_u8_c3, u8, C3, nppiSqrt_8u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(square_root_u8_c3_in_place, u8, C3, nppiSqrt_8u_C3IRSfs_Ctx);
impl_unary_operation_scaled!(square_root_u8_ac4, u8, AC4, nppiSqrt_8u_AC4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_u8_ac4_in_place,
    u8,
    AC4,
    nppiSqrt_8u_AC4IRSfs_Ctx
);
impl_unary_operation_scaled!(square_root_u16_c1, u16, C1, nppiSqrt_16u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_u16_c1_in_place,
    u16,
    C1,
    nppiSqrt_16u_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(square_root_u16_c3, u16, C3, nppiSqrt_16u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_u16_c3_in_place,
    u16,
    C3,
    nppiSqrt_16u_C3IRSfs_Ctx
);
impl_unary_operation_scaled!(square_root_u16_ac4, u16, AC4, nppiSqrt_16u_AC4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_u16_ac4_in_place,
    u16,
    AC4,
    nppiSqrt_16u_AC4IRSfs_Ctx
);
impl_unary_operation_scaled!(square_root_i16_c1, i16, C1, nppiSqrt_16s_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_i16_c1_in_place,
    i16,
    C1,
    nppiSqrt_16s_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(square_root_i16_c3, i16, C3, nppiSqrt_16s_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_i16_c3_in_place,
    i16,
    C3,
    nppiSqrt_16s_C3IRSfs_Ctx
);
impl_unary_operation_scaled!(square_root_i16_ac4, i16, AC4, nppiSqrt_16s_AC4RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    square_root_i16_ac4_in_place,
    i16,
    AC4,
    nppiSqrt_16s_AC4IRSfs_Ctx
);
impl_unary_operation!(square_root_f16_c1, f16, C1, nppiSqrt_16f_C1R_Ctx);
impl_unary_operation_in_place!(square_root_f16_c1_in_place, f16, C1, nppiSqrt_16f_C1IR_Ctx);
impl_unary_operation!(square_root_f16_c3, f16, C3, nppiSqrt_16f_C3R_Ctx);
impl_unary_operation_in_place!(square_root_f16_c3_in_place, f16, C3, nppiSqrt_16f_C3IR_Ctx);
impl_unary_operation!(square_root_f16_c4, f16, C4, nppiSqrt_16f_C4R_Ctx);
impl_unary_operation_in_place!(square_root_f16_c4_in_place, f16, C4, nppiSqrt_16f_C4IR_Ctx);
impl_unary_operation!(square_root_f32_c1, f32, C1, nppiSqrt_32f_C1R_Ctx);
impl_unary_operation_in_place!(square_root_f32_c1_in_place, f32, C1, nppiSqrt_32f_C1IR_Ctx);
impl_unary_operation!(square_root_f32_c3, f32, C3, nppiSqrt_32f_C3R_Ctx);
impl_unary_operation_in_place!(square_root_f32_c3_in_place, f32, C3, nppiSqrt_32f_C3IR_Ctx);
impl_unary_operation!(square_root_f32_ac4, f32, AC4, nppiSqrt_32f_AC4R_Ctx);
impl_unary_operation_in_place!(
    square_root_f32_ac4_in_place,
    f32,
    AC4,
    nppiSqrt_32f_AC4IR_Ctx
);
impl_unary_operation!(square_root_f32_c4, f32, C4, nppiSqrt_32f_C4R_Ctx);
impl_unary_operation_in_place!(square_root_f32_c4_in_place, f32, C4, nppiSqrt_32f_C4IR_Ctx);

impl_unary_operation_scaled!(natural_logarithm_u8_c1, u8, C1, nppiLn_8u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    natural_logarithm_u8_c1_in_place,
    u8,
    C1,
    nppiLn_8u_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(natural_logarithm_u8_c3, u8, C3, nppiLn_8u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    natural_logarithm_u8_c3_in_place,
    u8,
    C3,
    nppiLn_8u_C3IRSfs_Ctx
);
impl_unary_operation_scaled!(natural_logarithm_u16_c1, u16, C1, nppiLn_16u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    natural_logarithm_u16_c1_in_place,
    u16,
    C1,
    nppiLn_16u_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(natural_logarithm_u16_c3, u16, C3, nppiLn_16u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    natural_logarithm_u16_c3_in_place,
    u16,
    C3,
    nppiLn_16u_C3IRSfs_Ctx
);
impl_unary_operation_scaled!(natural_logarithm_i16_c1, i16, C1, nppiLn_16s_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    natural_logarithm_i16_c1_in_place,
    i16,
    C1,
    nppiLn_16s_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(natural_logarithm_i16_c3, i16, C3, nppiLn_16s_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    natural_logarithm_i16_c3_in_place,
    i16,
    C3,
    nppiLn_16s_C3IRSfs_Ctx
);
impl_unary_operation!(natural_logarithm_f16_c1, f16, C1, nppiLn_16f_C1R_Ctx);
impl_unary_operation_in_place!(
    natural_logarithm_f16_c1_in_place,
    f16,
    C1,
    nppiLn_16f_C1IR_Ctx
);
impl_unary_operation!(natural_logarithm_f16_c3, f16, C3, nppiLn_16f_C3R_Ctx);
impl_unary_operation_in_place!(
    natural_logarithm_f16_c3_in_place,
    f16,
    C3,
    nppiLn_16f_C3IR_Ctx
);
impl_unary_operation!(natural_logarithm_f32_c1, f32, C1, nppiLn_32f_C1R_Ctx);
impl_unary_operation_in_place!(
    natural_logarithm_f32_c1_in_place,
    f32,
    C1,
    nppiLn_32f_C1IR_Ctx
);
impl_unary_operation!(natural_logarithm_f32_c3, f32, C3, nppiLn_32f_C3R_Ctx);
impl_unary_operation_in_place!(
    natural_logarithm_f32_c3_in_place,
    f32,
    C3,
    nppiLn_32f_C3IR_Ctx
);

impl_unary_operation_scaled!(exponential_u8_c1, u8, C1, nppiExp_8u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(exponential_u8_c1_in_place, u8, C1, nppiExp_8u_C1IRSfs_Ctx);
impl_unary_operation_scaled!(exponential_u8_c3, u8, C3, nppiExp_8u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(exponential_u8_c3_in_place, u8, C3, nppiExp_8u_C3IRSfs_Ctx);
impl_unary_operation_scaled!(exponential_u16_c1, u16, C1, nppiExp_16u_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    exponential_u16_c1_in_place,
    u16,
    C1,
    nppiExp_16u_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(exponential_u16_c3, u16, C3, nppiExp_16u_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    exponential_u16_c3_in_place,
    u16,
    C3,
    nppiExp_16u_C3IRSfs_Ctx
);
impl_unary_operation_scaled!(exponential_i16_c1, i16, C1, nppiExp_16s_C1RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    exponential_i16_c1_in_place,
    i16,
    C1,
    nppiExp_16s_C1IRSfs_Ctx
);
impl_unary_operation_scaled!(exponential_i16_c3, i16, C3, nppiExp_16s_C3RSfs_Ctx);
impl_unary_operation_scaled_in_place!(
    exponential_i16_c3_in_place,
    i16,
    C3,
    nppiExp_16s_C3IRSfs_Ctx
);
impl_unary_operation!(exponential_f32_c1, f32, C1, nppiExp_32f_C1R_Ctx);
impl_unary_operation_in_place!(exponential_f32_c1_in_place, f32, C1, nppiExp_32f_C1IR_Ctx);
impl_unary_operation!(exponential_f32_c3, f32, C3, nppiExp_32f_C3R_Ctx);
impl_unary_operation_in_place!(exponential_f32_c3_in_place, f32, C3, nppiExp_32f_C3IR_Ctx);
