use super::*;

impl_binary_operation!(add_i16, i16, nppsAdd_16s_Ctx);
impl_binary_operation!(add_u16, u16, nppsAdd_16u_Ctx);
impl_binary_operation!(add_u32, u32, nppsAdd_32u_Ctx);
impl_binary_operation!(add_f32, f32, nppsAdd_32f_Ctx);
impl_binary_operation!(add_f64, f64, nppsAdd_64f_Ctx);
impl_binary_operation!(add_f32_complex, Complex32, nppsAdd_32fc_Ctx);
impl_binary_operation!(add_f64_complex, Complex64, nppsAdd_64fc_Ctx);
impl_reversed_binary_operation!(subtract_i16, i16, nppsSub_16s_Ctx);
impl_reversed_binary_operation!(subtract_f32, f32, nppsSub_32f_Ctx);
impl_reversed_binary_operation!(subtract_f64, f64, nppsSub_64f_Ctx);
impl_reversed_binary_operation!(subtract_f32_complex, Complex32, nppsSub_32fc_Ctx);
impl_reversed_binary_operation!(subtract_f64_complex, Complex64, nppsSub_64fc_Ctx);
impl_binary_operation!(multiply_i16, i16, nppsMul_16s_Ctx);
impl_binary_operation!(multiply_f32, f32, nppsMul_32f_Ctx);
impl_binary_operation!(multiply_f64, f64, nppsMul_64f_Ctx);
impl_binary_operation!(multiply_f32_complex, Complex32, nppsMul_32fc_Ctx);
impl_binary_operation!(multiply_f64_complex, Complex64, nppsMul_64fc_Ctx);
impl_reversed_binary_operation!(divide_f32, f32, nppsDiv_32f_Ctx);
impl_reversed_binary_operation!(divide_f64, f64, nppsDiv_64f_Ctx);
impl_reversed_binary_operation!(divide_f32_complex, Complex32, nppsDiv_32fc_Ctx);
impl_reversed_binary_operation!(divide_f64_complex, Complex64, nppsDiv_64fc_Ctx);
impl_binary_operation!(and_u8, u8, nppsAnd_8u_Ctx);
impl_binary_operation!(and_u16, u16, nppsAnd_16u_Ctx);
impl_binary_operation!(and_u32, u32, nppsAnd_32u_Ctx);
impl_binary_operation!(or_u8, u8, nppsOr_8u_Ctx);
impl_binary_operation!(or_u16, u16, nppsOr_16u_Ctx);
impl_binary_operation!(or_u32, u32, nppsOr_32u_Ctx);
impl_binary_operation!(xor_u8, u8, nppsXor_8u_Ctx);
impl_binary_operation!(xor_u16, u16, nppsXor_16u_Ctx);
impl_binary_operation!(xor_u32, u32, nppsXor_32u_Ctx);

impl_mixed_binary_operation!(add_u8_to_u16, u8, u16, nppsAdd_8u16u_Ctx);
impl_mixed_binary_operation!(add_i16_to_f32, i16, f32, nppsAdd_16s32f_Ctx);
impl_reversed_mixed_binary_operation!(subtract_i16_to_f32, i16, f32, nppsSub_16s32f_Ctx);
impl_mixed_binary_operation!(multiply_u8_to_u16, u8, u16, nppsMul_8u16u_Ctx);
impl_mixed_binary_operation!(multiply_i16_to_f32, i16, f32, nppsMul_16s32f_Ctx);
impl_heterogeneous_binary_operation!(
    multiply_f32_with_f32_complex,
    f32,
    Complex32,
    Complex32,
    nppsMul_32f32fc_Ctx
);

impl_scaled_binary_operation!(add_u8_scaled, u8, nppsAdd_8u_Sfs_Ctx);
impl_scaled_binary_operation!(add_u16_scaled, u16, nppsAdd_16u_Sfs_Ctx);
impl_scaled_binary_operation!(add_i16_scaled, i16, nppsAdd_16s_Sfs_Ctx);
impl_scaled_binary_operation!(add_i32_scaled, i32, nppsAdd_32s_Sfs_Ctx);
impl_scaled_binary_operation!(add_i64_scaled, i64, nppsAdd_64s_Sfs_Ctx);
impl_scaled_binary_operation!(add_i16_complex_scaled, ComplexI16, nppsAdd_16sc_Sfs_Ctx);
impl_scaled_binary_operation!(add_i32_complex_scaled, ComplexI32, nppsAdd_32sc_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(subtract_u8_scaled, u8, nppsSub_8u_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(subtract_u16_scaled, u16, nppsSub_16u_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(subtract_i16_scaled, i16, nppsSub_16s_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(subtract_i32_scaled, i32, nppsSub_32s_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(
    subtract_i16_complex_scaled,
    ComplexI16,
    nppsSub_16sc_Sfs_Ctx
);
impl_reversed_scaled_binary_operation!(
    subtract_i32_complex_scaled,
    ComplexI32,
    nppsSub_32sc_Sfs_Ctx
);
impl_scaled_binary_operation!(multiply_u8_scaled, u8, nppsMul_8u_Sfs_Ctx);
impl_scaled_binary_operation!(multiply_u16_scaled, u16, nppsMul_16u_Sfs_Ctx);
impl_scaled_binary_operation!(multiply_i16_scaled, i16, nppsMul_16s_Sfs_Ctx);
impl_scaled_binary_operation!(multiply_i32_scaled, i32, nppsMul_32s_Sfs_Ctx);
impl_scaled_binary_operation!(
    multiply_i16_complex_scaled,
    ComplexI16,
    nppsMul_16sc_Sfs_Ctx
);
impl_scaled_binary_operation!(
    multiply_i32_complex_scaled,
    ComplexI32,
    nppsMul_32sc_Sfs_Ctx
);
impl_scaled_heterogeneous_binary_operation!(
    multiply_u16_with_i16_scaled,
    u16,
    i16,
    i16,
    nppsMul_16u16s_Sfs_Ctx
);
impl_scaled_mixed_binary_operation!(multiply_i16_to_i32_scaled, i16, i32, nppsMul_16s32s_Sfs_Ctx);
impl_scaled_heterogeneous_binary_operation!(
    multiply_i32_with_i32_complex_scaled,
    i32,
    ComplexI32,
    ComplexI32,
    nppsMul_32s32sc_Sfs_Ctx
);
impl_scaled_binary_operation!(multiply_low_i32_scaled, i32, nppsMul_Low_32s_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(divide_u8_scaled, u8, nppsDiv_8u_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(divide_u16_scaled, u16, nppsDiv_16u_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(divide_i16_scaled, i16, nppsDiv_16s_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(divide_i32_scaled, i32, nppsDiv_32s_Sfs_Ctx);
impl_reversed_scaled_binary_operation!(divide_i16_complex_scaled, ComplexI16, nppsDiv_16sc_Sfs_Ctx);
impl_reversed_scaled_heterogeneous_binary_operation!(
    divide_i32_by_i16_scaled,
    i32,
    i16,
    i16,
    nppsDiv_32s16s_Sfs_Ctx
);
impl_reversed_scaled_round_binary_operation!(divide_u8_scaled_round, u8, nppsDiv_Round_8u_Sfs_Ctx);
impl_reversed_scaled_round_binary_operation!(
    divide_u16_scaled_round,
    u16,
    nppsDiv_Round_16u_Sfs_Ctx
);
impl_reversed_scaled_round_binary_operation!(
    divide_i16_scaled_round,
    i16,
    nppsDiv_Round_16s_Sfs_Ctx
);

impl_binary_operation_in_place!(add_i16_in_place, i16, nppsAdd_16s_I_Ctx);
impl_binary_operation_in_place!(add_f32_in_place, f32, nppsAdd_32f_I_Ctx);
impl_binary_operation_in_place!(add_f64_in_place, f64, nppsAdd_64f_I_Ctx);
impl_binary_operation_in_place!(add_f32_complex_in_place, Complex32, nppsAdd_32fc_I_Ctx);
impl_binary_operation_in_place!(add_f64_complex_in_place, Complex64, nppsAdd_64fc_I_Ctx);
impl_binary_operation_in_place!(subtract_i16_in_place, i16, nppsSub_16s_I_Ctx);
impl_binary_operation_in_place!(subtract_f32_in_place, f32, nppsSub_32f_I_Ctx);
impl_binary_operation_in_place!(subtract_f64_in_place, f64, nppsSub_64f_I_Ctx);
impl_binary_operation_in_place!(subtract_f32_complex_in_place, Complex32, nppsSub_32fc_I_Ctx);
impl_binary_operation_in_place!(subtract_f64_complex_in_place, Complex64, nppsSub_64fc_I_Ctx);
impl_binary_operation_in_place!(multiply_i16_in_place, i16, nppsMul_16s_I_Ctx);
impl_binary_operation_in_place!(multiply_f32_in_place, f32, nppsMul_32f_I_Ctx);
impl_binary_operation_in_place!(multiply_f64_in_place, f64, nppsMul_64f_I_Ctx);
impl_binary_operation_in_place!(multiply_f32_complex_in_place, Complex32, nppsMul_32fc_I_Ctx);
impl_binary_operation_in_place!(multiply_f64_complex_in_place, Complex64, nppsMul_64fc_I_Ctx);
impl_binary_operation_in_place!(divide_f32_in_place, f32, nppsDiv_32f_I_Ctx);
impl_binary_operation_in_place!(divide_f64_in_place, f64, nppsDiv_64f_I_Ctx);
impl_binary_operation_in_place!(divide_f32_complex_in_place, Complex32, nppsDiv_32fc_I_Ctx);
impl_binary_operation_in_place!(divide_f64_complex_in_place, Complex64, nppsDiv_64fc_I_Ctx);
impl_binary_operation_in_place!(and_u8_in_place, u8, nppsAnd_8u_I_Ctx);
impl_binary_operation_in_place!(and_u16_in_place, u16, nppsAnd_16u_I_Ctx);
impl_binary_operation_in_place!(and_u32_in_place, u32, nppsAnd_32u_I_Ctx);
impl_binary_operation_in_place!(or_u8_in_place, u8, nppsOr_8u_I_Ctx);
impl_binary_operation_in_place!(or_u16_in_place, u16, nppsOr_16u_I_Ctx);
impl_binary_operation_in_place!(or_u32_in_place, u32, nppsOr_32u_I_Ctx);
impl_binary_operation_in_place!(xor_u8_in_place, u8, nppsXor_8u_I_Ctx);
impl_binary_operation_in_place!(xor_u16_in_place, u16, nppsXor_16u_I_Ctx);
impl_binary_operation_in_place!(xor_u32_in_place, u32, nppsXor_32u_I_Ctx);
impl_mixed_binary_operation_in_place!(add_i16_to_i32_in_place, i16, i32, nppsAdd_16s32s_I_Ctx);
impl_heterogeneous_binary_operation_in_place!(
    multiply_f32_complex_by_f32_in_place,
    f32,
    Complex32,
    nppsMul_32f32fc_I_Ctx
);

impl_scaled_binary_operation_in_place!(add_u8_scaled_in_place, u8, nppsAdd_8u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(add_u16_scaled_in_place, u16, nppsAdd_16u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(add_i16_scaled_in_place, i16, nppsAdd_16s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(add_i32_scaled_in_place, i32, nppsAdd_32s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(
    add_i16_complex_scaled_in_place,
    ComplexI16,
    nppsAdd_16sc_ISfs_Ctx
);
impl_scaled_binary_operation_in_place!(
    add_i32_complex_scaled_in_place,
    ComplexI32,
    nppsAdd_32sc_ISfs_Ctx
);
impl_scaled_binary_operation_in_place!(subtract_u8_scaled_in_place, u8, nppsSub_8u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(subtract_u16_scaled_in_place, u16, nppsSub_16u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(subtract_i16_scaled_in_place, i16, nppsSub_16s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(subtract_i32_scaled_in_place, i32, nppsSub_32s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(
    subtract_i16_complex_scaled_in_place,
    ComplexI16,
    nppsSub_16sc_ISfs_Ctx
);
impl_scaled_binary_operation_in_place!(
    subtract_i32_complex_scaled_in_place,
    ComplexI32,
    nppsSub_32sc_ISfs_Ctx
);
impl_scaled_binary_operation_in_place!(multiply_u8_scaled_in_place, u8, nppsMul_8u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(multiply_u16_scaled_in_place, u16, nppsMul_16u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(multiply_i16_scaled_in_place, i16, nppsMul_16s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(multiply_i32_scaled_in_place, i32, nppsMul_32s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(
    multiply_i16_complex_scaled_in_place,
    ComplexI16,
    nppsMul_16sc_ISfs_Ctx
);
impl_scaled_binary_operation_in_place!(
    multiply_i32_complex_scaled_in_place,
    ComplexI32,
    nppsMul_32sc_ISfs_Ctx
);
impl_scaled_heterogeneous_binary_operation_in_place!(
    multiply_i32_complex_by_i32_scaled_in_place,
    i32,
    ComplexI32,
    nppsMul_32s32sc_ISfs_Ctx
);
impl_scaled_binary_operation_in_place!(divide_u8_scaled_in_place, u8, nppsDiv_8u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(divide_u16_scaled_in_place, u16, nppsDiv_16u_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(divide_i16_scaled_in_place, i16, nppsDiv_16s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(divide_i32_scaled_in_place, i32, nppsDiv_32s_ISfs_Ctx);
impl_scaled_binary_operation_in_place!(
    divide_i16_complex_scaled_in_place,
    ComplexI16,
    nppsDiv_16sc_ISfs_Ctx
);
impl_scaled_round_binary_operation_in_place!(
    divide_u8_scaled_round_in_place,
    u8,
    nppsDiv_Round_8u_ISfs_Ctx
);
impl_scaled_round_binary_operation_in_place!(
    divide_u16_scaled_round_in_place,
    u16,
    nppsDiv_Round_16u_ISfs_Ctx
);
impl_scaled_round_binary_operation_in_place!(
    divide_i16_scaled_round_in_place,
    i16,
    nppsDiv_Round_16s_ISfs_Ctx
);

impl_destination_update_operation!(add_product_f32, f32, nppsAddProduct_32f_Ctx);
impl_destination_update_operation!(add_product_f64, f64, nppsAddProduct_64f_Ctx);
impl_destination_update_operation!(add_product_f32_complex, Complex32, nppsAddProduct_32fc_Ctx);
impl_destination_update_operation!(add_product_f64_complex, Complex64, nppsAddProduct_64fc_Ctx);
impl_scaled_destination_update_operation!(
    add_product_i16_scaled,
    i16,
    i16,
    nppsAddProduct_16s_Sfs_Ctx
);
impl_scaled_destination_update_operation!(
    add_product_i32_scaled,
    i32,
    i32,
    nppsAddProduct_32s_Sfs_Ctx
);
impl_scaled_destination_update_operation!(
    add_product_i16_to_i32_scaled,
    i16,
    i32,
    nppsAddProduct_16s32s_Sfs_Ctx
);

impl_unary_operation!(absolute_i16, i16, nppsAbs_16s_Ctx);
impl_unary_operation!(absolute_i32, i32, nppsAbs_32s_Ctx);
impl_unary_operation!(absolute_f32, f32, nppsAbs_32f_Ctx);
impl_unary_operation!(absolute_f64, f64, nppsAbs_64f_Ctx);
impl_unary_operation!(exponent_f32, f32, nppsExp_32f_Ctx);
impl_unary_operation!(exponent_f64, f64, nppsExp_64f_Ctx);
impl_mixed_unary_operation!(exponent_f32_to_f64, f32, f64, nppsExp_32f64f_Ctx);
impl_scaled_unary_operation!(exponent_i16_scaled, i16, nppsExp_16s_Sfs_Ctx);
impl_scaled_unary_operation!(exponent_i32_scaled, i32, nppsExp_32s_Sfs_Ctx);
impl_scaled_unary_operation!(exponent_i64_scaled, i64, nppsExp_64s_Sfs_Ctx);
impl_unary_operation!(natural_logarithm_f32, f32, nppsLn_32f_Ctx);
impl_unary_operation!(natural_logarithm_f64, f64, nppsLn_64f_Ctx);
impl_mixed_unary_operation!(natural_logarithm_f64_to_f32, f64, f32, nppsLn_64f32f_Ctx);
impl_scaled_unary_operation!(natural_logarithm_i16_scaled, i16, nppsLn_16s_Sfs_Ctx);
impl_scaled_unary_operation!(natural_logarithm_i32_scaled, i32, nppsLn_32s_Sfs_Ctx);
impl_scaled_mixed_unary_operation!(
    natural_logarithm_i32_to_i16_scaled,
    i32,
    i16,
    nppsLn_32s16s_Sfs_Ctx
);
impl_scaled_unary_operation!(ten_times_log10_i32_scaled, i32, npps10Log10_32s_Sfs_Ctx);
impl_unary_operation!(arctangent_f32, f32, nppsArctan_32f_Ctx);
impl_unary_operation!(arctangent_f64, f64, nppsArctan_64f_Ctx);
impl_unary_operation!(cube_root_f32, f32, nppsCubrt_32f_Ctx);
impl_scaled_mixed_unary_operation!(
    cube_root_i32_to_i16_scaled,
    i32,
    i16,
    nppsCubrt_32s16s_Sfs_Ctx
);
impl_unary_operation!(not_u8, u8, nppsNot_8u_Ctx);
impl_unary_operation!(not_u16, u16, nppsNot_16u_Ctx);
impl_unary_operation!(not_u32, u32, nppsNot_32u_Ctx);
impl_unary_operation!(square_f32, f32, nppsSqr_32f_Ctx);
impl_unary_operation!(square_f64, f64, nppsSqr_64f_Ctx);
impl_unary_operation!(square_f32_complex, Complex32, nppsSqr_32fc_Ctx);
impl_unary_operation!(square_f64_complex, Complex64, nppsSqr_64fc_Ctx);
impl_scaled_unary_operation!(square_u8_scaled, u8, nppsSqr_8u_Sfs_Ctx);
impl_scaled_unary_operation!(square_u16_scaled, u16, nppsSqr_16u_Sfs_Ctx);
impl_scaled_unary_operation!(square_i16_scaled, i16, nppsSqr_16s_Sfs_Ctx);
impl_scaled_unary_operation!(square_i16_complex_scaled, ComplexI16, nppsSqr_16sc_Sfs_Ctx);
impl_unary_operation!(square_root_f32, f32, nppsSqrt_32f_Ctx);
impl_unary_operation!(square_root_f64, f64, nppsSqrt_64f_Ctx);
impl_unary_operation!(square_root_f32_complex, Complex32, nppsSqrt_32fc_Ctx);
impl_unary_operation!(square_root_f64_complex, Complex64, nppsSqrt_64fc_Ctx);
impl_scaled_unary_operation!(square_root_u8_scaled, u8, nppsSqrt_8u_Sfs_Ctx);
impl_scaled_unary_operation!(square_root_u16_scaled, u16, nppsSqrt_16u_Sfs_Ctx);
impl_scaled_unary_operation!(square_root_i16_scaled, i16, nppsSqrt_16s_Sfs_Ctx);
impl_scaled_unary_operation!(
    square_root_i16_complex_scaled,
    ComplexI16,
    nppsSqrt_16sc_Sfs_Ctx
);
impl_scaled_unary_operation!(square_root_i64_scaled, i64, nppsSqrt_64s_Sfs_Ctx);
impl_scaled_mixed_unary_operation!(
    square_root_i32_to_i16_scaled,
    i32,
    i16,
    nppsSqrt_32s16s_Sfs_Ctx
);
impl_scaled_mixed_unary_operation!(
    square_root_i64_to_i16_scaled,
    i64,
    i16,
    nppsSqrt_64s16s_Sfs_Ctx
);

impl_unary_operation_in_place!(absolute_i16_in_place, i16, nppsAbs_16s_I_Ctx);
impl_unary_operation_in_place!(absolute_i32_in_place, i32, nppsAbs_32s_I_Ctx);
impl_unary_operation_in_place!(absolute_f32_in_place, f32, nppsAbs_32f_I_Ctx);
impl_unary_operation_in_place!(absolute_f64_in_place, f64, nppsAbs_64f_I_Ctx);
impl_unary_operation_in_place!(exponent_f32_in_place, f32, nppsExp_32f_I_Ctx);
impl_unary_operation_in_place!(exponent_f64_in_place, f64, nppsExp_64f_I_Ctx);
impl_scaled_unary_operation_in_place!(exponent_i16_scaled_in_place, i16, nppsExp_16s_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(exponent_i32_scaled_in_place, i32, nppsExp_32s_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(exponent_i64_scaled_in_place, i64, nppsExp_64s_ISfs_Ctx);
impl_unary_operation_in_place!(natural_logarithm_f32_in_place, f32, nppsLn_32f_I_Ctx);
impl_unary_operation_in_place!(natural_logarithm_f64_in_place, f64, nppsLn_64f_I_Ctx);
impl_scaled_unary_operation_in_place!(
    natural_logarithm_i16_scaled_in_place,
    i16,
    nppsLn_16s_ISfs_Ctx
);
impl_scaled_unary_operation_in_place!(
    natural_logarithm_i32_scaled_in_place,
    i32,
    nppsLn_32s_ISfs_Ctx
);
impl_scaled_unary_operation_in_place!(
    ten_times_log10_i32_scaled_in_place,
    i32,
    npps10Log10_32s_ISfs_Ctx
);
impl_unary_operation_in_place!(arctangent_f32_in_place, f32, nppsArctan_32f_I_Ctx);
impl_unary_operation_in_place!(arctangent_f64_in_place, f64, nppsArctan_64f_I_Ctx);
impl_unary_operation_in_place!(not_u8_in_place, u8, nppsNot_8u_I_Ctx);
impl_unary_operation_in_place!(not_u16_in_place, u16, nppsNot_16u_I_Ctx);
impl_unary_operation_in_place!(not_u32_in_place, u32, nppsNot_32u_I_Ctx);
impl_unary_operation_in_place!(square_f32_in_place, f32, nppsSqr_32f_I_Ctx);
impl_unary_operation_in_place!(square_f64_in_place, f64, nppsSqr_64f_I_Ctx);
impl_unary_operation_in_place!(square_f32_complex_in_place, Complex32, nppsSqr_32fc_I_Ctx);
impl_unary_operation_in_place!(square_f64_complex_in_place, Complex64, nppsSqr_64fc_I_Ctx);
impl_scaled_unary_operation_in_place!(square_u8_scaled_in_place, u8, nppsSqr_8u_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(square_u16_scaled_in_place, u16, nppsSqr_16u_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(square_i16_scaled_in_place, i16, nppsSqr_16s_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(
    square_i16_complex_scaled_in_place,
    ComplexI16,
    nppsSqr_16sc_ISfs_Ctx
);
impl_unary_operation_in_place!(square_root_f32_in_place, f32, nppsSqrt_32f_I_Ctx);
impl_unary_operation_in_place!(square_root_f64_in_place, f64, nppsSqrt_64f_I_Ctx);
impl_unary_operation_in_place!(
    square_root_f32_complex_in_place,
    Complex32,
    nppsSqrt_32fc_I_Ctx
);
impl_unary_operation_in_place!(
    square_root_f64_complex_in_place,
    Complex64,
    nppsSqrt_64fc_I_Ctx
);
impl_scaled_unary_operation_in_place!(square_root_u8_scaled_in_place, u8, nppsSqrt_8u_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(square_root_u16_scaled_in_place, u16, nppsSqrt_16u_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(square_root_i16_scaled_in_place, i16, nppsSqrt_16s_ISfs_Ctx);
impl_scaled_unary_operation_in_place!(
    square_root_i16_complex_scaled_in_place,
    ComplexI16,
    nppsSqrt_16sc_ISfs_Ctx
);
impl_scaled_unary_operation_in_place!(square_root_i64_scaled_in_place, i64, nppsSqrt_64s_ISfs_Ctx);

impl_normalize!(normalize_f32, f32, nppsNormalize_32f_Ctx);
impl_normalize!(normalize_f64, f64, nppsNormalize_64f_Ctx);
impl_complex_normalize!(
    normalize_f32_complex,
    Complex32,
    f32,
    nppsNormalize_32fc_Ctx
);
impl_complex_normalize!(
    normalize_f64_complex,
    Complex64,
    f64,
    nppsNormalize_64fc_Ctx
);
impl_scaled_normalize!(normalize_i16_scaled, i16, nppsNormalize_16s_Sfs_Ctx);
impl_scaled_normalize!(
    normalize_i16_complex_scaled,
    ComplexI16,
    nppsNormalize_16sc_Sfs_Ctx
);

impl_parameterized_unary_operation_in_place!(cauchy_f32_in_place, f32, f32, nppsCauchy_32f_I_Ctx);
impl_parameterized_unary_operation_in_place!(
    cauchy_derivative_f32_in_place,
    f32,
    f32,
    nppsCauchyD_32f_I_Ctx
);

macro_rules! impl_signal_binary_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                left: &SignalView<'_, Self>,
                right: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    left: &SignalView<'_, Self>,
                    right: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, left, right, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            left: &SignalView<'_, T>,
            right: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, left, right, destination)
        }
    };
}

macro_rules! impl_signal_binary_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, signal, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, signal, destination)
        }
    };
}

macro_rules! impl_signal_scaled_binary_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                left: &SignalView<'_, Self>,
                right: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    left: &SignalView<'_, Self>,
                    right: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, left, right, destination, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            left: &SignalView<'_, T>,
            right: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, left, right, destination, scale_factor)
        }
    };
}

macro_rules! impl_signal_scaled_binary_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, signal, destination, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, signal, destination, scale_factor)
        }
    };
}

macro_rules! impl_signal_scaled_round_binary_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                left: &SignalView<'_, Self>,
                right: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    left: &SignalView<'_, Self>,
                    right: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    $direct(stream_context, left, right, destination, scale_factor, round_mode)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            left: &SignalView<'_, T>,
            right: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            T::$method(stream_context, left, right, destination, scale_factor, round_mode)
        }
    };
}

macro_rules! impl_signal_scaled_round_binary_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    $direct(stream_context, signal, destination, scale_factor, round_mode)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            T::$method(stream_context, signal, destination, scale_factor, round_mode)
        }
    };
}

impl_signal_binary_dispatch!(Add, add, add, [
    i16 => add_i16,
    u16 => add_u16,
    u32 => add_u32,
    f32 => add_f32,
    f64 => add_f64,
    Complex32 => add_f32_complex,
    Complex64 => add_f64_complex
]);
impl_signal_binary_dispatch!(Subtract, subtract, subtract, [
    i16 => subtract_i16,
    f32 => subtract_f32,
    f64 => subtract_f64,
    Complex32 => subtract_f32_complex,
    Complex64 => subtract_f64_complex
]);
impl_signal_binary_dispatch!(Multiply, multiply, multiply, [
    i16 => multiply_i16,
    f32 => multiply_f32,
    f64 => multiply_f64,
    Complex32 => multiply_f32_complex,
    Complex64 => multiply_f64_complex
]);
impl_signal_binary_dispatch!(Divide, divide, divide, [
    f32 => divide_f32,
    f64 => divide_f64,
    Complex32 => divide_f32_complex,
    Complex64 => divide_f64_complex
]);
impl_signal_binary_dispatch!(And, and, and, [
    u8 => and_u8,
    u16 => and_u16,
    u32 => and_u32
]);
impl_signal_binary_dispatch!(Or, or, or, [
    u8 => or_u8,
    u16 => or_u16,
    u32 => or_u32
]);
impl_signal_binary_dispatch!(Xor, xor, xor, [
    u8 => xor_u8,
    u16 => xor_u16,
    u32 => xor_u32
]);

impl_signal_binary_in_place_dispatch!(AddInPlace, add_in_place, add_in_place, [
    i16 => add_i16_in_place,
    f32 => add_f32_in_place,
    f64 => add_f64_in_place,
    Complex32 => add_f32_complex_in_place,
    Complex64 => add_f64_complex_in_place
]);
impl_signal_binary_in_place_dispatch!(SubtractInPlace, subtract_in_place, subtract_in_place, [
    i16 => subtract_i16_in_place,
    f32 => subtract_f32_in_place,
    f64 => subtract_f64_in_place,
    Complex32 => subtract_f32_complex_in_place,
    Complex64 => subtract_f64_complex_in_place
]);
impl_signal_binary_in_place_dispatch!(MultiplyInPlace, multiply_in_place, multiply_in_place, [
    i16 => multiply_i16_in_place,
    f32 => multiply_f32_in_place,
    f64 => multiply_f64_in_place,
    Complex32 => multiply_f32_complex_in_place,
    Complex64 => multiply_f64_complex_in_place
]);
impl_signal_binary_in_place_dispatch!(DivideInPlace, divide_in_place, divide_in_place, [
    f32 => divide_f32_in_place,
    f64 => divide_f64_in_place,
    Complex32 => divide_f32_complex_in_place,
    Complex64 => divide_f64_complex_in_place
]);
impl_signal_binary_in_place_dispatch!(AndInPlace, and_in_place, and_in_place, [
    u8 => and_u8_in_place,
    u16 => and_u16_in_place,
    u32 => and_u32_in_place
]);
impl_signal_binary_in_place_dispatch!(OrInPlace, or_in_place, or_in_place, [
    u8 => or_u8_in_place,
    u16 => or_u16_in_place,
    u32 => or_u32_in_place
]);
impl_signal_binary_in_place_dispatch!(XorInPlace, xor_in_place, xor_in_place, [
    u8 => xor_u8_in_place,
    u16 => xor_u16_in_place,
    u32 => xor_u32_in_place
]);

impl_signal_scaled_binary_dispatch!(AddScaled, add_scaled, add_scaled, [
    u8 => add_u8_scaled,
    u16 => add_u16_scaled,
    i16 => add_i16_scaled,
    i32 => add_i32_scaled,
    i64 => add_i64_scaled,
    ComplexI16 => add_i16_complex_scaled,
    ComplexI32 => add_i32_complex_scaled
]);
impl_signal_scaled_binary_dispatch!(SubtractScaled, subtract_scaled, subtract_scaled, [
    u8 => subtract_u8_scaled,
    u16 => subtract_u16_scaled,
    i16 => subtract_i16_scaled,
    i32 => subtract_i32_scaled,
    ComplexI16 => subtract_i16_complex_scaled,
    ComplexI32 => subtract_i32_complex_scaled
]);
impl_signal_scaled_binary_dispatch!(MultiplyScaled, multiply_scaled, multiply_scaled, [
    u8 => multiply_u8_scaled,
    u16 => multiply_u16_scaled,
    i16 => multiply_i16_scaled,
    i32 => multiply_i32_scaled,
    ComplexI16 => multiply_i16_complex_scaled,
    ComplexI32 => multiply_i32_complex_scaled
]);
impl_signal_scaled_binary_dispatch!(DivideScaled, divide_scaled, divide_scaled, [
    u8 => divide_u8_scaled,
    u16 => divide_u16_scaled,
    i16 => divide_i16_scaled,
    i32 => divide_i32_scaled,
    ComplexI16 => divide_i16_complex_scaled
]);
impl_signal_scaled_round_binary_dispatch!(DivideScaledRound, divide_scaled_round, divide_scaled_round, [
    u8 => divide_u8_scaled_round,
    u16 => divide_u16_scaled_round,
    i16 => divide_i16_scaled_round
]);

impl_signal_scaled_binary_in_place_dispatch!(AddScaledInPlace, add_scaled_in_place, add_scaled_in_place, [
    u8 => add_u8_scaled_in_place,
    u16 => add_u16_scaled_in_place,
    i16 => add_i16_scaled_in_place,
    i32 => add_i32_scaled_in_place,
    ComplexI16 => add_i16_complex_scaled_in_place,
    ComplexI32 => add_i32_complex_scaled_in_place
]);
impl_signal_scaled_binary_in_place_dispatch!(SubtractScaledInPlace, subtract_scaled_in_place, subtract_scaled_in_place, [
    u8 => subtract_u8_scaled_in_place,
    u16 => subtract_u16_scaled_in_place,
    i16 => subtract_i16_scaled_in_place,
    i32 => subtract_i32_scaled_in_place,
    ComplexI16 => subtract_i16_complex_scaled_in_place,
    ComplexI32 => subtract_i32_complex_scaled_in_place
]);
impl_signal_scaled_binary_in_place_dispatch!(MultiplyScaledInPlace, multiply_scaled_in_place, multiply_scaled_in_place, [
    u8 => multiply_u8_scaled_in_place,
    u16 => multiply_u16_scaled_in_place,
    i16 => multiply_i16_scaled_in_place,
    i32 => multiply_i32_scaled_in_place,
    ComplexI16 => multiply_i16_complex_scaled_in_place,
    ComplexI32 => multiply_i32_complex_scaled_in_place
]);
impl_signal_scaled_binary_in_place_dispatch!(DivideScaledInPlace, divide_scaled_in_place, divide_scaled_in_place, [
    u8 => divide_u8_scaled_in_place,
    u16 => divide_u16_scaled_in_place,
    i16 => divide_i16_scaled_in_place,
    i32 => divide_i32_scaled_in_place,
    ComplexI16 => divide_i16_complex_scaled_in_place
]);
impl_signal_scaled_round_binary_in_place_dispatch!(DivideScaledRoundInPlace, divide_scaled_round_in_place, divide_scaled_round_in_place, [
    u8 => divide_u8_scaled_round_in_place,
    u16 => divide_u16_scaled_round_in_place,
    i16 => divide_i16_scaled_round_in_place
]);
