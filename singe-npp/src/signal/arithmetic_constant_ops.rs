use super::*;

impl_constant_operation!(add_constant_f32, f32, nppsAddC_32f_Ctx);
impl_constant_operation!(add_constant_f64, f64, nppsAddC_64f_Ctx);
impl_constant_operation!(add_constant_f32_complex, Complex32, nppsAddC_32fc_Ctx);
impl_constant_operation!(add_constant_f64_complex, Complex64, nppsAddC_64fc_Ctx);
impl_constant_operation!(subtract_constant_f32, f32, nppsSubC_32f_Ctx);
impl_constant_operation!(subtract_constant_f64, f64, nppsSubC_64f_Ctx);
impl_constant_operation!(subtract_constant_f32_complex, Complex32, nppsSubC_32fc_Ctx);
impl_constant_operation!(subtract_constant_f64_complex, Complex64, nppsSubC_64fc_Ctx);
impl_constant_operation!(subtract_from_constant_f32, f32, nppsSubCRev_32f_Ctx);
impl_constant_operation!(subtract_from_constant_f64, f64, nppsSubCRev_64f_Ctx);
impl_constant_operation!(
    subtract_from_constant_f32_complex,
    Complex32,
    nppsSubCRev_32fc_Ctx
);
impl_constant_operation!(
    subtract_from_constant_f64_complex,
    Complex64,
    nppsSubCRev_64fc_Ctx
);
impl_constant_operation!(multiply_constant_f32, f32, nppsMulC_32f_Ctx);
impl_constant_operation!(multiply_constant_f64, f64, nppsMulC_64f_Ctx);
impl_constant_operation!(multiply_constant_f32_complex, Complex32, nppsMulC_32fc_Ctx);
impl_constant_operation!(multiply_constant_f64_complex, Complex64, nppsMulC_64fc_Ctx);
impl_constant_operation!(divide_constant_f32, f32, nppsDivC_32f_Ctx);
impl_constant_operation!(divide_constant_f64, f64, nppsDivC_64f_Ctx);
impl_constant_operation!(divide_constant_f32_complex, Complex32, nppsDivC_32fc_Ctx);
impl_constant_operation!(divide_constant_f64_complex, Complex64, nppsDivC_64fc_Ctx);
impl_constant_operation!(divide_into_constant_u16, u16, nppsDivCRev_16u_Ctx);
impl_constant_operation!(divide_into_constant_f32, f32, nppsDivCRev_32f_Ctx);
impl_constant_operation!(and_constant_u8, u8, nppsAndC_8u_Ctx);
impl_constant_operation!(and_constant_u16, u16, nppsAndC_16u_Ctx);
impl_constant_operation!(and_constant_u32, u32, nppsAndC_32u_Ctx);
impl_constant_operation!(or_constant_u8, u8, nppsOrC_8u_Ctx);
impl_constant_operation!(or_constant_u16, u16, nppsOrC_16u_Ctx);
impl_constant_operation!(or_constant_u32, u32, nppsOrC_32u_Ctx);
impl_constant_operation!(xor_constant_u8, u8, nppsXorC_8u_Ctx);
impl_constant_operation!(xor_constant_u16, u16, nppsXorC_16u_Ctx);
impl_constant_operation!(xor_constant_u32, u32, nppsXorC_32u_Ctx);

impl_scaled_constant_operation!(add_constant_u8_scaled, u8, nppsAddC_8u_Sfs_Ctx);
impl_scaled_constant_operation!(add_constant_u16_scaled, u16, nppsAddC_16u_Sfs_Ctx);
impl_scaled_constant_operation!(add_constant_i16_scaled, i16, nppsAddC_16s_Sfs_Ctx);
impl_scaled_constant_operation!(add_constant_i32_scaled, i32, nppsAddC_32s_Sfs_Ctx);
impl_scaled_constant_operation!(
    add_constant_i16_complex_scaled,
    ComplexI16,
    nppsAddC_16sc_Sfs_Ctx
);
impl_scaled_constant_operation!(
    add_constant_i32_complex_scaled,
    ComplexI32,
    nppsAddC_32sc_Sfs_Ctx
);
impl_scaled_constant_operation!(subtract_constant_u8_scaled, u8, nppsSubC_8u_Sfs_Ctx);
impl_scaled_constant_operation!(subtract_constant_u16_scaled, u16, nppsSubC_16u_Sfs_Ctx);
impl_scaled_constant_operation!(subtract_constant_i16_scaled, i16, nppsSubC_16s_Sfs_Ctx);
impl_scaled_constant_operation!(subtract_constant_i32_scaled, i32, nppsSubC_32s_Sfs_Ctx);
impl_scaled_constant_operation!(
    subtract_constant_i16_complex_scaled,
    ComplexI16,
    nppsSubC_16sc_Sfs_Ctx
);
impl_scaled_constant_operation!(
    subtract_constant_i32_complex_scaled,
    ComplexI32,
    nppsSubC_32sc_Sfs_Ctx
);
impl_scaled_constant_operation!(subtract_from_constant_u8_scaled, u8, nppsSubCRev_8u_Sfs_Ctx);
impl_scaled_constant_operation!(
    subtract_from_constant_u16_scaled,
    u16,
    nppsSubCRev_16u_Sfs_Ctx
);
impl_scaled_constant_operation!(
    subtract_from_constant_i16_scaled,
    i16,
    nppsSubCRev_16s_Sfs_Ctx
);
impl_scaled_constant_operation!(
    subtract_from_constant_i32_scaled,
    i32,
    nppsSubCRev_32s_Sfs_Ctx
);
impl_scaled_constant_operation!(
    subtract_from_constant_i16_complex_scaled,
    ComplexI16,
    nppsSubCRev_16sc_Sfs_Ctx
);
impl_scaled_constant_operation!(
    subtract_from_constant_i32_complex_scaled,
    ComplexI32,
    nppsSubCRev_32sc_Sfs_Ctx
);
impl_scaled_constant_operation!(multiply_constant_u8_scaled, u8, nppsMulC_8u_Sfs_Ctx);
impl_scaled_constant_operation!(multiply_constant_u16_scaled, u16, nppsMulC_16u_Sfs_Ctx);
impl_scaled_constant_operation!(multiply_constant_i16_scaled, i16, nppsMulC_16s_Sfs_Ctx);
impl_scaled_constant_operation!(multiply_constant_i32_scaled, i32, nppsMulC_32s_Sfs_Ctx);
impl_scaled_constant_operation!(
    multiply_constant_i16_complex_scaled,
    ComplexI16,
    nppsMulC_16sc_Sfs_Ctx
);
impl_scaled_constant_operation!(
    multiply_constant_i32_complex_scaled,
    ComplexI32,
    nppsMulC_32sc_Sfs_Ctx
);
impl_scaled_constant_operation!(divide_constant_u8_scaled, u8, nppsDivC_8u_Sfs_Ctx);
impl_scaled_constant_operation!(divide_constant_u16_scaled, u16, nppsDivC_16u_Sfs_Ctx);
impl_scaled_constant_operation!(divide_constant_i16_scaled, i16, nppsDivC_16s_Sfs_Ctx);
impl_scaled_constant_operation!(
    divide_constant_i16_complex_scaled,
    ComplexI16,
    nppsDivC_16sc_Sfs_Ctx
);
impl_destination_update_constant_operation!(add_product_constant_f32, f32, nppsAddProductC_32f_Ctx);
impl_mixed_constant_operation!(
    multiply_constant_f32_to_i16_low,
    f32,
    i16,
    nppsMulC_Low_32f16s_Ctx
);
impl_scaled_mixed_constant_operation!(
    multiply_constant_f32_to_i16_scaled,
    f32,
    i16,
    nppsMulC_32f16s_Sfs_Ctx
);

impl_constant_operation_in_place!(add_constant_f32_in_place, f32, nppsAddC_32f_I_Ctx);
impl_constant_operation_in_place!(add_constant_f64_in_place, f64, nppsAddC_64f_I_Ctx);
impl_constant_operation_in_place!(
    add_constant_f32_complex_in_place,
    Complex32,
    nppsAddC_32fc_I_Ctx
);
impl_constant_operation_in_place!(
    add_constant_f64_complex_in_place,
    Complex64,
    nppsAddC_64fc_I_Ctx
);
impl_constant_operation_in_place!(subtract_constant_f32_in_place, f32, nppsSubC_32f_I_Ctx);
impl_constant_operation_in_place!(subtract_constant_f64_in_place, f64, nppsSubC_64f_I_Ctx);
impl_constant_operation_in_place!(
    subtract_constant_f32_complex_in_place,
    Complex32,
    nppsSubC_32fc_I_Ctx
);
impl_constant_operation_in_place!(
    subtract_constant_f64_complex_in_place,
    Complex64,
    nppsSubC_64fc_I_Ctx
);
impl_constant_operation_in_place!(
    subtract_from_constant_f32_in_place,
    f32,
    nppsSubCRev_32f_I_Ctx
);
impl_constant_operation_in_place!(
    subtract_from_constant_f64_in_place,
    f64,
    nppsSubCRev_64f_I_Ctx
);
impl_constant_operation_in_place!(
    subtract_from_constant_f32_complex_in_place,
    Complex32,
    nppsSubCRev_32fc_I_Ctx
);
impl_constant_operation_in_place!(
    subtract_from_constant_f64_complex_in_place,
    Complex64,
    nppsSubCRev_64fc_I_Ctx
);
impl_constant_operation_in_place!(multiply_constant_f32_in_place, f32, nppsMulC_32f_I_Ctx);
impl_constant_operation_in_place!(multiply_constant_f64_in_place, f64, nppsMulC_64f_I_Ctx);
impl_constant_operation_in_place!(
    multiply_constant_f32_complex_in_place,
    Complex32,
    nppsMulC_32fc_I_Ctx
);
impl_constant_operation_in_place!(
    multiply_constant_f64_complex_in_place,
    Complex64,
    nppsMulC_64fc_I_Ctx
);
impl_constant_operation_in_place!(divide_constant_f32_in_place, f32, nppsDivC_32f_I_Ctx);
impl_constant_operation_in_place!(divide_constant_f64_in_place, f64, nppsDivC_64f_I_Ctx);
impl_constant_operation_in_place!(
    divide_constant_f32_complex_in_place,
    Complex32,
    nppsDivC_32fc_I_Ctx
);
impl_constant_operation_in_place!(
    divide_constant_f64_complex_in_place,
    Complex64,
    nppsDivC_64fc_I_Ctx
);
impl_constant_operation_in_place!(
    divide_into_constant_u16_in_place,
    u16,
    nppsDivCRev_16u_I_Ctx
);
impl_constant_operation_in_place!(
    divide_into_constant_f32_in_place,
    f32,
    nppsDivCRev_32f_I_Ctx
);
impl_constant_operation_in_place!(and_constant_u8_in_place, u8, nppsAndC_8u_I_Ctx);
impl_constant_operation_in_place!(and_constant_u16_in_place, u16, nppsAndC_16u_I_Ctx);
impl_constant_operation_in_place!(and_constant_u32_in_place, u32, nppsAndC_32u_I_Ctx);
impl_constant_operation_in_place!(or_constant_u8_in_place, u8, nppsOrC_8u_I_Ctx);
impl_constant_operation_in_place!(or_constant_u16_in_place, u16, nppsOrC_16u_I_Ctx);
impl_constant_operation_in_place!(or_constant_u32_in_place, u32, nppsOrC_32u_I_Ctx);
impl_constant_operation_in_place!(xor_constant_u8_in_place, u8, nppsXorC_8u_I_Ctx);
impl_constant_operation_in_place!(xor_constant_u16_in_place, u16, nppsXorC_16u_I_Ctx);
impl_constant_operation_in_place!(xor_constant_u32_in_place, u32, nppsXorC_32u_I_Ctx);

impl_scaled_constant_operation_in_place!(add_constant_u8_scaled_in_place, u8, nppsAddC_8u_ISfs_Ctx);
impl_scaled_constant_operation_in_place!(
    add_constant_u16_scaled_in_place,
    u16,
    nppsAddC_16u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    add_constant_i16_scaled_in_place,
    i16,
    nppsAddC_16s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    add_constant_i32_scaled_in_place,
    i32,
    nppsAddC_32s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    add_constant_i16_complex_scaled_in_place,
    ComplexI16,
    nppsAddC_16sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    add_constant_i32_complex_scaled_in_place,
    ComplexI32,
    nppsAddC_32sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_constant_u8_scaled_in_place,
    u8,
    nppsSubC_8u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_constant_u16_scaled_in_place,
    u16,
    nppsSubC_16u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_constant_i16_scaled_in_place,
    i16,
    nppsSubC_16s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_constant_i32_scaled_in_place,
    i32,
    nppsSubC_32s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_constant_i16_complex_scaled_in_place,
    ComplexI16,
    nppsSubC_16sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_constant_i32_complex_scaled_in_place,
    ComplexI32,
    nppsSubC_32sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_from_constant_u8_scaled_in_place,
    u8,
    nppsSubCRev_8u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_from_constant_u16_scaled_in_place,
    u16,
    nppsSubCRev_16u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_from_constant_i16_scaled_in_place,
    i16,
    nppsSubCRev_16s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_from_constant_i32_scaled_in_place,
    i32,
    nppsSubCRev_32s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_from_constant_i16_complex_scaled_in_place,
    ComplexI16,
    nppsSubCRev_16sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    subtract_from_constant_i32_complex_scaled_in_place,
    ComplexI32,
    nppsSubCRev_32sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    multiply_constant_u8_scaled_in_place,
    u8,
    nppsMulC_8u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    multiply_constant_u16_scaled_in_place,
    u16,
    nppsMulC_16u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    multiply_constant_i16_scaled_in_place,
    i16,
    nppsMulC_16s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    multiply_constant_i32_scaled_in_place,
    i32,
    nppsMulC_32s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    multiply_constant_i16_complex_scaled_in_place,
    ComplexI16,
    nppsMulC_16sc_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    multiply_constant_i32_complex_scaled_in_place,
    ComplexI32,
    nppsMulC_32sc_ISfs_Ctx
);
impl_scaled_mixed_constant_operation_in_place!(
    multiply_constant_f64_to_i64_scaled_in_place,
    i64,
    f64,
    nppsMulC_64f64s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    divide_constant_u8_scaled_in_place,
    u8,
    nppsDivC_8u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    divide_constant_u16_scaled_in_place,
    u16,
    nppsDivC_16u_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    divide_constant_i16_scaled_in_place,
    i16,
    nppsDivC_16s_ISfs_Ctx
);
impl_scaled_constant_operation_in_place!(
    divide_constant_i16_complex_scaled_in_place,
    ComplexI16,
    nppsDivC_16sc_ISfs_Ctx
);

impl_shift_constant_operation!(left_shift_constant_u8, u8, nppsLShiftC_8u_Ctx);
impl_shift_constant_operation!(left_shift_constant_u16, u16, nppsLShiftC_16u_Ctx);
impl_shift_constant_operation!(left_shift_constant_i16, i16, nppsLShiftC_16s_Ctx);
impl_shift_constant_operation!(left_shift_constant_u32, u32, nppsLShiftC_32u_Ctx);
impl_shift_constant_operation!(left_shift_constant_i32, i32, nppsLShiftC_32s_Ctx);
impl_shift_constant_operation!(right_shift_constant_u8, u8, nppsRShiftC_8u_Ctx);
impl_shift_constant_operation!(right_shift_constant_u16, u16, nppsRShiftC_16u_Ctx);
impl_shift_constant_operation!(right_shift_constant_i16, i16, nppsRShiftC_16s_Ctx);
impl_shift_constant_operation!(right_shift_constant_u32, u32, nppsRShiftC_32u_Ctx);
impl_shift_constant_operation!(right_shift_constant_i32, i32, nppsRShiftC_32s_Ctx);

impl_shift_constant_operation_in_place!(left_shift_constant_u8_in_place, u8, nppsLShiftC_8u_I_Ctx);
impl_shift_constant_operation_in_place!(
    left_shift_constant_u16_in_place,
    u16,
    nppsLShiftC_16u_I_Ctx
);
impl_shift_constant_operation_in_place!(
    left_shift_constant_i16_in_place,
    i16,
    nppsLShiftC_16s_I_Ctx
);
impl_shift_constant_operation_in_place!(
    left_shift_constant_u32_in_place,
    u32,
    nppsLShiftC_32u_I_Ctx
);
impl_shift_constant_operation_in_place!(
    left_shift_constant_i32_in_place,
    i32,
    nppsLShiftC_32s_I_Ctx
);
impl_shift_constant_operation_in_place!(right_shift_constant_u8_in_place, u8, nppsRShiftC_8u_I_Ctx);
impl_shift_constant_operation_in_place!(
    right_shift_constant_u16_in_place,
    u16,
    nppsRShiftC_16u_I_Ctx
);
impl_shift_constant_operation_in_place!(
    right_shift_constant_i16_in_place,
    i16,
    nppsRShiftC_16s_I_Ctx
);
impl_shift_constant_operation_in_place!(
    right_shift_constant_u32_in_place,
    u32,
    nppsRShiftC_32u_I_Ctx
);
impl_shift_constant_operation_in_place!(
    right_shift_constant_i32_in_place,
    i32,
    nppsRShiftC_32s_I_Ctx
);
