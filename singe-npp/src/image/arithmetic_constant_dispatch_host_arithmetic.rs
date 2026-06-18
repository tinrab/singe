use super::*;

impl_generic_constant_scalar_operation!(AddConstantC1, add_constant, add_constant_c1, C1, [
    f16, f32 => add_constant_f16_c1,
    f32, f32 => add_constant_f32_c1,
    Complex32, Complex32 => add_constant_f32_complex_c1,
]);
impl_generic_constant_scalar_operation_in_place!(
    AddConstantC1InPlace,
    add_constant_in_place,
    add_constant_c1_in_place,
    C1,
    [
        f16, f32 => add_constant_f16_c1_in_place,
        f32, f32 => add_constant_f32_c1_in_place,
        Complex32, Complex32 => add_constant_f32_complex_c1_in_place,
    ]
);
impl_generic_scaled_constant_scalar_operation!(
    ScaledAddConstantC1,
    add_constant_scaled,
    add_constant_scaled_c1,
    C1,
    [
        u8 => add_constant_u8_c1,
        u16 => add_constant_u16_c1,
        i16 => add_constant_i16_c1,
        ComplexI16 => add_constant_i16_complex_c1,
        i32 => add_constant_i32_c1,
        ComplexI32 => add_constant_i32_complex_c1,
    ]
);
impl_generic_scaled_constant_scalar_operation_in_place!(
    ScaledAddConstantC1InPlace,
    add_constant_scaled_in_place,
    add_constant_scaled_c1_in_place,
    C1,
    [
        u8 => add_constant_u8_c1_in_place,
        u16 => add_constant_u16_c1_in_place,
        i16 => add_constant_i16_c1_in_place,
        ComplexI16 => add_constant_i16_complex_c1_in_place,
        i32 => add_constant_i32_c1_in_place,
        ComplexI32 => add_constant_i32_complex_c1_in_place,
    ]
);

impl_generic_constant_array_operation!(AddConstantC3, add_constant, add_constant_c3, C3, 3, [
    f16, f32 => add_constant_f16_c3,
    f32, f32 => add_constant_f32_c3,
    Complex32, Complex32 => add_constant_f32_complex_c3,
]);
impl_generic_constant_array_operation_in_place!(
    AddConstantC3InPlace,
    add_constant_in_place,
    add_constant_c3_in_place,
    C3,
    3,
    [
        f16, f32 => add_constant_f16_c3_in_place,
        f32, f32 => add_constant_f32_c3_in_place,
        Complex32, Complex32 => add_constant_f32_complex_c3_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledAddConstantC3,
    add_constant_scaled,
    add_constant_scaled_c3,
    C3,
    3,
    [
        u8 => add_constant_u8_c3,
        u16 => add_constant_u16_c3,
        i16 => add_constant_i16_c3,
        ComplexI16 => add_constant_i16_complex_c3,
        i32 => add_constant_i32_c3,
        ComplexI32 => add_constant_i32_complex_c3,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledAddConstantC3InPlace,
    add_constant_scaled_in_place,
    add_constant_scaled_c3_in_place,
    C3,
    3,
    [
        u8 => add_constant_u8_c3_in_place,
        u16 => add_constant_u16_c3_in_place,
        i16 => add_constant_i16_c3_in_place,
        ComplexI16 => add_constant_i16_complex_c3_in_place,
        i32 => add_constant_i32_c3_in_place,
        ComplexI32 => add_constant_i32_complex_c3_in_place,
    ]
);

impl_generic_constant_array_operation!(AddConstantC4, add_constant, add_constant_c4, C4, 4, [
    f16, f32 => add_constant_f16_c4,
    f32, f32 => add_constant_f32_c4,
    Complex32, Complex32 => add_constant_f32_complex_c4,
]);
impl_generic_constant_array_operation_in_place!(
    AddConstantC4InPlace,
    add_constant_in_place,
    add_constant_c4_in_place,
    C4,
    4,
    [
        f16, f32 => add_constant_f16_c4_in_place,
        f32, f32 => add_constant_f32_c4_in_place,
        Complex32, Complex32 => add_constant_f32_complex_c4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledAddConstantC4,
    add_constant_scaled,
    add_constant_scaled_c4,
    C4,
    4,
    [
        u8 => add_constant_u8_c4,
        u16 => add_constant_u16_c4,
        i16 => add_constant_i16_c4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledAddConstantC4InPlace,
    add_constant_scaled_in_place,
    add_constant_scaled_c4_in_place,
    C4,
    4,
    [
        u8 => add_constant_u8_c4_in_place,
        u16 => add_constant_u16_c4_in_place,
        i16 => add_constant_i16_c4_in_place,
    ]
);

impl_generic_constant_array_operation!(AddConstantAc4, add_constant, add_constant_ac4, AC4, 3, [
    f32, f32 => add_constant_f32_ac4,
    Complex32, Complex32 => add_constant_f32_complex_ac4,
]);
impl_generic_constant_array_operation_in_place!(
    AddConstantAc4InPlace,
    add_constant_in_place,
    add_constant_ac4_in_place,
    AC4,
    3,
    [
        f32, f32 => add_constant_f32_ac4_in_place,
        Complex32, Complex32 => add_constant_f32_complex_ac4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledAddConstantAc4,
    add_constant_scaled,
    add_constant_scaled_ac4,
    AC4,
    3,
    [
        u8 => add_constant_u8_ac4,
        u16 => add_constant_u16_ac4,
        i16 => add_constant_i16_ac4,
        ComplexI16 => add_constant_i16_complex_ac4,
        ComplexI32 => add_constant_i32_complex_ac4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledAddConstantAc4InPlace,
    add_constant_scaled_in_place,
    add_constant_scaled_ac4_in_place,
    AC4,
    3,
    [
        u8 => add_constant_u8_ac4_in_place,
        u16 => add_constant_u16_ac4_in_place,
        i16 => add_constant_i16_ac4_in_place,
        ComplexI16 => add_constant_i16_complex_ac4_in_place,
        ComplexI32 => add_constant_i32_complex_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(
    MultiplyConstantC1,
    multiply_constant,
    multiply_constant_c1,
    C1,
    [
        f16, f32 => multiply_constant_f16_c1,
        f32, f32 => multiply_constant_f32_c1,
        Complex32, Complex32 => multiply_constant_f32_complex_c1,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    MultiplyConstantC1InPlace,
    multiply_constant_in_place,
    multiply_constant_c1_in_place,
    C1,
    [
        f16, f32 => multiply_constant_f16_c1_in_place,
        f32, f32 => multiply_constant_f32_c1_in_place,
        Complex32, Complex32 => multiply_constant_f32_complex_c1_in_place,
    ]
);
impl_generic_scaled_constant_scalar_operation!(
    ScaledMultiplyConstantC1,
    multiply_constant_scaled,
    multiply_constant_scaled_c1,
    C1,
    [
        u8 => multiply_constant_u8_c1,
        u16 => multiply_constant_u16_c1,
        i16 => multiply_constant_i16_c1,
        ComplexI16 => multiply_constant_i16_complex_c1,
        i32 => multiply_constant_i32_c1,
        ComplexI32 => multiply_constant_i32_complex_c1,
    ]
);
impl_generic_scaled_constant_scalar_operation_in_place!(
    ScaledMultiplyConstantC1InPlace,
    multiply_constant_scaled_in_place,
    multiply_constant_scaled_c1_in_place,
    C1,
    [
        u8 => multiply_constant_u8_c1_in_place,
        u16 => multiply_constant_u16_c1_in_place,
        i16 => multiply_constant_i16_c1_in_place,
        ComplexI16 => multiply_constant_i16_complex_c1_in_place,
        i32 => multiply_constant_i32_c1_in_place,
        ComplexI32 => multiply_constant_i32_complex_c1_in_place,
    ]
);

impl_generic_constant_array_operation!(
    MultiplyConstantC3,
    multiply_constant,
    multiply_constant_c3,
    C3,
    3,
    [
        f16, f32 => multiply_constant_f16_c3,
        f32, f32 => multiply_constant_f32_c3,
        Complex32, Complex32 => multiply_constant_f32_complex_c3,
    ]
);
impl_generic_constant_array_operation_in_place!(
    MultiplyConstantC3InPlace,
    multiply_constant_in_place,
    multiply_constant_c3_in_place,
    C3,
    3,
    [
        f16, f32 => multiply_constant_f16_c3_in_place,
        f32, f32 => multiply_constant_f32_c3_in_place,
        Complex32, Complex32 => multiply_constant_f32_complex_c3_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledMultiplyConstantC3,
    multiply_constant_scaled,
    multiply_constant_scaled_c3,
    C3,
    3,
    [
        u8 => multiply_constant_u8_c3,
        u16 => multiply_constant_u16_c3,
        i16 => multiply_constant_i16_c3,
        ComplexI16 => multiply_constant_i16_complex_c3,
        i32 => multiply_constant_i32_c3,
        ComplexI32 => multiply_constant_i32_complex_c3,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledMultiplyConstantC3InPlace,
    multiply_constant_scaled_in_place,
    multiply_constant_scaled_c3_in_place,
    C3,
    3,
    [
        u8 => multiply_constant_u8_c3_in_place,
        u16 => multiply_constant_u16_c3_in_place,
        i16 => multiply_constant_i16_c3_in_place,
        ComplexI16 => multiply_constant_i16_complex_c3_in_place,
        i32 => multiply_constant_i32_c3_in_place,
        ComplexI32 => multiply_constant_i32_complex_c3_in_place,
    ]
);

impl_generic_constant_array_operation!(
    MultiplyConstantC4,
    multiply_constant,
    multiply_constant_c4,
    C4,
    4,
    [
        f16, f32 => multiply_constant_f16_c4,
        f32, f32 => multiply_constant_f32_c4,
        Complex32, Complex32 => multiply_constant_f32_complex_c4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    MultiplyConstantC4InPlace,
    multiply_constant_in_place,
    multiply_constant_c4_in_place,
    C4,
    4,
    [
        f16, f32 => multiply_constant_f16_c4_in_place,
        f32, f32 => multiply_constant_f32_c4_in_place,
        Complex32, Complex32 => multiply_constant_f32_complex_c4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledMultiplyConstantC4,
    multiply_constant_scaled,
    multiply_constant_scaled_c4,
    C4,
    4,
    [
        u8 => multiply_constant_u8_c4,
        u16 => multiply_constant_u16_c4,
        i16 => multiply_constant_i16_c4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledMultiplyConstantC4InPlace,
    multiply_constant_scaled_in_place,
    multiply_constant_scaled_c4_in_place,
    C4,
    4,
    [
        u8 => multiply_constant_u8_c4_in_place,
        u16 => multiply_constant_u16_c4_in_place,
        i16 => multiply_constant_i16_c4_in_place,
    ]
);

impl_generic_constant_array_operation!(
    MultiplyConstantAc4,
    multiply_constant,
    multiply_constant_ac4,
    AC4,
    3,
    [
        f32, f32 => multiply_constant_f32_ac4,
        Complex32, Complex32 => multiply_constant_f32_complex_ac4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    MultiplyConstantAc4InPlace,
    multiply_constant_in_place,
    multiply_constant_ac4_in_place,
    AC4,
    3,
    [
        f32, f32 => multiply_constant_f32_ac4_in_place,
        Complex32, Complex32 => multiply_constant_f32_complex_ac4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledMultiplyConstantAc4,
    multiply_constant_scaled,
    multiply_constant_scaled_ac4,
    AC4,
    3,
    [
        u8 => multiply_constant_u8_ac4,
        u16 => multiply_constant_u16_ac4,
        i16 => multiply_constant_i16_ac4,
        ComplexI16 => multiply_constant_i16_complex_ac4,
        ComplexI32 => multiply_constant_i32_complex_ac4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledMultiplyConstantAc4InPlace,
    multiply_constant_scaled_in_place,
    multiply_constant_scaled_ac4_in_place,
    AC4,
    3,
    [
        u8 => multiply_constant_u8_ac4_in_place,
        u16 => multiply_constant_u16_ac4_in_place,
        i16 => multiply_constant_i16_ac4_in_place,
        ComplexI16 => multiply_constant_i16_complex_ac4_in_place,
        ComplexI32 => multiply_constant_i32_complex_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(
    SubtractConstantC1,
    subtract_constant,
    subtract_constant_c1,
    C1,
    [
        f16, f32 => subtract_constant_f16_c1,
        f32, f32 => subtract_constant_f32_c1,
        Complex32, Complex32 => subtract_constant_f32_complex_c1,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    SubtractConstantC1InPlace,
    subtract_constant_in_place,
    subtract_constant_c1_in_place,
    C1,
    [
        f16, f32 => subtract_constant_f16_c1_in_place,
        f32, f32 => subtract_constant_f32_c1_in_place,
        Complex32, Complex32 => subtract_constant_f32_complex_c1_in_place,
    ]
);
impl_generic_scaled_constant_scalar_operation!(
    ScaledSubtractConstantC1,
    subtract_constant_scaled,
    subtract_constant_scaled_c1,
    C1,
    [
        u8 => subtract_constant_u8_c1,
        u16 => subtract_constant_u16_c1,
        i16 => subtract_constant_i16_c1,
        ComplexI16 => subtract_constant_i16_complex_c1,
        i32 => subtract_constant_i32_c1,
        ComplexI32 => subtract_constant_i32_complex_c1,
    ]
);
impl_generic_scaled_constant_scalar_operation_in_place!(
    ScaledSubtractConstantC1InPlace,
    subtract_constant_scaled_in_place,
    subtract_constant_scaled_c1_in_place,
    C1,
    [
        u8 => subtract_constant_u8_c1_in_place,
        u16 => subtract_constant_u16_c1_in_place,
        i16 => subtract_constant_i16_c1_in_place,
        ComplexI16 => subtract_constant_i16_complex_c1_in_place,
        i32 => subtract_constant_i32_c1_in_place,
        ComplexI32 => subtract_constant_i32_complex_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(
    SubtractConstantC3,
    subtract_constant,
    subtract_constant_c3,
    C3,
    3,
    [
        f16, f32 => subtract_constant_f16_c3,
        f32, f32 => subtract_constant_f32_c3,
        Complex32, Complex32 => subtract_constant_f32_complex_c3,
    ]
);
impl_generic_constant_array_operation_in_place!(
    SubtractConstantC3InPlace,
    subtract_constant_in_place,
    subtract_constant_c3_in_place,
    C3,
    3,
    [
        f16, f32 => subtract_constant_f16_c3_in_place,
        f32, f32 => subtract_constant_f32_c3_in_place,
        Complex32, Complex32 => subtract_constant_f32_complex_c3_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledSubtractConstantC3,
    subtract_constant_scaled,
    subtract_constant_scaled_c3,
    C3,
    3,
    [
        u8 => subtract_constant_u8_c3,
        u16 => subtract_constant_u16_c3,
        i16 => subtract_constant_i16_c3,
        ComplexI16 => subtract_constant_i16_complex_c3,
        i32 => subtract_constant_i32_c3,
        ComplexI32 => subtract_constant_i32_complex_c3,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledSubtractConstantC3InPlace,
    subtract_constant_scaled_in_place,
    subtract_constant_scaled_c3_in_place,
    C3,
    3,
    [
        u8 => subtract_constant_u8_c3_in_place,
        u16 => subtract_constant_u16_c3_in_place,
        i16 => subtract_constant_i16_c3_in_place,
        ComplexI16 => subtract_constant_i16_complex_c3_in_place,
        i32 => subtract_constant_i32_c3_in_place,
        ComplexI32 => subtract_constant_i32_complex_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(
    SubtractConstantC4,
    subtract_constant,
    subtract_constant_c4,
    C4,
    4,
    [
        f16, f32 => subtract_constant_f16_c4,
        f32, f32 => subtract_constant_f32_c4,
        Complex32, Complex32 => subtract_constant_f32_complex_c4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    SubtractConstantC4InPlace,
    subtract_constant_in_place,
    subtract_constant_c4_in_place,
    C4,
    4,
    [
        f16, f32 => subtract_constant_f16_c4_in_place,
        f32, f32 => subtract_constant_f32_c4_in_place,
        Complex32, Complex32 => subtract_constant_f32_complex_c4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledSubtractConstantC4,
    subtract_constant_scaled,
    subtract_constant_scaled_c4,
    C4,
    4,
    [
        u8 => subtract_constant_u8_c4,
        u16 => subtract_constant_u16_c4,
        i16 => subtract_constant_i16_c4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledSubtractConstantC4InPlace,
    subtract_constant_scaled_in_place,
    subtract_constant_scaled_c4_in_place,
    C4,
    4,
    [
        u8 => subtract_constant_u8_c4_in_place,
        u16 => subtract_constant_u16_c4_in_place,
        i16 => subtract_constant_i16_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(
    SubtractConstantAc4,
    subtract_constant,
    subtract_constant_ac4,
    AC4,
    3,
    [
        f32, f32 => subtract_constant_f32_ac4,
        Complex32, Complex32 => subtract_constant_f32_complex_ac4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    SubtractConstantAc4InPlace,
    subtract_constant_in_place,
    subtract_constant_ac4_in_place,
    AC4,
    3,
    [
        f32, f32 => subtract_constant_f32_ac4_in_place,
        Complex32, Complex32 => subtract_constant_f32_complex_ac4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledSubtractConstantAc4,
    subtract_constant_scaled,
    subtract_constant_scaled_ac4,
    AC4,
    3,
    [
        u8 => subtract_constant_u8_ac4,
        u16 => subtract_constant_u16_ac4,
        i16 => subtract_constant_i16_ac4,
        ComplexI16 => subtract_constant_i16_complex_ac4,
        ComplexI32 => subtract_constant_i32_complex_ac4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledSubtractConstantAc4InPlace,
    subtract_constant_scaled_in_place,
    subtract_constant_scaled_ac4_in_place,
    AC4,
    3,
    [
        u8 => subtract_constant_u8_ac4_in_place,
        u16 => subtract_constant_u16_ac4_in_place,
        i16 => subtract_constant_i16_ac4_in_place,
        ComplexI16 => subtract_constant_i16_complex_ac4_in_place,
        ComplexI32 => subtract_constant_i32_complex_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(DivideConstantC1, divide_constant, divide_constant_c1, C1, [
    f16, f32 => divide_constant_f16_c1,
    f32, f32 => divide_constant_f32_c1,
    Complex32, Complex32 => divide_constant_f32_complex_c1,
]);
impl_generic_constant_scalar_operation_in_place!(
    DivideConstantC1InPlace,
    divide_constant_in_place,
    divide_constant_c1_in_place,
    C1,
    [
        f16, f32 => divide_constant_f16_c1_in_place,
        f32, f32 => divide_constant_f32_c1_in_place,
        Complex32, Complex32 => divide_constant_f32_complex_c1_in_place,
    ]
);
impl_generic_scaled_constant_scalar_operation!(
    ScaledDivideConstantC1,
    divide_constant_scaled,
    divide_constant_scaled_c1,
    C1,
    [
        u8 => divide_constant_u8_c1,
        u16 => divide_constant_u16_c1,
        i16 => divide_constant_i16_c1,
        ComplexI16 => divide_constant_i16_complex_c1,
        i32 => divide_constant_i32_c1,
        ComplexI32 => divide_constant_i32_complex_c1,
    ]
);
impl_generic_scaled_constant_scalar_operation_in_place!(
    ScaledDivideConstantC1InPlace,
    divide_constant_scaled_in_place,
    divide_constant_scaled_c1_in_place,
    C1,
    [
        u8 => divide_constant_u8_c1_in_place,
        u16 => divide_constant_u16_c1_in_place,
        i16 => divide_constant_i16_c1_in_place,
        ComplexI16 => divide_constant_i16_complex_c1_in_place,
        i32 => divide_constant_i32_c1_in_place,
        ComplexI32 => divide_constant_i32_complex_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(DivideConstantC3, divide_constant, divide_constant_c3, C3, 3, [
    f16, f32 => divide_constant_f16_c3,
    f32, f32 => divide_constant_f32_c3,
    Complex32, Complex32 => divide_constant_f32_complex_c3,
]);
impl_generic_constant_array_operation_in_place!(
    DivideConstantC3InPlace,
    divide_constant_in_place,
    divide_constant_c3_in_place,
    C3,
    3,
    [
        f16, f32 => divide_constant_f16_c3_in_place,
        f32, f32 => divide_constant_f32_c3_in_place,
        Complex32, Complex32 => divide_constant_f32_complex_c3_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledDivideConstantC3,
    divide_constant_scaled,
    divide_constant_scaled_c3,
    C3,
    3,
    [
        u8 => divide_constant_u8_c3,
        u16 => divide_constant_u16_c3,
        i16 => divide_constant_i16_c3,
        ComplexI16 => divide_constant_i16_complex_c3,
        i32 => divide_constant_i32_c3,
        ComplexI32 => divide_constant_i32_complex_c3,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledDivideConstantC3InPlace,
    divide_constant_scaled_in_place,
    divide_constant_scaled_c3_in_place,
    C3,
    3,
    [
        u8 => divide_constant_u8_c3_in_place,
        u16 => divide_constant_u16_c3_in_place,
        i16 => divide_constant_i16_c3_in_place,
        ComplexI16 => divide_constant_i16_complex_c3_in_place,
        i32 => divide_constant_i32_c3_in_place,
        ComplexI32 => divide_constant_i32_complex_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(DivideConstantC4, divide_constant, divide_constant_c4, C4, 4, [
    f16, f32 => divide_constant_f16_c4,
    f32, f32 => divide_constant_f32_c4,
    Complex32, Complex32 => divide_constant_f32_complex_c4,
]);
impl_generic_constant_array_operation_in_place!(
    DivideConstantC4InPlace,
    divide_constant_in_place,
    divide_constant_c4_in_place,
    C4,
    4,
    [
        f16, f32 => divide_constant_f16_c4_in_place,
        f32, f32 => divide_constant_f32_c4_in_place,
        Complex32, Complex32 => divide_constant_f32_complex_c4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledDivideConstantC4,
    divide_constant_scaled,
    divide_constant_scaled_c4,
    C4,
    4,
    [
        u8 => divide_constant_u8_c4,
        u16 => divide_constant_u16_c4,
        i16 => divide_constant_i16_c4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledDivideConstantC4InPlace,
    divide_constant_scaled_in_place,
    divide_constant_scaled_c4_in_place,
    C4,
    4,
    [
        u8 => divide_constant_u8_c4_in_place,
        u16 => divide_constant_u16_c4_in_place,
        i16 => divide_constant_i16_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(
    DivideConstantAc4,
    divide_constant,
    divide_constant_ac4,
    AC4,
    3,
    [
        f32, f32 => divide_constant_f32_ac4,
        Complex32, Complex32 => divide_constant_f32_complex_ac4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    DivideConstantAc4InPlace,
    divide_constant_in_place,
    divide_constant_ac4_in_place,
    AC4,
    3,
    [
        f32, f32 => divide_constant_f32_ac4_in_place,
        Complex32, Complex32 => divide_constant_f32_complex_ac4_in_place,
    ]
);
impl_generic_scaled_constant_array_operation!(
    ScaledDivideConstantAc4,
    divide_constant_scaled,
    divide_constant_scaled_ac4,
    AC4,
    3,
    [
        u8 => divide_constant_u8_ac4,
        u16 => divide_constant_u16_ac4,
        i16 => divide_constant_i16_ac4,
        ComplexI16 => divide_constant_i16_complex_ac4,
        ComplexI32 => divide_constant_i32_complex_ac4,
    ]
);
impl_generic_scaled_constant_array_operation_in_place!(
    ScaledDivideConstantAc4InPlace,
    divide_constant_scaled_in_place,
    divide_constant_scaled_ac4_in_place,
    AC4,
    3,
    [
        u8 => divide_constant_u8_ac4_in_place,
        u16 => divide_constant_u16_ac4_in_place,
        i16 => divide_constant_i16_ac4_in_place,
        ComplexI16 => divide_constant_i16_complex_ac4_in_place,
        ComplexI32 => divide_constant_i32_complex_ac4_in_place,
    ]
);
