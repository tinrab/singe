use super::*;

impl_generic_constant_scalar_operation!(AndConstantC1, and_constant, and_constant_c1, C1, [
    u8, u8 => and_constant_u8_c1,
    u16, u16 => and_constant_u16_c1,
    i32, i32 => and_constant_i32_c1,
]);
impl_generic_constant_scalar_operation_in_place!(
    AndConstantC1InPlace,
    and_constant_in_place,
    and_constant_c1_in_place,
    C1,
    [
        u8, u8 => and_constant_u8_c1_in_place,
        u16, u16 => and_constant_u16_c1_in_place,
        i32, i32 => and_constant_i32_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(AndConstantC3, and_constant, and_constant_c3, C3, 3, [
    u8, u8 => and_constant_u8_c3,
    u16, u16 => and_constant_u16_c3,
    i32, i32 => and_constant_i32_c3,
]);
impl_generic_constant_array_operation_in_place!(
    AndConstantC3InPlace,
    and_constant_in_place,
    and_constant_c3_in_place,
    C3,
    3,
    [
        u8, u8 => and_constant_u8_c3_in_place,
        u16, u16 => and_constant_u16_c3_in_place,
        i32, i32 => and_constant_i32_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(AndConstantC4, and_constant, and_constant_c4, C4, 4, [
    u8, u8 => and_constant_u8_c4,
    u16, u16 => and_constant_u16_c4,
    i32, i32 => and_constant_i32_c4,
]);
impl_generic_constant_array_operation_in_place!(
    AndConstantC4InPlace,
    and_constant_in_place,
    and_constant_c4_in_place,
    C4,
    4,
    [
        u8, u8 => and_constant_u8_c4_in_place,
        u16, u16 => and_constant_u16_c4_in_place,
        i32, i32 => and_constant_i32_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(AndConstantAc4, and_constant, and_constant_ac4, AC4, 3, [
    u8, u8 => and_constant_u8_ac4,
    u16, u16 => and_constant_u16_ac4,
    i32, i32 => and_constant_i32_ac4,
]);
impl_generic_constant_array_operation_in_place!(
    AndConstantAc4InPlace,
    and_constant_in_place,
    and_constant_ac4_in_place,
    AC4,
    3,
    [
        u8, u8 => and_constant_u8_ac4_in_place,
        u16, u16 => and_constant_u16_ac4_in_place,
        i32, i32 => and_constant_i32_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(OrConstantC1, or_constant, or_constant_c1, C1, [
    u8, u8 => or_constant_u8_c1,
    u16, u16 => or_constant_u16_c1,
    i32, i32 => or_constant_i32_c1,
]);
impl_generic_constant_scalar_operation_in_place!(
    OrConstantC1InPlace,
    or_constant_in_place,
    or_constant_c1_in_place,
    C1,
    [
        u8, u8 => or_constant_u8_c1_in_place,
        u16, u16 => or_constant_u16_c1_in_place,
        i32, i32 => or_constant_i32_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(OrConstantC3, or_constant, or_constant_c3, C3, 3, [
    u8, u8 => or_constant_u8_c3,
    u16, u16 => or_constant_u16_c3,
    i32, i32 => or_constant_i32_c3,
]);
impl_generic_constant_array_operation_in_place!(
    OrConstantC3InPlace,
    or_constant_in_place,
    or_constant_c3_in_place,
    C3,
    3,
    [
        u8, u8 => or_constant_u8_c3_in_place,
        u16, u16 => or_constant_u16_c3_in_place,
        i32, i32 => or_constant_i32_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(OrConstantC4, or_constant, or_constant_c4, C4, 4, [
    u8, u8 => or_constant_u8_c4,
    u16, u16 => or_constant_u16_c4,
    i32, i32 => or_constant_i32_c4,
]);
impl_generic_constant_array_operation_in_place!(
    OrConstantC4InPlace,
    or_constant_in_place,
    or_constant_c4_in_place,
    C4,
    4,
    [
        u8, u8 => or_constant_u8_c4_in_place,
        u16, u16 => or_constant_u16_c4_in_place,
        i32, i32 => or_constant_i32_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(OrConstantAc4, or_constant, or_constant_ac4, AC4, 3, [
    u8, u8 => or_constant_u8_ac4,
    u16, u16 => or_constant_u16_ac4,
    i32, i32 => or_constant_i32_ac4,
]);
impl_generic_constant_array_operation_in_place!(
    OrConstantAc4InPlace,
    or_constant_in_place,
    or_constant_ac4_in_place,
    AC4,
    3,
    [
        u8, u8 => or_constant_u8_ac4_in_place,
        u16, u16 => or_constant_u16_ac4_in_place,
        i32, i32 => or_constant_i32_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(XorConstantC1, xor_constant, xor_constant_c1, C1, [
    u8, u8 => xor_constant_u8_c1,
    u16, u16 => xor_constant_u16_c1,
    i32, i32 => xor_constant_i32_c1,
]);
impl_generic_constant_scalar_operation_in_place!(
    XorConstantC1InPlace,
    xor_constant_in_place,
    xor_constant_c1_in_place,
    C1,
    [
        u8, u8 => xor_constant_u8_c1_in_place,
        u16, u16 => xor_constant_u16_c1_in_place,
        i32, i32 => xor_constant_i32_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(XorConstantC3, xor_constant, xor_constant_c3, C3, 3, [
    u8, u8 => xor_constant_u8_c3,
    u16, u16 => xor_constant_u16_c3,
    i32, i32 => xor_constant_i32_c3,
]);
impl_generic_constant_array_operation_in_place!(
    XorConstantC3InPlace,
    xor_constant_in_place,
    xor_constant_c3_in_place,
    C3,
    3,
    [
        u8, u8 => xor_constant_u8_c3_in_place,
        u16, u16 => xor_constant_u16_c3_in_place,
        i32, i32 => xor_constant_i32_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(XorConstantC4, xor_constant, xor_constant_c4, C4, 4, [
    u8, u8 => xor_constant_u8_c4,
    u16, u16 => xor_constant_u16_c4,
    i32, i32 => xor_constant_i32_c4,
]);
impl_generic_constant_array_operation_in_place!(
    XorConstantC4InPlace,
    xor_constant_in_place,
    xor_constant_c4_in_place,
    C4,
    4,
    [
        u8, u8 => xor_constant_u8_c4_in_place,
        u16, u16 => xor_constant_u16_c4_in_place,
        i32, i32 => xor_constant_i32_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(XorConstantAc4, xor_constant, xor_constant_ac4, AC4, 3, [
    u8, u8 => xor_constant_u8_ac4,
    u16, u16 => xor_constant_u16_ac4,
    i32, i32 => xor_constant_i32_ac4,
]);
impl_generic_constant_array_operation_in_place!(
    XorConstantAc4InPlace,
    xor_constant_in_place,
    xor_constant_ac4_in_place,
    AC4,
    3,
    [
        u8, u8 => xor_constant_u8_ac4_in_place,
        u16, u16 => xor_constant_u16_ac4_in_place,
        i32, i32 => xor_constant_i32_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(
    RightShiftConstantC1,
    right_shift_constant,
    right_shift_constant_c1,
    C1,
    [
        u8, u32 => right_shift_constant_u8_c1,
        i8, u32 => right_shift_constant_i8_c1,
        u16, u32 => right_shift_constant_u16_c1,
        i16, u32 => right_shift_constant_i16_c1,
        i32, u32 => right_shift_constant_i32_c1,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    RightShiftConstantC1InPlace,
    right_shift_constant_in_place,
    right_shift_constant_c1_in_place,
    C1,
    [
        u8, u32 => right_shift_constant_u8_c1_in_place,
        i8, u32 => right_shift_constant_i8_c1_in_place,
        u16, u32 => right_shift_constant_u16_c1_in_place,
        i16, u32 => right_shift_constant_i16_c1_in_place,
        i32, u32 => right_shift_constant_i32_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(
    RightShiftConstantC3,
    right_shift_constant,
    right_shift_constant_c3,
    C3,
    3,
    [
        u8, u32 => right_shift_constant_u8_c3,
        i8, u32 => right_shift_constant_i8_c3,
        u16, u32 => right_shift_constant_u16_c3,
        i16, u32 => right_shift_constant_i16_c3,
        i32, u32 => right_shift_constant_i32_c3,
    ]
);
impl_generic_constant_array_operation_in_place!(
    RightShiftConstantC3InPlace,
    right_shift_constant_in_place,
    right_shift_constant_c3_in_place,
    C3,
    3,
    [
        u8, u32 => right_shift_constant_u8_c3_in_place,
        i8, u32 => right_shift_constant_i8_c3_in_place,
        u16, u32 => right_shift_constant_u16_c3_in_place,
        i16, u32 => right_shift_constant_i16_c3_in_place,
        i32, u32 => right_shift_constant_i32_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(
    RightShiftConstantC4,
    right_shift_constant,
    right_shift_constant_c4,
    C4,
    4,
    [
        u8, u32 => right_shift_constant_u8_c4,
        i8, u32 => right_shift_constant_i8_c4,
        u16, u32 => right_shift_constant_u16_c4,
        i16, u32 => right_shift_constant_i16_c4,
        i32, u32 => right_shift_constant_i32_c4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    RightShiftConstantC4InPlace,
    right_shift_constant_in_place,
    right_shift_constant_c4_in_place,
    C4,
    4,
    [
        u8, u32 => right_shift_constant_u8_c4_in_place,
        i8, u32 => right_shift_constant_i8_c4_in_place,
        u16, u32 => right_shift_constant_u16_c4_in_place,
        i16, u32 => right_shift_constant_i16_c4_in_place,
        i32, u32 => right_shift_constant_i32_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(
    RightShiftConstantAc4,
    right_shift_constant,
    right_shift_constant_ac4,
    AC4,
    3,
    [
        u8, u32 => right_shift_constant_u8_ac4,
        i8, u32 => right_shift_constant_i8_ac4,
        u16, u32 => right_shift_constant_u16_ac4,
        i16, u32 => right_shift_constant_i16_ac4,
        i32, u32 => right_shift_constant_i32_ac4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    RightShiftConstantAc4InPlace,
    right_shift_constant_in_place,
    right_shift_constant_ac4_in_place,
    AC4,
    3,
    [
        u8, u32 => right_shift_constant_u8_ac4_in_place,
        i8, u32 => right_shift_constant_i8_ac4_in_place,
        u16, u32 => right_shift_constant_u16_ac4_in_place,
        i16, u32 => right_shift_constant_i16_ac4_in_place,
        i32, u32 => right_shift_constant_i32_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(
    LeftShiftConstantC1,
    left_shift_constant,
    left_shift_constant_c1,
    C1,
    [
        u8, u32 => left_shift_constant_u8_c1,
        u16, u32 => left_shift_constant_u16_c1,
        i32, u32 => left_shift_constant_i32_c1,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    LeftShiftConstantC1InPlace,
    left_shift_constant_in_place,
    left_shift_constant_c1_in_place,
    C1,
    [
        u8, u32 => left_shift_constant_u8_c1_in_place,
        u16, u32 => left_shift_constant_u16_c1_in_place,
        i32, u32 => left_shift_constant_i32_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(
    LeftShiftConstantC3,
    left_shift_constant,
    left_shift_constant_c3,
    C3,
    3,
    [
        u8, u32 => left_shift_constant_u8_c3,
        u16, u32 => left_shift_constant_u16_c3,
        i32, u32 => left_shift_constant_i32_c3,
    ]
);
impl_generic_constant_array_operation_in_place!(
    LeftShiftConstantC3InPlace,
    left_shift_constant_in_place,
    left_shift_constant_c3_in_place,
    C3,
    3,
    [
        u8, u32 => left_shift_constant_u8_c3_in_place,
        u16, u32 => left_shift_constant_u16_c3_in_place,
        i32, u32 => left_shift_constant_i32_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(
    LeftShiftConstantC4,
    left_shift_constant,
    left_shift_constant_c4,
    C4,
    4,
    [
        u8, u32 => left_shift_constant_u8_c4,
        u16, u32 => left_shift_constant_u16_c4,
        i32, u32 => left_shift_constant_i32_c4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    LeftShiftConstantC4InPlace,
    left_shift_constant_in_place,
    left_shift_constant_c4_in_place,
    C4,
    4,
    [
        u8, u32 => left_shift_constant_u8_c4_in_place,
        u16, u32 => left_shift_constant_u16_c4_in_place,
        i32, u32 => left_shift_constant_i32_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(
    LeftShiftConstantAc4,
    left_shift_constant,
    left_shift_constant_ac4,
    AC4,
    3,
    [
        u8, u32 => left_shift_constant_u8_ac4,
        u16, u32 => left_shift_constant_u16_ac4,
        i32, u32 => left_shift_constant_i32_ac4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    LeftShiftConstantAc4InPlace,
    left_shift_constant_in_place,
    left_shift_constant_ac4_in_place,
    AC4,
    3,
    [
        u8, u32 => left_shift_constant_u8_ac4_in_place,
        u16, u32 => left_shift_constant_u16_ac4_in_place,
        i32, u32 => left_shift_constant_i32_ac4_in_place,
    ]
);
