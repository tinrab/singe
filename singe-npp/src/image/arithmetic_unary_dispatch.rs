use super::*;

impl_generic_unary_operation!(
    AbsoluteC1,
    absolute,
    absolute_c1,
    C1,
    [
        i16 => absolute_i16_c1,
        f16 => absolute_f16_c1,
        f32 => absolute_f32_c1,
    ]
);
impl_generic_unary_operation_in_place!(
    AbsoluteC1InPlace,
    absolute_in_place,
    absolute_c1_in_place,
    C1,
    [
        i16 => absolute_i16_c1_in_place,
        f16 => absolute_f16_c1_in_place,
        f32 => absolute_f32_c1_in_place,
    ]
);
impl_generic_unary_operation!(
    AbsoluteC3,
    absolute,
    absolute_c3,
    C3,
    [
        i16 => absolute_i16_c3,
        f16 => absolute_f16_c3,
        f32 => absolute_f32_c3,
    ]
);
impl_generic_unary_operation_in_place!(
    AbsoluteC3InPlace,
    absolute_in_place,
    absolute_c3_in_place,
    C3,
    [
        i16 => absolute_i16_c3_in_place,
        f16 => absolute_f16_c3_in_place,
        f32 => absolute_f32_c3_in_place,
    ]
);
impl_generic_unary_operation!(
    AbsoluteC4,
    absolute,
    absolute_c4,
    C4,
    [
        i16 => absolute_i16_c4,
        f16 => absolute_f16_c4,
        f32 => absolute_f32_c4,
    ]
);
impl_generic_unary_operation_in_place!(
    AbsoluteC4InPlace,
    absolute_in_place,
    absolute_c4_in_place,
    C4,
    [
        i16 => absolute_i16_c4_in_place,
        f16 => absolute_f16_c4_in_place,
        f32 => absolute_f32_c4_in_place,
    ]
);
impl_generic_unary_operation!(
    AbsoluteAc4,
    absolute,
    absolute_ac4,
    AC4,
    [
        i16 => absolute_i16_ac4,
        f32 => absolute_f32_ac4,
    ]
);
impl_generic_unary_operation_in_place!(
    AbsoluteAc4InPlace,
    absolute_in_place,
    absolute_ac4_in_place,
    AC4,
    [
        i16 => absolute_i16_ac4_in_place,
        f32 => absolute_f32_ac4_in_place,
    ]
);

impl_generic_binary_operation!(
    AbsoluteDifferenceC1,
    absolute_difference,
    absolute_difference_c1,
    C1,
    [
        u8 => absolute_difference_u8_c1,
        u16 => absolute_difference_u16_c1,
        f16 => absolute_difference_f16_c1,
        f32 => absolute_difference_f32_c1,
    ]
);
impl_generic_binary_operation!(
    AbsoluteDifferenceC3,
    absolute_difference,
    absolute_difference_c3,
    C3,
    [u8 => absolute_difference_u8_c3]
);
impl_generic_binary_operation!(
    AbsoluteDifferenceC4,
    absolute_difference,
    absolute_difference_c4,
    C4,
    [u8 => absolute_difference_u8_c4]
);

impl_generic_binary_operation!(
    LogicalAndC1,
    logical_and,
    logical_and_c1,
    C1,
    [
        u8 => logical_and_u8_c1,
        u16 => logical_and_u16_c1,
        i32 => logical_and_i32_c1,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalAndC1InPlace,
    logical_and_in_place,
    logical_and_c1_in_place,
    C1,
    [
        u8 => logical_and_u8_c1_in_place,
        u16 => logical_and_u16_c1_in_place,
        i32 => logical_and_i32_c1_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalAndC3,
    logical_and,
    logical_and_c3,
    C3,
    [
        u8 => logical_and_u8_c3,
        u16 => logical_and_u16_c3,
        i32 => logical_and_i32_c3,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalAndC3InPlace,
    logical_and_in_place,
    logical_and_c3_in_place,
    C3,
    [
        u8 => logical_and_u8_c3_in_place,
        u16 => logical_and_u16_c3_in_place,
        i32 => logical_and_i32_c3_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalAndC4,
    logical_and,
    logical_and_c4,
    C4,
    [
        u8 => logical_and_u8_c4,
        u16 => logical_and_u16_c4,
        i32 => logical_and_i32_c4,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalAndC4InPlace,
    logical_and_in_place,
    logical_and_c4_in_place,
    C4,
    [
        u8 => logical_and_u8_c4_in_place,
        u16 => logical_and_u16_c4_in_place,
        i32 => logical_and_i32_c4_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalAndAc4,
    logical_and,
    logical_and_ac4,
    AC4,
    [
        u8 => logical_and_u8_ac4,
        u16 => logical_and_u16_ac4,
        i32 => logical_and_i32_ac4,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalAndAc4InPlace,
    logical_and_in_place,
    logical_and_ac4_in_place,
    AC4,
    [
        u8 => logical_and_u8_ac4_in_place,
        u16 => logical_and_u16_ac4_in_place,
        i32 => logical_and_i32_ac4_in_place,
    ]
);

impl_generic_binary_operation!(
    LogicalOrC1,
    logical_or,
    logical_or_c1,
    C1,
    [
        u8 => logical_or_u8_c1,
        u16 => logical_or_u16_c1,
        i32 => logical_or_i32_c1,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalOrC1InPlace,
    logical_or_in_place,
    logical_or_c1_in_place,
    C1,
    [
        u8 => logical_or_u8_c1_in_place,
        u16 => logical_or_u16_c1_in_place,
        i32 => logical_or_i32_c1_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalOrC3,
    logical_or,
    logical_or_c3,
    C3,
    [
        u8 => logical_or_u8_c3,
        u16 => logical_or_u16_c3,
        i32 => logical_or_i32_c3,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalOrC3InPlace,
    logical_or_in_place,
    logical_or_c3_in_place,
    C3,
    [
        u8 => logical_or_u8_c3_in_place,
        u16 => logical_or_u16_c3_in_place,
        i32 => logical_or_i32_c3_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalOrC4,
    logical_or,
    logical_or_c4,
    C4,
    [
        u8 => logical_or_u8_c4,
        u16 => logical_or_u16_c4,
        i32 => logical_or_i32_c4,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalOrC4InPlace,
    logical_or_in_place,
    logical_or_c4_in_place,
    C4,
    [
        u8 => logical_or_u8_c4_in_place,
        u16 => logical_or_u16_c4_in_place,
        i32 => logical_or_i32_c4_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalOrAc4,
    logical_or,
    logical_or_ac4,
    AC4,
    [
        u8 => logical_or_u8_ac4,
        u16 => logical_or_u16_ac4,
        i32 => logical_or_i32_ac4,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalOrAc4InPlace,
    logical_or_in_place,
    logical_or_ac4_in_place,
    AC4,
    [
        u8 => logical_or_u8_ac4_in_place,
        u16 => logical_or_u16_ac4_in_place,
        i32 => logical_or_i32_ac4_in_place,
    ]
);

impl_generic_binary_operation!(
    LogicalXorC1,
    logical_xor,
    logical_xor_c1,
    C1,
    [
        u8 => logical_xor_u8_c1,
        u16 => logical_xor_u16_c1,
        i32 => logical_xor_i32_c1,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalXorC1InPlace,
    logical_xor_in_place,
    logical_xor_c1_in_place,
    C1,
    [
        u8 => logical_xor_u8_c1_in_place,
        u16 => logical_xor_u16_c1_in_place,
        i32 => logical_xor_i32_c1_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalXorC3,
    logical_xor,
    logical_xor_c3,
    C3,
    [
        u8 => logical_xor_u8_c3,
        u16 => logical_xor_u16_c3,
        i32 => logical_xor_i32_c3,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalXorC3InPlace,
    logical_xor_in_place,
    logical_xor_c3_in_place,
    C3,
    [
        u8 => logical_xor_u8_c3_in_place,
        u16 => logical_xor_u16_c3_in_place,
        i32 => logical_xor_i32_c3_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalXorC4,
    logical_xor,
    logical_xor_c4,
    C4,
    [
        u8 => logical_xor_u8_c4,
        u16 => logical_xor_u16_c4,
        i32 => logical_xor_i32_c4,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalXorC4InPlace,
    logical_xor_in_place,
    logical_xor_c4_in_place,
    C4,
    [
        u8 => logical_xor_u8_c4_in_place,
        u16 => logical_xor_u16_c4_in_place,
        i32 => logical_xor_i32_c4_in_place,
    ]
);
impl_generic_binary_operation!(
    LogicalXorAc4,
    logical_xor,
    logical_xor_ac4,
    AC4,
    [
        u8 => logical_xor_u8_ac4,
        u16 => logical_xor_u16_ac4,
        i32 => logical_xor_i32_ac4,
    ]
);
impl_generic_binary_operation_in_place!(
    LogicalXorAc4InPlace,
    logical_xor_in_place,
    logical_xor_ac4_in_place,
    AC4,
    [
        u8 => logical_xor_u8_ac4_in_place,
        u16 => logical_xor_u16_ac4_in_place,
        i32 => logical_xor_i32_ac4_in_place,
    ]
);

impl_generic_unary_operation!(
    LogicalNotC1,
    logical_not,
    logical_not_c1,
    C1,
    [u8 => logical_not_u8_c1]
);
impl_generic_unary_operation_in_place!(
    LogicalNotC1InPlace,
    logical_not_in_place,
    logical_not_c1_in_place,
    C1,
    [u8 => logical_not_u8_c1_in_place]
);
impl_generic_unary_operation!(
    LogicalNotC3,
    logical_not,
    logical_not_c3,
    C3,
    [u8 => logical_not_u8_c3]
);
impl_generic_unary_operation_in_place!(
    LogicalNotC3InPlace,
    logical_not_in_place,
    logical_not_c3_in_place,
    C3,
    [u8 => logical_not_u8_c3_in_place]
);
impl_generic_unary_operation!(
    LogicalNotC4,
    logical_not,
    logical_not_c4,
    C4,
    [u8 => logical_not_u8_c4]
);
impl_generic_unary_operation_in_place!(
    LogicalNotC4InPlace,
    logical_not_in_place,
    logical_not_c4_in_place,
    C4,
    [u8 => logical_not_u8_c4_in_place]
);
impl_generic_unary_operation!(
    LogicalNotAc4,
    logical_not,
    logical_not_ac4,
    AC4,
    [u8 => logical_not_u8_ac4]
);
impl_generic_unary_operation_in_place!(
    LogicalNotAc4InPlace,
    logical_not_in_place,
    logical_not_ac4_in_place,
    AC4,
    [u8 => logical_not_u8_ac4_in_place]
);

impl_generic_unary_operation!(SquareC1, square, square_c1, C1, [
    f16 => square_f16_c1,
    f32 => square_f32_c1,
]);
impl_generic_unary_operation_in_place!(SquareC1InPlace, square_in_place, square_c1_in_place, C1, [
    f16 => square_f16_c1_in_place,
    f32 => square_f32_c1_in_place,
]);
impl_generic_scaled_unary_operation!(ScaledSquareC1, square_scaled, square_scaled_c1, C1, [
    u8 => square_u8_c1,
    u16 => square_u16_c1,
    i16 => square_i16_c1,
]);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareC1InPlace,
    square_scaled_in_place,
    square_scaled_c1_in_place,
    C1,
    [
        u8 => square_u8_c1_in_place,
        u16 => square_u16_c1_in_place,
        i16 => square_i16_c1_in_place,
    ]
);
impl_generic_unary_operation!(SquareC3, square, square_c3, C3, [
    f16 => square_f16_c3,
    f32 => square_f32_c3,
]);
impl_generic_unary_operation_in_place!(SquareC3InPlace, square_in_place, square_c3_in_place, C3, [
    f16 => square_f16_c3_in_place,
    f32 => square_f32_c3_in_place,
]);
impl_generic_scaled_unary_operation!(ScaledSquareC3, square_scaled, square_scaled_c3, C3, [
    u8 => square_u8_c3,
    u16 => square_u16_c3,
    i16 => square_i16_c3,
]);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareC3InPlace,
    square_scaled_in_place,
    square_scaled_c3_in_place,
    C3,
    [
        u8 => square_u8_c3_in_place,
        u16 => square_u16_c3_in_place,
        i16 => square_i16_c3_in_place,
    ]
);
impl_generic_unary_operation!(SquareC4, square, square_c4, C4, [
    f16 => square_f16_c4,
    f32 => square_f32_c4,
]);
impl_generic_unary_operation_in_place!(SquareC4InPlace, square_in_place, square_c4_in_place, C4, [
    f16 => square_f16_c4_in_place,
    f32 => square_f32_c4_in_place,
]);
impl_generic_scaled_unary_operation!(ScaledSquareC4, square_scaled, square_scaled_c4, C4, [
    u8 => square_u8_c4,
    u16 => square_u16_c4,
    i16 => square_i16_c4,
]);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareC4InPlace,
    square_scaled_in_place,
    square_scaled_c4_in_place,
    C4,
    [
        u8 => square_u8_c4_in_place,
        u16 => square_u16_c4_in_place,
        i16 => square_i16_c4_in_place,
    ]
);
impl_generic_unary_operation!(SquareAc4, square, square_ac4, AC4, [f32 => square_f32_ac4]);
impl_generic_unary_operation_in_place!(
    SquareAc4InPlace,
    square_in_place,
    square_ac4_in_place,
    AC4,
    [f32 => square_f32_ac4_in_place]
);
impl_generic_scaled_unary_operation!(ScaledSquareAc4, square_scaled, square_scaled_ac4, AC4, [
    u8 => square_u8_ac4,
    u16 => square_u16_ac4,
    i16 => square_i16_ac4,
]);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareAc4InPlace,
    square_scaled_in_place,
    square_scaled_ac4_in_place,
    AC4,
    [
        u8 => square_u8_ac4_in_place,
        u16 => square_u16_ac4_in_place,
        i16 => square_i16_ac4_in_place,
    ]
);

impl_generic_unary_operation!(SquareRootC1, square_root, square_root_c1, C1, [
    f16 => square_root_f16_c1,
    f32 => square_root_f32_c1,
]);
impl_generic_unary_operation_in_place!(
    SquareRootC1InPlace,
    square_root_in_place,
    square_root_c1_in_place,
    C1,
    [
        f16 => square_root_f16_c1_in_place,
        f32 => square_root_f32_c1_in_place,
    ]
);
impl_generic_scaled_unary_operation!(
    ScaledSquareRootC1,
    square_root_scaled,
    square_root_scaled_c1,
    C1,
    [
        u8 => square_root_u8_c1,
        u16 => square_root_u16_c1,
        i16 => square_root_i16_c1,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareRootC1InPlace,
    square_root_scaled_in_place,
    square_root_scaled_c1_in_place,
    C1,
    [
        u8 => square_root_u8_c1_in_place,
        u16 => square_root_u16_c1_in_place,
        i16 => square_root_i16_c1_in_place,
    ]
);
impl_generic_unary_operation!(SquareRootC3, square_root, square_root_c3, C3, [
    f16 => square_root_f16_c3,
    f32 => square_root_f32_c3,
]);
impl_generic_unary_operation_in_place!(
    SquareRootC3InPlace,
    square_root_in_place,
    square_root_c3_in_place,
    C3,
    [
        f16 => square_root_f16_c3_in_place,
        f32 => square_root_f32_c3_in_place,
    ]
);
impl_generic_scaled_unary_operation!(
    ScaledSquareRootC3,
    square_root_scaled,
    square_root_scaled_c3,
    C3,
    [
        u8 => square_root_u8_c3,
        u16 => square_root_u16_c3,
        i16 => square_root_i16_c3,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareRootC3InPlace,
    square_root_scaled_in_place,
    square_root_scaled_c3_in_place,
    C3,
    [
        u8 => square_root_u8_c3_in_place,
        u16 => square_root_u16_c3_in_place,
        i16 => square_root_i16_c3_in_place,
    ]
);
impl_generic_unary_operation!(SquareRootC4, square_root, square_root_c4, C4, [
    f16 => square_root_f16_c4,
    f32 => square_root_f32_c4,
]);
impl_generic_unary_operation_in_place!(
    SquareRootC4InPlace,
    square_root_in_place,
    square_root_c4_in_place,
    C4,
    [
        f16 => square_root_f16_c4_in_place,
        f32 => square_root_f32_c4_in_place,
    ]
);
impl_generic_unary_operation!(SquareRootAc4, square_root, square_root_ac4, AC4, [
    f32 => square_root_f32_ac4
]);
impl_generic_unary_operation_in_place!(
    SquareRootAc4InPlace,
    square_root_in_place,
    square_root_ac4_in_place,
    AC4,
    [f32 => square_root_f32_ac4_in_place]
);
impl_generic_scaled_unary_operation!(
    ScaledSquareRootAc4,
    square_root_scaled,
    square_root_scaled_ac4,
    AC4,
    [
        u8 => square_root_u8_ac4,
        u16 => square_root_u16_ac4,
        i16 => square_root_i16_ac4,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledSquareRootAc4InPlace,
    square_root_scaled_in_place,
    square_root_scaled_ac4_in_place,
    AC4,
    [
        u8 => square_root_u8_ac4_in_place,
        u16 => square_root_u16_ac4_in_place,
        i16 => square_root_i16_ac4_in_place,
    ]
);

impl_generic_unary_operation!(
    NaturalLogarithmC1,
    natural_logarithm,
    natural_logarithm_c1,
    C1,
    [
        f16 => natural_logarithm_f16_c1,
        f32 => natural_logarithm_f32_c1,
    ]
);
impl_generic_unary_operation_in_place!(
    NaturalLogarithmC1InPlace,
    natural_logarithm_in_place,
    natural_logarithm_c1_in_place,
    C1,
    [
        f16 => natural_logarithm_f16_c1_in_place,
        f32 => natural_logarithm_f32_c1_in_place,
    ]
);
impl_generic_scaled_unary_operation!(
    ScaledNaturalLogarithmC1,
    natural_logarithm_scaled,
    natural_logarithm_scaled_c1,
    C1,
    [
        u8 => natural_logarithm_u8_c1,
        u16 => natural_logarithm_u16_c1,
        i16 => natural_logarithm_i16_c1,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledNaturalLogarithmC1InPlace,
    natural_logarithm_scaled_in_place,
    natural_logarithm_scaled_c1_in_place,
    C1,
    [
        u8 => natural_logarithm_u8_c1_in_place,
        u16 => natural_logarithm_u16_c1_in_place,
        i16 => natural_logarithm_i16_c1_in_place,
    ]
);
impl_generic_unary_operation!(
    NaturalLogarithmC3,
    natural_logarithm,
    natural_logarithm_c3,
    C3,
    [
        f16 => natural_logarithm_f16_c3,
        f32 => natural_logarithm_f32_c3,
    ]
);
impl_generic_unary_operation_in_place!(
    NaturalLogarithmC3InPlace,
    natural_logarithm_in_place,
    natural_logarithm_c3_in_place,
    C3,
    [
        f16 => natural_logarithm_f16_c3_in_place,
        f32 => natural_logarithm_f32_c3_in_place,
    ]
);
impl_generic_scaled_unary_operation!(
    ScaledNaturalLogarithmC3,
    natural_logarithm_scaled,
    natural_logarithm_scaled_c3,
    C3,
    [
        u8 => natural_logarithm_u8_c3,
        u16 => natural_logarithm_u16_c3,
        i16 => natural_logarithm_i16_c3,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledNaturalLogarithmC3InPlace,
    natural_logarithm_scaled_in_place,
    natural_logarithm_scaled_c3_in_place,
    C3,
    [
        u8 => natural_logarithm_u8_c3_in_place,
        u16 => natural_logarithm_u16_c3_in_place,
        i16 => natural_logarithm_i16_c3_in_place,
    ]
);

impl_generic_unary_operation!(
    ExponentialC1,
    exponential,
    exponential_c1,
    C1,
    [f32 => exponential_f32_c1]
);
impl_generic_unary_operation_in_place!(
    ExponentialC1InPlace,
    exponential_in_place,
    exponential_c1_in_place,
    C1,
    [f32 => exponential_f32_c1_in_place]
);
impl_generic_scaled_unary_operation!(
    ScaledExponentialC1,
    exponential_scaled,
    exponential_scaled_c1,
    C1,
    [
        u8 => exponential_u8_c1,
        u16 => exponential_u16_c1,
        i16 => exponential_i16_c1,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledExponentialC1InPlace,
    exponential_scaled_in_place,
    exponential_scaled_c1_in_place,
    C1,
    [
        u8 => exponential_u8_c1_in_place,
        u16 => exponential_u16_c1_in_place,
        i16 => exponential_i16_c1_in_place,
    ]
);
impl_generic_unary_operation!(
    ExponentialC3,
    exponential,
    exponential_c3,
    C3,
    [f32 => exponential_f32_c3]
);
impl_generic_unary_operation_in_place!(
    ExponentialC3InPlace,
    exponential_in_place,
    exponential_c3_in_place,
    C3,
    [f32 => exponential_f32_c3_in_place]
);
impl_generic_scaled_unary_operation!(
    ScaledExponentialC3,
    exponential_scaled,
    exponential_scaled_c3,
    C3,
    [
        u8 => exponential_u8_c3,
        u16 => exponential_u16_c3,
        i16 => exponential_i16_c3,
    ]
);
impl_generic_scaled_unary_operation_in_place!(
    ScaledExponentialC3InPlace,
    exponential_scaled_in_place,
    exponential_scaled_c3_in_place,
    C3,
    [
        u8 => exponential_u8_c3_in_place,
        u16 => exponential_u16_c3_in_place,
        i16 => exponential_i16_c3_in_place,
    ]
);
