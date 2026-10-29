use crate::{
    context::StreamContext,
    error::Result,
    image::{
        raw::arithmetic::{
            and_constant_i32_ac4, and_constant_i32_ac4_in_place, and_constant_i32_c1,
            and_constant_i32_c1_in_place, and_constant_i32_c3, and_constant_i32_c3_in_place,
            and_constant_i32_c4, and_constant_i32_c4_in_place, and_constant_u8_ac4,
            and_constant_u8_ac4_in_place, and_constant_u8_c1, and_constant_u8_c1_in_place,
            and_constant_u8_c3, and_constant_u8_c3_in_place, and_constant_u8_c4,
            and_constant_u8_c4_in_place, and_constant_u16_ac4, and_constant_u16_ac4_in_place,
            and_constant_u16_c1, and_constant_u16_c1_in_place, and_constant_u16_c3,
            and_constant_u16_c3_in_place, and_constant_u16_c4, and_constant_u16_c4_in_place,
            left_shift_constant_i32_ac4, left_shift_constant_i32_ac4_in_place,
            left_shift_constant_i32_c1, left_shift_constant_i32_c1_in_place,
            left_shift_constant_i32_c3, left_shift_constant_i32_c3_in_place,
            left_shift_constant_i32_c4, left_shift_constant_i32_c4_in_place,
            left_shift_constant_u8_ac4, left_shift_constant_u8_ac4_in_place,
            left_shift_constant_u8_c1, left_shift_constant_u8_c1_in_place,
            left_shift_constant_u8_c3, left_shift_constant_u8_c3_in_place,
            left_shift_constant_u8_c4, left_shift_constant_u8_c4_in_place,
            left_shift_constant_u16_ac4, left_shift_constant_u16_ac4_in_place,
            left_shift_constant_u16_c1, left_shift_constant_u16_c1_in_place,
            left_shift_constant_u16_c3, left_shift_constant_u16_c3_in_place,
            left_shift_constant_u16_c4, left_shift_constant_u16_c4_in_place, or_constant_i32_ac4,
            or_constant_i32_ac4_in_place, or_constant_i32_c1, or_constant_i32_c1_in_place,
            or_constant_i32_c3, or_constant_i32_c3_in_place, or_constant_i32_c4,
            or_constant_i32_c4_in_place, or_constant_u8_ac4, or_constant_u8_ac4_in_place,
            or_constant_u8_c1, or_constant_u8_c1_in_place, or_constant_u8_c3,
            or_constant_u8_c3_in_place, or_constant_u8_c4, or_constant_u8_c4_in_place,
            or_constant_u16_ac4, or_constant_u16_ac4_in_place, or_constant_u16_c1,
            or_constant_u16_c1_in_place, or_constant_u16_c3, or_constant_u16_c3_in_place,
            or_constant_u16_c4, or_constant_u16_c4_in_place, right_shift_constant_i8_ac4,
            right_shift_constant_i8_ac4_in_place, right_shift_constant_i8_c1,
            right_shift_constant_i8_c1_in_place, right_shift_constant_i8_c3,
            right_shift_constant_i8_c3_in_place, right_shift_constant_i8_c4,
            right_shift_constant_i8_c4_in_place, right_shift_constant_i16_ac4,
            right_shift_constant_i16_ac4_in_place, right_shift_constant_i16_c1,
            right_shift_constant_i16_c1_in_place, right_shift_constant_i16_c3,
            right_shift_constant_i16_c3_in_place, right_shift_constant_i16_c4,
            right_shift_constant_i16_c4_in_place, right_shift_constant_i32_ac4,
            right_shift_constant_i32_ac4_in_place, right_shift_constant_i32_c1,
            right_shift_constant_i32_c1_in_place, right_shift_constant_i32_c3,
            right_shift_constant_i32_c3_in_place, right_shift_constant_i32_c4,
            right_shift_constant_i32_c4_in_place, right_shift_constant_u8_ac4,
            right_shift_constant_u8_ac4_in_place, right_shift_constant_u8_c1,
            right_shift_constant_u8_c1_in_place, right_shift_constant_u8_c3,
            right_shift_constant_u8_c3_in_place, right_shift_constant_u8_c4,
            right_shift_constant_u8_c4_in_place, right_shift_constant_u16_ac4,
            right_shift_constant_u16_ac4_in_place, right_shift_constant_u16_c1,
            right_shift_constant_u16_c1_in_place, right_shift_constant_u16_c3,
            right_shift_constant_u16_c3_in_place, right_shift_constant_u16_c4,
            right_shift_constant_u16_c4_in_place, xor_constant_i32_ac4,
            xor_constant_i32_ac4_in_place, xor_constant_i32_c1, xor_constant_i32_c1_in_place,
            xor_constant_i32_c3, xor_constant_i32_c3_in_place, xor_constant_i32_c4,
            xor_constant_i32_c4_in_place, xor_constant_u8_ac4, xor_constant_u8_ac4_in_place,
            xor_constant_u8_c1, xor_constant_u8_c1_in_place, xor_constant_u8_c3,
            xor_constant_u8_c3_in_place, xor_constant_u8_c4, xor_constant_u8_c4_in_place,
            xor_constant_u16_ac4, xor_constant_u16_ac4_in_place, xor_constant_u16_c1,
            xor_constant_u16_c1_in_place, xor_constant_u16_c3, xor_constant_u16_c3_in_place,
            xor_constant_u16_c4, xor_constant_u16_c4_in_place,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    types::DataTypeLike,
};

impl_generic_constant_scalar_operation!("Applies bitwise AND between each processed channel and its constant. Writes the result to `destination`.",
AndConstantC1, and_constant, and_constant_c1, C1, [
    u8, u8 => and_constant_u8_c1,
    u16, u16 => and_constant_u16_c1,
    i32, i32 => and_constant_i32_c1,
]);

impl_generic_constant_scalar_operation_in_place!("Applies bitwise AND between each processed channel and its constant. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise AND between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`.",
AndConstantC3, and_constant, and_constant_c3, C3, 3, [
    u8, u8 => and_constant_u8_c3,
    u16, u16 => and_constant_u16_c3,
    i32, i32 => and_constant_i32_c3,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise AND between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise AND between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`.",
AndConstantC4, and_constant, and_constant_c4, C4, 4, [
    u8, u8 => and_constant_u8_c4,
    u16, u16 => and_constant_u16_c4,
    i32, i32 => and_constant_i32_c4,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise AND between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise AND between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
AndConstantAc4, and_constant, and_constant_ac4, AC4, 3, [
    u8, u8 => and_constant_u8_ac4,
    u16, u16 => and_constant_u16_ac4,
    i32, i32 => and_constant_i32_ac4,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise AND between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Applies bitwise OR between each processed channel and its constant. Writes the result to `destination`.",
OrConstantC1, or_constant, or_constant_c1, C1, [
    u8, u8 => or_constant_u8_c1,
    u16, u16 => or_constant_u16_c1,
    i32, i32 => or_constant_i32_c1,
]);

impl_generic_constant_scalar_operation_in_place!("Applies bitwise OR between each processed channel and its constant. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise OR between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`.",
OrConstantC3, or_constant, or_constant_c3, C3, 3, [
    u8, u8 => or_constant_u8_c3,
    u16, u16 => or_constant_u16_c3,
    i32, i32 => or_constant_i32_c3,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise OR between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise OR between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`.",
OrConstantC4, or_constant, or_constant_c4, C4, 4, [
    u8, u8 => or_constant_u8_c4,
    u16, u16 => or_constant_u16_c4,
    i32, i32 => or_constant_i32_c4,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise OR between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise OR between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
OrConstantAc4, or_constant, or_constant_ac4, AC4, 3, [
    u8, u8 => or_constant_u8_ac4,
    u16, u16 => or_constant_u16_ac4,
    i32, i32 => or_constant_i32_ac4,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise OR between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Applies bitwise exclusive OR between each processed channel and its constant. Writes the result to `destination`.",
XorConstantC1, xor_constant, xor_constant_c1, C1, [
    u8, u8 => xor_constant_u8_c1,
    u16, u16 => xor_constant_u16_c1,
    i32, i32 => xor_constant_i32_c1,
]);

impl_generic_constant_scalar_operation_in_place!("Applies bitwise exclusive OR between each processed channel and its constant. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise exclusive OR between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`.",
XorConstantC3, xor_constant, xor_constant_c3, C3, 3, [
    u8, u8 => xor_constant_u8_c3,
    u16, u16 => xor_constant_u16_c3,
    i32, i32 => xor_constant_i32_c3,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise exclusive OR between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise exclusive OR between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`.",
XorConstantC4, xor_constant, xor_constant_c4, C4, 4, [
    u8, u8 => xor_constant_u8_c4,
    u16, u16 => xor_constant_u16_c4,
    i32, i32 => xor_constant_i32_c4,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise exclusive OR between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Applies bitwise exclusive OR between each processed channel and its constant. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
XorConstantAc4, xor_constant, xor_constant_ac4, AC4, 3, [
    u8, u8 => xor_constant_u8_ac4,
    u16, u16 => xor_constant_u16_ac4,
    i32, i32 => xor_constant_i32_ac4,
]);

impl_generic_constant_array_operation_in_place!("Applies bitwise exclusive OR between each processed channel and its constant. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Shifts each processed channel right by its constant bit count. Writes the result to `destination`.",

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

impl_generic_constant_scalar_operation_in_place!("Shifts each processed channel right by its constant bit count. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_constant_array_operation_in_place!("Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Shifts each processed channel left by its constant bit count. Writes the result to `destination`.",

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

impl_generic_constant_scalar_operation_in_place!("Shifts each processed channel left by its constant bit count. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Shifts each processed channel left by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Shifts each processed channel left by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Shifts each processed channel left by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Shifts each processed channel left by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Shifts each processed channel left by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_constant_array_operation_in_place!("Shifts each processed channel left by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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
