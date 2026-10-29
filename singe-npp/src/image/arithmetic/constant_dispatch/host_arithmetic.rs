use singe_cuda::types::{Complex32, f16};

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        raw::arithmetic::{
            add_constant_f16_c1, add_constant_f16_c1_in_place, add_constant_f16_c3,
            add_constant_f16_c3_in_place, add_constant_f16_c4, add_constant_f16_c4_in_place,
            add_constant_f32_ac4, add_constant_f32_ac4_in_place, add_constant_f32_c1,
            add_constant_f32_c1_in_place, add_constant_f32_c3, add_constant_f32_c3_in_place,
            add_constant_f32_c4, add_constant_f32_c4_in_place, add_constant_f32_complex_ac4,
            add_constant_f32_complex_ac4_in_place, add_constant_f32_complex_c1,
            add_constant_f32_complex_c1_in_place, add_constant_f32_complex_c3,
            add_constant_f32_complex_c3_in_place, add_constant_f32_complex_c4,
            add_constant_f32_complex_c4_in_place, add_constant_i16_ac4,
            add_constant_i16_ac4_in_place, add_constant_i16_c1, add_constant_i16_c1_in_place,
            add_constant_i16_c3, add_constant_i16_c3_in_place, add_constant_i16_c4,
            add_constant_i16_c4_in_place, add_constant_i16_complex_ac4,
            add_constant_i16_complex_ac4_in_place, add_constant_i16_complex_c1,
            add_constant_i16_complex_c1_in_place, add_constant_i16_complex_c3,
            add_constant_i16_complex_c3_in_place, add_constant_i32_c1,
            add_constant_i32_c1_in_place, add_constant_i32_c3, add_constant_i32_c3_in_place,
            add_constant_i32_complex_ac4, add_constant_i32_complex_ac4_in_place,
            add_constant_i32_complex_c1, add_constant_i32_complex_c1_in_place,
            add_constant_i32_complex_c3, add_constant_i32_complex_c3_in_place, add_constant_u8_ac4,
            add_constant_u8_ac4_in_place, add_constant_u8_c1, add_constant_u8_c1_in_place,
            add_constant_u8_c3, add_constant_u8_c3_in_place, add_constant_u8_c4,
            add_constant_u8_c4_in_place, add_constant_u16_ac4, add_constant_u16_ac4_in_place,
            add_constant_u16_c1, add_constant_u16_c1_in_place, add_constant_u16_c3,
            add_constant_u16_c3_in_place, add_constant_u16_c4, add_constant_u16_c4_in_place,
            divide_constant_f16_c1, divide_constant_f16_c1_in_place, divide_constant_f16_c3,
            divide_constant_f16_c3_in_place, divide_constant_f16_c4,
            divide_constant_f16_c4_in_place, divide_constant_f32_ac4,
            divide_constant_f32_ac4_in_place, divide_constant_f32_c1,
            divide_constant_f32_c1_in_place, divide_constant_f32_c3,
            divide_constant_f32_c3_in_place, divide_constant_f32_c4,
            divide_constant_f32_c4_in_place, divide_constant_f32_complex_ac4,
            divide_constant_f32_complex_ac4_in_place, divide_constant_f32_complex_c1,
            divide_constant_f32_complex_c1_in_place, divide_constant_f32_complex_c3,
            divide_constant_f32_complex_c3_in_place, divide_constant_f32_complex_c4,
            divide_constant_f32_complex_c4_in_place, divide_constant_i16_ac4,
            divide_constant_i16_ac4_in_place, divide_constant_i16_c1,
            divide_constant_i16_c1_in_place, divide_constant_i16_c3,
            divide_constant_i16_c3_in_place, divide_constant_i16_c4,
            divide_constant_i16_c4_in_place, divide_constant_i16_complex_ac4,
            divide_constant_i16_complex_ac4_in_place, divide_constant_i16_complex_c1,
            divide_constant_i16_complex_c1_in_place, divide_constant_i16_complex_c3,
            divide_constant_i16_complex_c3_in_place, divide_constant_i32_c1,
            divide_constant_i32_c1_in_place, divide_constant_i32_c3,
            divide_constant_i32_c3_in_place, divide_constant_i32_complex_ac4,
            divide_constant_i32_complex_ac4_in_place, divide_constant_i32_complex_c1,
            divide_constant_i32_complex_c1_in_place, divide_constant_i32_complex_c3,
            divide_constant_i32_complex_c3_in_place, divide_constant_u8_ac4,
            divide_constant_u8_ac4_in_place, divide_constant_u8_c1, divide_constant_u8_c1_in_place,
            divide_constant_u8_c3, divide_constant_u8_c3_in_place, divide_constant_u8_c4,
            divide_constant_u8_c4_in_place, divide_constant_u16_ac4,
            divide_constant_u16_ac4_in_place, divide_constant_u16_c1,
            divide_constant_u16_c1_in_place, divide_constant_u16_c3,
            divide_constant_u16_c3_in_place, divide_constant_u16_c4,
            divide_constant_u16_c4_in_place, multiply_constant_f16_c1,
            multiply_constant_f16_c1_in_place, multiply_constant_f16_c3,
            multiply_constant_f16_c3_in_place, multiply_constant_f16_c4,
            multiply_constant_f16_c4_in_place, multiply_constant_f32_ac4,
            multiply_constant_f32_ac4_in_place, multiply_constant_f32_c1,
            multiply_constant_f32_c1_in_place, multiply_constant_f32_c3,
            multiply_constant_f32_c3_in_place, multiply_constant_f32_c4,
            multiply_constant_f32_c4_in_place, multiply_constant_f32_complex_ac4,
            multiply_constant_f32_complex_ac4_in_place, multiply_constant_f32_complex_c1,
            multiply_constant_f32_complex_c1_in_place, multiply_constant_f32_complex_c3,
            multiply_constant_f32_complex_c3_in_place, multiply_constant_f32_complex_c4,
            multiply_constant_f32_complex_c4_in_place, multiply_constant_i16_ac4,
            multiply_constant_i16_ac4_in_place, multiply_constant_i16_c1,
            multiply_constant_i16_c1_in_place, multiply_constant_i16_c3,
            multiply_constant_i16_c3_in_place, multiply_constant_i16_c4,
            multiply_constant_i16_c4_in_place, multiply_constant_i16_complex_ac4,
            multiply_constant_i16_complex_ac4_in_place, multiply_constant_i16_complex_c1,
            multiply_constant_i16_complex_c1_in_place, multiply_constant_i16_complex_c3,
            multiply_constant_i16_complex_c3_in_place, multiply_constant_i32_c1,
            multiply_constant_i32_c1_in_place, multiply_constant_i32_c3,
            multiply_constant_i32_c3_in_place, multiply_constant_i32_complex_ac4,
            multiply_constant_i32_complex_ac4_in_place, multiply_constant_i32_complex_c1,
            multiply_constant_i32_complex_c1_in_place, multiply_constant_i32_complex_c3,
            multiply_constant_i32_complex_c3_in_place, multiply_constant_u8_ac4,
            multiply_constant_u8_ac4_in_place, multiply_constant_u8_c1,
            multiply_constant_u8_c1_in_place, multiply_constant_u8_c3,
            multiply_constant_u8_c3_in_place, multiply_constant_u8_c4,
            multiply_constant_u8_c4_in_place, multiply_constant_u16_ac4,
            multiply_constant_u16_ac4_in_place, multiply_constant_u16_c1,
            multiply_constant_u16_c1_in_place, multiply_constant_u16_c3,
            multiply_constant_u16_c3_in_place, multiply_constant_u16_c4,
            multiply_constant_u16_c4_in_place, subtract_constant_f16_c1,
            subtract_constant_f16_c1_in_place, subtract_constant_f16_c3,
            subtract_constant_f16_c3_in_place, subtract_constant_f16_c4,
            subtract_constant_f16_c4_in_place, subtract_constant_f32_ac4,
            subtract_constant_f32_ac4_in_place, subtract_constant_f32_c1,
            subtract_constant_f32_c1_in_place, subtract_constant_f32_c3,
            subtract_constant_f32_c3_in_place, subtract_constant_f32_c4,
            subtract_constant_f32_c4_in_place, subtract_constant_f32_complex_ac4,
            subtract_constant_f32_complex_ac4_in_place, subtract_constant_f32_complex_c1,
            subtract_constant_f32_complex_c1_in_place, subtract_constant_f32_complex_c3,
            subtract_constant_f32_complex_c3_in_place, subtract_constant_f32_complex_c4,
            subtract_constant_f32_complex_c4_in_place, subtract_constant_i16_ac4,
            subtract_constant_i16_ac4_in_place, subtract_constant_i16_c1,
            subtract_constant_i16_c1_in_place, subtract_constant_i16_c3,
            subtract_constant_i16_c3_in_place, subtract_constant_i16_c4,
            subtract_constant_i16_c4_in_place, subtract_constant_i16_complex_ac4,
            subtract_constant_i16_complex_ac4_in_place, subtract_constant_i16_complex_c1,
            subtract_constant_i16_complex_c1_in_place, subtract_constant_i16_complex_c3,
            subtract_constant_i16_complex_c3_in_place, subtract_constant_i32_c1,
            subtract_constant_i32_c1_in_place, subtract_constant_i32_c3,
            subtract_constant_i32_c3_in_place, subtract_constant_i32_complex_ac4,
            subtract_constant_i32_complex_ac4_in_place, subtract_constant_i32_complex_c1,
            subtract_constant_i32_complex_c1_in_place, subtract_constant_i32_complex_c3,
            subtract_constant_i32_complex_c3_in_place, subtract_constant_u8_ac4,
            subtract_constant_u8_ac4_in_place, subtract_constant_u8_c1,
            subtract_constant_u8_c1_in_place, subtract_constant_u8_c3,
            subtract_constant_u8_c3_in_place, subtract_constant_u8_c4,
            subtract_constant_u8_c4_in_place, subtract_constant_u16_ac4,
            subtract_constant_u16_ac4_in_place, subtract_constant_u16_c1,
            subtract_constant_u16_c1_in_place, subtract_constant_u16_c3,
            subtract_constant_u16_c3_in_place, subtract_constant_u16_c4,
            subtract_constant_u16_c4_in_place,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    types::{ComplexI16, ComplexI32, DataTypeLike},
};

impl_generic_constant_scalar_operation!("Adds a constant to each processed channel. Writes the result to `destination`.",
AddConstantC1, add_constant, add_constant_c1, C1, [
    f16, f32 => add_constant_f16_c1,
    f32, f32 => add_constant_f32_c1,
    Complex32, Complex32 => add_constant_f32_complex_c1,
]);

impl_generic_constant_scalar_operation_in_place!("Adds a constant to each processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_scalar_operation!("Adds a constant to each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_scalar_operation_in_place!("Adds a constant to each processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Adds a constant to each processed channel. Uses one constant per processed channel. Writes the result to `destination`.",
AddConstantC3, add_constant, add_constant_c3, C3, 3, [
    f16, f32 => add_constant_f16_c3,
    f32, f32 => add_constant_f32_c3,
    Complex32, Complex32 => add_constant_f32_complex_c3,
]);

impl_generic_constant_array_operation_in_place!("Adds a constant to each processed channel. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Adds a constant to each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Adds a constant to each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Adds a constant to each processed channel. Uses one constant per processed channel. Writes the result to `destination`.",
AddConstantC4, add_constant, add_constant_c4, C4, 4, [
    f16, f32 => add_constant_f16_c4,
    f32, f32 => add_constant_f32_c4,
    Complex32, Complex32 => add_constant_f32_complex_c4,
]);

impl_generic_constant_array_operation_in_place!("Adds a constant to each processed channel. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Adds a constant to each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Adds a constant to each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Adds a constant to each processed channel. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
AddConstantAc4, add_constant, add_constant_ac4, AC4, 3, [
    f32, f32 => add_constant_f32_ac4,
    Complex32, Complex32 => add_constant_f32_complex_ac4,
]);

impl_generic_constant_array_operation_in_place!("Adds a constant to each processed channel. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation!("Adds a constant to each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation_in_place!("Adds a constant to each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Multiplies each processed channel by a constant. Writes the result to `destination`.",

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

impl_generic_constant_scalar_operation_in_place!("Multiplies each processed channel by a constant. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_scalar_operation!("Multiplies each processed channel by a constant. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_scalar_operation_in_place!("Multiplies each processed channel by a constant. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_constant_array_operation_in_place!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation_in_place!("Multiplies each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Subtracts a constant from each processed channel. Writes the result to `destination`.",

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

impl_generic_constant_scalar_operation_in_place!("Subtracts a constant from each processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_scalar_operation!("Subtracts a constant from each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_scalar_operation_in_place!("Subtracts a constant from each processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Writes the result to `destination`.",

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

impl_generic_constant_array_operation_in_place!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_constant_array_operation_in_place!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation_in_place!("Subtracts a constant from each processed channel. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_constant_scalar_operation!("Divides each processed channel by a constant. Writes the result to `destination`.",
DivideConstantC1, divide_constant, divide_constant_c1, C1, [
    f16, f32 => divide_constant_f16_c1,
    f32, f32 => divide_constant_f32_c1,
    Complex32, Complex32 => divide_constant_f32_complex_c1,
]);

impl_generic_constant_scalar_operation_in_place!("Divides each processed channel by a constant. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_scalar_operation!("Divides each processed channel by a constant. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_scalar_operation_in_place!("Divides each processed channel by a constant. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Divides each processed channel by a constant. Uses one constant per processed channel. Writes the result to `destination`.",
DivideConstantC3, divide_constant, divide_constant_c3, C3, 3, [
    f16, f32 => divide_constant_f16_c3,
    f32, f32 => divide_constant_f32_c3,
    Complex32, Complex32 => divide_constant_f32_complex_c3,
]);

impl_generic_constant_array_operation_in_place!("Divides each processed channel by a constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Divides each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Divides each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Divides each processed channel by a constant. Uses one constant per processed channel. Writes the result to `destination`.",
DivideConstantC4, divide_constant, divide_constant_c4, C4, 4, [
    f16, f32 => divide_constant_f16_c4,
    f32, f32 => divide_constant_f32_c4,
    Complex32, Complex32 => divide_constant_f32_complex_c4,
]);

impl_generic_constant_array_operation_in_place!("Divides each processed channel by a constant. Uses one constant per processed channel. Updates `source_destination` in place.",

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

impl_generic_scaled_constant_array_operation!("Divides each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_constant_array_operation_in_place!("Divides each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_constant_array_operation!("Divides each processed channel by a constant. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_constant_array_operation_in_place!("Divides each processed channel by a constant. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation!("Divides each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_constant_array_operation_in_place!("Divides each processed channel by a constant. Uses one constant per processed channel. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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
