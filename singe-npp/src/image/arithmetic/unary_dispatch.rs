use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic::unary_ops::{
            absolute_difference_f16_c1, absolute_difference_f32_c1, absolute_difference_u8_c1,
            absolute_difference_u8_c3, absolute_difference_u8_c4, absolute_difference_u16_c1,
            absolute_f16_c1, absolute_f16_c1_in_place, absolute_f16_c3, absolute_f16_c3_in_place,
            absolute_f16_c4, absolute_f16_c4_in_place, absolute_f32_ac4, absolute_f32_ac4_in_place,
            absolute_f32_c1, absolute_f32_c1_in_place, absolute_f32_c3, absolute_f32_c3_in_place,
            absolute_f32_c4, absolute_f32_c4_in_place, absolute_i16_ac4, absolute_i16_ac4_in_place,
            absolute_i16_c1, absolute_i16_c1_in_place, absolute_i16_c3, absolute_i16_c3_in_place,
            absolute_i16_c4, absolute_i16_c4_in_place, exponential_f32_c1,
            exponential_f32_c1_in_place, exponential_f32_c3, exponential_f32_c3_in_place,
            exponential_i16_c1, exponential_i16_c1_in_place, exponential_i16_c3,
            exponential_i16_c3_in_place, exponential_u8_c1, exponential_u8_c1_in_place,
            exponential_u8_c3, exponential_u8_c3_in_place, exponential_u16_c1,
            exponential_u16_c1_in_place, exponential_u16_c3, exponential_u16_c3_in_place,
            natural_logarithm_f16_c1, natural_logarithm_f16_c1_in_place, natural_logarithm_f16_c3,
            natural_logarithm_f16_c3_in_place, natural_logarithm_f32_c1,
            natural_logarithm_f32_c1_in_place, natural_logarithm_f32_c3,
            natural_logarithm_f32_c3_in_place, natural_logarithm_i16_c1,
            natural_logarithm_i16_c1_in_place, natural_logarithm_i16_c3,
            natural_logarithm_i16_c3_in_place, natural_logarithm_u8_c1,
            natural_logarithm_u8_c1_in_place, natural_logarithm_u8_c3,
            natural_logarithm_u8_c3_in_place, natural_logarithm_u16_c1,
            natural_logarithm_u16_c1_in_place, natural_logarithm_u16_c3,
            natural_logarithm_u16_c3_in_place, square_f16_c1, square_f16_c1_in_place,
            square_f16_c3, square_f16_c3_in_place, square_f16_c4, square_f16_c4_in_place,
            square_f32_ac4, square_f32_ac4_in_place, square_f32_c1, square_f32_c1_in_place,
            square_f32_c3, square_f32_c3_in_place, square_f32_c4, square_f32_c4_in_place,
            square_i16_ac4, square_i16_ac4_in_place, square_i16_c1, square_i16_c1_in_place,
            square_i16_c3, square_i16_c3_in_place, square_i16_c4, square_i16_c4_in_place,
            square_root_f16_c1, square_root_f16_c1_in_place, square_root_f16_c3,
            square_root_f16_c3_in_place, square_root_f16_c4, square_root_f16_c4_in_place,
            square_root_f32_ac4, square_root_f32_ac4_in_place, square_root_f32_c1,
            square_root_f32_c1_in_place, square_root_f32_c3, square_root_f32_c3_in_place,
            square_root_f32_c4, square_root_f32_c4_in_place, square_root_i16_ac4,
            square_root_i16_ac4_in_place, square_root_i16_c1, square_root_i16_c1_in_place,
            square_root_i16_c3, square_root_i16_c3_in_place, square_root_u8_ac4,
            square_root_u8_ac4_in_place, square_root_u8_c1, square_root_u8_c1_in_place,
            square_root_u8_c3, square_root_u8_c3_in_place, square_root_u16_ac4,
            square_root_u16_ac4_in_place, square_root_u16_c1, square_root_u16_c1_in_place,
            square_root_u16_c3, square_root_u16_c3_in_place, square_u8_ac4, square_u8_ac4_in_place,
            square_u8_c1, square_u8_c1_in_place, square_u8_c3, square_u8_c3_in_place, square_u8_c4,
            square_u8_c4_in_place, square_u16_ac4, square_u16_ac4_in_place, square_u16_c1,
            square_u16_c1_in_place, square_u16_c3, square_u16_c3_in_place, square_u16_c4,
            square_u16_c4_in_place,
        },
        raw::arithmetic::{
            logical_and_i32_ac4, logical_and_i32_ac4_in_place, logical_and_i32_c1,
            logical_and_i32_c1_in_place, logical_and_i32_c3, logical_and_i32_c3_in_place,
            logical_and_i32_c4, logical_and_i32_c4_in_place, logical_and_u8_ac4,
            logical_and_u8_ac4_in_place, logical_and_u8_c1, logical_and_u8_c1_in_place,
            logical_and_u8_c3, logical_and_u8_c3_in_place, logical_and_u8_c4,
            logical_and_u8_c4_in_place, logical_and_u16_ac4, logical_and_u16_ac4_in_place,
            logical_and_u16_c1, logical_and_u16_c1_in_place, logical_and_u16_c3,
            logical_and_u16_c3_in_place, logical_and_u16_c4, logical_and_u16_c4_in_place,
            logical_not_u8_ac4, logical_not_u8_ac4_in_place, logical_not_u8_c1,
            logical_not_u8_c1_in_place, logical_not_u8_c3, logical_not_u8_c3_in_place,
            logical_not_u8_c4, logical_not_u8_c4_in_place, logical_or_i32_ac4,
            logical_or_i32_ac4_in_place, logical_or_i32_c1, logical_or_i32_c1_in_place,
            logical_or_i32_c3, logical_or_i32_c3_in_place, logical_or_i32_c4,
            logical_or_i32_c4_in_place, logical_or_u8_ac4, logical_or_u8_ac4_in_place,
            logical_or_u8_c1, logical_or_u8_c1_in_place, logical_or_u8_c3,
            logical_or_u8_c3_in_place, logical_or_u8_c4, logical_or_u8_c4_in_place,
            logical_or_u16_ac4, logical_or_u16_ac4_in_place, logical_or_u16_c1,
            logical_or_u16_c1_in_place, logical_or_u16_c3, logical_or_u16_c3_in_place,
            logical_or_u16_c4, logical_or_u16_c4_in_place, logical_xor_i32_ac4,
            logical_xor_i32_ac4_in_place, logical_xor_i32_c1, logical_xor_i32_c1_in_place,
            logical_xor_i32_c3, logical_xor_i32_c3_in_place, logical_xor_i32_c4,
            logical_xor_i32_c4_in_place, logical_xor_u8_ac4, logical_xor_u8_ac4_in_place,
            logical_xor_u8_c1, logical_xor_u8_c1_in_place, logical_xor_u8_c3,
            logical_xor_u8_c3_in_place, logical_xor_u8_c4, logical_xor_u8_c4_in_place,
            logical_xor_u16_ac4, logical_xor_u16_ac4_in_place, logical_xor_u16_c1,
            logical_xor_u16_c1_in_place, logical_xor_u16_c3, logical_xor_u16_c3_in_place,
            logical_xor_u16_c4, logical_xor_u16_c4_in_place,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    types::DataTypeLike,
};

impl_generic_unary_operation!("Computes the absolute value of each processed channel. Writes the result to `destination`.",

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

impl_generic_unary_operation_in_place!("Computes the absolute value of each processed channel. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the absolute value of each processed channel. Writes the result to `destination`.",

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

impl_generic_unary_operation_in_place!("Computes the absolute value of each processed channel. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the absolute value of each processed channel. Writes the result to `destination`.",

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

impl_generic_unary_operation_in_place!("Computes the absolute value of each processed channel. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the absolute value of each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    AbsoluteAc4,
    absolute,
    absolute_ac4,
    AC4,
    [
        i16 => absolute_i16_ac4,
        f32 => absolute_f32_ac4,
    ]
);

impl_generic_unary_operation_in_place!("Computes the absolute value of each processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    AbsoluteAc4InPlace,
    absolute_in_place,
    absolute_ac4_in_place,
    AC4,
    [
        i16 => absolute_i16_ac4_in_place,
        f32 => absolute_f32_ac4_in_place,
    ]
);

impl_generic_binary_operation!("Computes the absolute difference of corresponding channels in `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation!("Computes the absolute difference of corresponding channels in `source1` and `source2`. Writes the result to `destination`.",

    AbsoluteDifferenceC3,
    absolute_difference,
    absolute_difference_c3,
    C3,
    [u8 => absolute_difference_u8_c3]
);

impl_generic_binary_operation!("Computes the absolute difference of corresponding channels in `source1` and `source2`. Writes the result to `destination`.",

    AbsoluteDifferenceC4,
    absolute_difference,
    absolute_difference_c4,
    C4,
    [u8 => absolute_difference_u8_c4]
);

impl_generic_binary_operation!("Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_binary_operation_in_place!("Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_binary_operation!("Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_binary_operation_in_place!("Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_binary_operation!("Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",

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

impl_generic_binary_operation_in_place!("Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",

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

impl_generic_binary_operation!("Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_binary_operation_in_place!("Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_unary_operation!("Inverts the bits of each processed channel value. Writes the result to `destination`.",

    LogicalNotC1,
    logical_not,
    logical_not_c1,
    C1,
    [u8 => logical_not_u8_c1]
);

impl_generic_unary_operation_in_place!("Inverts the bits of each processed channel value. Updates `source_destination` in place.",

    LogicalNotC1InPlace,
    logical_not_in_place,
    logical_not_c1_in_place,
    C1,
    [u8 => logical_not_u8_c1_in_place]
);

impl_generic_unary_operation!("Inverts the bits of each processed channel value. Writes the result to `destination`.",

    LogicalNotC3,
    logical_not,
    logical_not_c3,
    C3,
    [u8 => logical_not_u8_c3]
);

impl_generic_unary_operation_in_place!("Inverts the bits of each processed channel value. Updates `source_destination` in place.",

    LogicalNotC3InPlace,
    logical_not_in_place,
    logical_not_c3_in_place,
    C3,
    [u8 => logical_not_u8_c3_in_place]
);

impl_generic_unary_operation!("Inverts the bits of each processed channel value. Writes the result to `destination`.",

    LogicalNotC4,
    logical_not,
    logical_not_c4,
    C4,
    [u8 => logical_not_u8_c4]
);

impl_generic_unary_operation_in_place!("Inverts the bits of each processed channel value. Updates `source_destination` in place.",

    LogicalNotC4InPlace,
    logical_not_in_place,
    logical_not_c4_in_place,
    C4,
    [u8 => logical_not_u8_c4_in_place]
);

impl_generic_unary_operation!("Inverts the bits of each processed channel value. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    LogicalNotAc4,
    logical_not,
    logical_not_ac4,
    AC4,
    [u8 => logical_not_u8_ac4]
);

impl_generic_unary_operation_in_place!("Inverts the bits of each processed channel value. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    LogicalNotAc4InPlace,
    logical_not_in_place,
    logical_not_ac4_in_place,
    AC4,
    [u8 => logical_not_u8_ac4_in_place]
);

impl_generic_unary_operation!("Squares each processed channel value. Writes the result to `destination`.",
SquareC1, square, square_c1, C1, [
    f16 => square_f16_c1,
    f32 => square_f32_c1,
]);

impl_generic_unary_operation_in_place!("Squares each processed channel value. Updates `source_destination` in place.",
SquareC1InPlace, square_in_place, square_c1_in_place, C1, [
    f16 => square_f16_c1_in_place,
    f32 => square_f32_c1_in_place,
]);

impl_generic_scaled_unary_operation!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
ScaledSquareC1, square_scaled, square_scaled_c1, C1, [
    u8 => square_u8_c1,
    u16 => square_u16_c1,
    i16 => square_i16_c1,
]);

impl_generic_scaled_unary_operation_in_place!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Squares each processed channel value. Writes the result to `destination`.",
SquareC3, square, square_c3, C3, [
    f16 => square_f16_c3,
    f32 => square_f32_c3,
]);

impl_generic_unary_operation_in_place!("Squares each processed channel value. Updates `source_destination` in place.",
SquareC3InPlace, square_in_place, square_c3_in_place, C3, [
    f16 => square_f16_c3_in_place,
    f32 => square_f32_c3_in_place,
]);

impl_generic_scaled_unary_operation!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
ScaledSquareC3, square_scaled, square_scaled_c3, C3, [
    u8 => square_u8_c3,
    u16 => square_u16_c3,
    i16 => square_i16_c3,
]);

impl_generic_scaled_unary_operation_in_place!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Squares each processed channel value. Writes the result to `destination`.",
SquareC4, square, square_c4, C4, [
    f16 => square_f16_c4,
    f32 => square_f32_c4,
]);

impl_generic_unary_operation_in_place!("Squares each processed channel value. Updates `source_destination` in place.",
SquareC4InPlace, square_in_place, square_c4_in_place, C4, [
    f16 => square_f16_c4_in_place,
    f32 => square_f32_c4_in_place,
]);

impl_generic_scaled_unary_operation!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
ScaledSquareC4, square_scaled, square_scaled_c4, C4, [
    u8 => square_u8_c4,
    u16 => square_u16_c4,
    i16 => square_i16_c4,
]);

impl_generic_scaled_unary_operation_in_place!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Squares each processed channel value. Writes the result to `destination`. Leaves the alpha channel unchanged.",
SquareAc4, square, square_ac4, AC4, [f32 => square_f32_ac4]);

impl_generic_unary_operation_in_place!("Squares each processed channel value. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    SquareAc4InPlace,
    square_in_place,
    square_ac4_in_place,
    AC4,
    [f32 => square_f32_ac4_in_place]
);

impl_generic_scaled_unary_operation!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
ScaledSquareAc4, square_scaled, square_scaled_ac4, AC4, [
    u8 => square_u8_ac4,
    u16 => square_u16_ac4,
    i16 => square_i16_ac4,
]);

impl_generic_scaled_unary_operation_in_place!("Squares each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_unary_operation!("Computes the square root of each processed channel value. Writes the result to `destination`.",
SquareRootC1, square_root, square_root_c1, C1, [
    f16 => square_root_f16_c1,
    f32 => square_root_f32_c1,
]);

impl_generic_unary_operation_in_place!("Computes the square root of each processed channel value. Updates `source_destination` in place.",

    SquareRootC1InPlace,
    square_root_in_place,
    square_root_c1_in_place,
    C1,
    [
        f16 => square_root_f16_c1_in_place,
        f32 => square_root_f32_c1_in_place,
    ]
);

impl_generic_scaled_unary_operation!("Computes the square root of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the square root of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the square root of each processed channel value. Writes the result to `destination`.",
SquareRootC3, square_root, square_root_c3, C3, [
    f16 => square_root_f16_c3,
    f32 => square_root_f32_c3,
]);

impl_generic_unary_operation_in_place!("Computes the square root of each processed channel value. Updates `source_destination` in place.",

    SquareRootC3InPlace,
    square_root_in_place,
    square_root_c3_in_place,
    C3,
    [
        f16 => square_root_f16_c3_in_place,
        f32 => square_root_f32_c3_in_place,
    ]
);

impl_generic_scaled_unary_operation!("Computes the square root of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the square root of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the square root of each processed channel value. Writes the result to `destination`.",
SquareRootC4, square_root, square_root_c4, C4, [
    f16 => square_root_f16_c4,
    f32 => square_root_f32_c4,
]);

impl_generic_unary_operation_in_place!("Computes the square root of each processed channel value. Updates `source_destination` in place.",

    SquareRootC4InPlace,
    square_root_in_place,
    square_root_c4_in_place,
    C4,
    [
        f16 => square_root_f16_c4_in_place,
        f32 => square_root_f32_c4_in_place,
    ]
);

impl_generic_unary_operation!("Computes the square root of each processed channel value. Writes the result to `destination`. Leaves the alpha channel unchanged.",
SquareRootAc4, square_root, square_root_ac4, AC4, [
    f32 => square_root_f32_ac4
]);

impl_generic_unary_operation_in_place!("Computes the square root of each processed channel value. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    SquareRootAc4InPlace,
    square_root_in_place,
    square_root_ac4_in_place,
    AC4,
    [f32 => square_root_f32_ac4_in_place]
);

impl_generic_scaled_unary_operation!("Computes the square root of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the square root of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_unary_operation!("Computes the natural logarithm of each processed channel value. Writes the result to `destination`.",

    NaturalLogarithmC1,
    natural_logarithm,
    natural_logarithm_c1,
    C1,
    [
        f16 => natural_logarithm_f16_c1,
        f32 => natural_logarithm_f32_c1,
    ]
);

impl_generic_unary_operation_in_place!("Computes the natural logarithm of each processed channel value. Updates `source_destination` in place.",

    NaturalLogarithmC1InPlace,
    natural_logarithm_in_place,
    natural_logarithm_c1_in_place,
    C1,
    [
        f16 => natural_logarithm_f16_c1_in_place,
        f32 => natural_logarithm_f32_c1_in_place,
    ]
);

impl_generic_scaled_unary_operation!("Computes the natural logarithm of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the natural logarithm of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the natural logarithm of each processed channel value. Writes the result to `destination`.",

    NaturalLogarithmC3,
    natural_logarithm,
    natural_logarithm_c3,
    C3,
    [
        f16 => natural_logarithm_f16_c3,
        f32 => natural_logarithm_f32_c3,
    ]
);

impl_generic_unary_operation_in_place!("Computes the natural logarithm of each processed channel value. Updates `source_destination` in place.",

    NaturalLogarithmC3InPlace,
    natural_logarithm_in_place,
    natural_logarithm_c3_in_place,
    C3,
    [
        f16 => natural_logarithm_f16_c3_in_place,
        f32 => natural_logarithm_f32_c3_in_place,
    ]
);

impl_generic_scaled_unary_operation!("Computes the natural logarithm of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the natural logarithm of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the exponential of each processed channel value. Writes the result to `destination`.",

    ExponentialC1,
    exponential,
    exponential_c1,
    C1,
    [f32 => exponential_f32_c1]
);

impl_generic_unary_operation_in_place!("Computes the exponential of each processed channel value. Updates `source_destination` in place.",

    ExponentialC1InPlace,
    exponential_in_place,
    exponential_c1_in_place,
    C1,
    [f32 => exponential_f32_c1_in_place]
);

impl_generic_scaled_unary_operation!("Computes the exponential of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the exponential of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_unary_operation!("Computes the exponential of each processed channel value. Writes the result to `destination`.",

    ExponentialC3,
    exponential,
    exponential_c3,
    C3,
    [f32 => exponential_f32_c3]
);

impl_generic_unary_operation_in_place!("Computes the exponential of each processed channel value. Updates `source_destination` in place.",

    ExponentialC3InPlace,
    exponential_in_place,
    exponential_c3_in_place,
    C3,
    [f32 => exponential_f32_c3_in_place]
);

impl_generic_scaled_unary_operation!("Computes the exponential of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_unary_operation_in_place!("Computes the exponential of each processed channel value. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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
