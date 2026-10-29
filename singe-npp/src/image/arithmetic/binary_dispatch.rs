use singe_cuda::types::{Complex32, f16};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic::binary_ops::{
            divide_f16_c1, divide_f16_c1_in_place, divide_f16_c3, divide_f16_c3_in_place,
            divide_f16_c4, divide_f16_c4_in_place, divide_f32_ac4, divide_f32_ac4_in_place,
            divide_f32_c1, divide_f32_c1_in_place, divide_f32_c3, divide_f32_c3_in_place,
            divide_f32_c4, divide_f32_c4_in_place, divide_f32_complex_ac4,
            divide_f32_complex_ac4_in_place, divide_f32_complex_c1, divide_f32_complex_c1_in_place,
            divide_f32_complex_c3, divide_f32_complex_c3_in_place, divide_f32_complex_c4,
            divide_f32_complex_c4_in_place, divide_i16_ac4, divide_i16_ac4_in_place, divide_i16_c1,
            divide_i16_c1_in_place, divide_i16_c3, divide_i16_c3_in_place, divide_i16_c4,
            divide_i16_c4_in_place, divide_i16_complex_ac4, divide_i16_complex_ac4_in_place,
            divide_i16_complex_c1, divide_i16_complex_c1_in_place, divide_i16_complex_c3,
            divide_i16_complex_c3_in_place, divide_i32_c1, divide_i32_c1_in_place,
            divide_i32_c1_unscaled, divide_i32_c3, divide_i32_c3_in_place, divide_i32_complex_ac4,
            divide_i32_complex_ac4_in_place, divide_i32_complex_c1, divide_i32_complex_c1_in_place,
            divide_i32_complex_c3, divide_i32_complex_c3_in_place, divide_round_i16_ac4,
            divide_round_i16_ac4_in_place, divide_round_i16_c1, divide_round_i16_c1_in_place,
            divide_round_i16_c3, divide_round_i16_c3_in_place, divide_round_i16_c4,
            divide_round_i16_c4_in_place, divide_round_u8_ac4, divide_round_u8_ac4_in_place,
            divide_round_u8_c1, divide_round_u8_c1_in_place, divide_round_u8_c3,
            divide_round_u8_c3_in_place, divide_round_u8_c4, divide_round_u8_c4_in_place,
            divide_round_u16_ac4, divide_round_u16_ac4_in_place, divide_round_u16_c1,
            divide_round_u16_c1_in_place, divide_round_u16_c3, divide_round_u16_c3_in_place,
            divide_round_u16_c4, divide_round_u16_c4_in_place, divide_u8_ac4,
            divide_u8_ac4_in_place, divide_u8_c1, divide_u8_c1_in_place, divide_u8_c3,
            divide_u8_c3_in_place, divide_u8_c4, divide_u8_c4_in_place, divide_u16_ac4,
            divide_u16_ac4_in_place, divide_u16_c1, divide_u16_c1_in_place, divide_u16_c3,
            divide_u16_c3_in_place, divide_u16_c4, divide_u16_c4_in_place, multiply_f16_c1,
            multiply_f16_c1_in_place, multiply_f16_c3, multiply_f16_c3_in_place, multiply_f16_c4,
            multiply_f16_c4_in_place, multiply_f32_ac4, multiply_f32_ac4_in_place, multiply_f32_c1,
            multiply_f32_c1_in_place, multiply_f32_c3, multiply_f32_c3_in_place, multiply_f32_c4,
            multiply_f32_c4_in_place, multiply_f32_complex_ac4, multiply_f32_complex_ac4_in_place,
            multiply_f32_complex_c1, multiply_f32_complex_c1_in_place, multiply_f32_complex_c3,
            multiply_f32_complex_c3_in_place, multiply_f32_complex_c4,
            multiply_f32_complex_c4_in_place, multiply_i16_ac4, multiply_i16_ac4_in_place,
            multiply_i16_c1, multiply_i16_c1_in_place, multiply_i16_c3, multiply_i16_c3_in_place,
            multiply_i16_c4, multiply_i16_c4_in_place, multiply_i16_complex_ac4,
            multiply_i16_complex_ac4_in_place, multiply_i16_complex_c1,
            multiply_i16_complex_c1_in_place, multiply_i16_complex_c3,
            multiply_i16_complex_c3_in_place, multiply_i32_c1, multiply_i32_c1_in_place,
            multiply_i32_c1_unscaled, multiply_i32_c3, multiply_i32_c3_in_place,
            multiply_i32_complex_ac4, multiply_i32_complex_ac4_in_place, multiply_i32_complex_c1,
            multiply_i32_complex_c1_in_place, multiply_i32_complex_c3,
            multiply_i32_complex_c3_in_place, multiply_scale_u8_ac4,
            multiply_scale_u8_ac4_in_place, multiply_scale_u8_c1, multiply_scale_u8_c1_in_place,
            multiply_scale_u8_c3, multiply_scale_u8_c3_in_place, multiply_scale_u8_c4,
            multiply_scale_u8_c4_in_place, multiply_scale_u16_ac4, multiply_scale_u16_ac4_in_place,
            multiply_scale_u16_c1, multiply_scale_u16_c1_in_place, multiply_scale_u16_c3,
            multiply_scale_u16_c3_in_place, multiply_scale_u16_c4, multiply_scale_u16_c4_in_place,
            multiply_u8_ac4, multiply_u8_ac4_in_place, multiply_u8_c1, multiply_u8_c1_in_place,
            multiply_u8_c3, multiply_u8_c3_in_place, multiply_u8_c4, multiply_u8_c4_in_place,
            multiply_u16_ac4, multiply_u16_ac4_in_place, multiply_u16_c1, multiply_u16_c1_in_place,
            multiply_u16_c3, multiply_u16_c3_in_place, multiply_u16_c4, multiply_u16_c4_in_place,
            subtract_f16_c1, subtract_f16_c1_in_place, subtract_f16_c3, subtract_f16_c3_in_place,
            subtract_f16_c4, subtract_f16_c4_in_place, subtract_f32_ac4, subtract_f32_ac4_in_place,
            subtract_f32_c1, subtract_f32_c1_in_place, subtract_f32_c3, subtract_f32_c3_in_place,
            subtract_f32_c4, subtract_f32_c4_in_place, subtract_f32_complex_ac4,
            subtract_f32_complex_ac4_in_place, subtract_f32_complex_c1,
            subtract_f32_complex_c1_in_place, subtract_f32_complex_c3,
            subtract_f32_complex_c3_in_place, subtract_f32_complex_c4,
            subtract_f32_complex_c4_in_place, subtract_i16_ac4, subtract_i16_ac4_in_place,
            subtract_i16_c1, subtract_i16_c1_in_place, subtract_i16_c3, subtract_i16_c3_in_place,
            subtract_i16_c4, subtract_i16_c4_in_place, subtract_i16_complex_ac4,
            subtract_i16_complex_ac4_in_place, subtract_i16_complex_c1,
            subtract_i16_complex_c1_in_place, subtract_i16_complex_c3,
            subtract_i16_complex_c3_in_place, subtract_i32_c1, subtract_i32_c1_in_place,
            subtract_i32_c1_unscaled, subtract_i32_c3, subtract_i32_c3_in_place, subtract_i32_c4,
            subtract_i32_c4_in_place, subtract_i32_complex_ac4, subtract_i32_complex_ac4_in_place,
            subtract_i32_complex_c1, subtract_i32_complex_c1_in_place, subtract_i32_complex_c3,
            subtract_i32_complex_c3_in_place, subtract_u8_ac4, subtract_u8_ac4_in_place,
            subtract_u8_c1, subtract_u8_c1_in_place, subtract_u8_c3, subtract_u8_c3_in_place,
            subtract_u8_c4, subtract_u8_c4_in_place, subtract_u16_ac4, subtract_u16_ac4_in_place,
            subtract_u16_c1, subtract_u16_c1_in_place, subtract_u16_c3, subtract_u16_c3_in_place,
            subtract_u16_c4, subtract_u16_c4_in_place,
        },
        arithmetic_validation::validate_same_size,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{ComplexI16, ComplexI32, DataTypeLike, RoundMode},
};

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_u8_c1,
    u8,
    C1,
    nppiAdd_8u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_u8_c1_in_place,
    u8,
    C1,
    nppiAdd_8u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_u8_c3,
    u8,
    C3,
    nppiAdd_8u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_u8_c3_in_place,
    u8,
    C3,
    nppiAdd_8u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_u8_ac4,
    u8,
    AC4,
    nppiAdd_8u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_u8_ac4_in_place,
    u8,
    AC4,
    nppiAdd_8u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_u8_c4,
    u8,
    C4,
    nppiAdd_8u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_u8_c4_in_place,
    u8,
    C4,
    nppiAdd_8u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_u16_c1,
    u16,
    C1,
    nppiAdd_16u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_u16_c1_in_place,
    u16,
    C1,
    nppiAdd_16u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_u16_c3,
    u16,
    C3,
    nppiAdd_16u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_u16_c3_in_place,
    u16,
    C3,
    nppiAdd_16u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_u16_ac4,
    u16,
    AC4,
    nppiAdd_16u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_u16_ac4_in_place,
    u16,
    AC4,
    nppiAdd_16u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_u16_c4,
    u16,
    C4,
    nppiAdd_16u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_u16_c4_in_place,
    u16,
    C4,
    nppiAdd_16u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i16_c1,
    i16,
    C1,
    nppiAdd_16s_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i16_c1_in_place,
    i16,
    C1,
    nppiAdd_16s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i16_c3,
    i16,
    C3,
    nppiAdd_16s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i16_c3_in_place,
    i16,
    C3,
    nppiAdd_16s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_i16_ac4,
    i16,
    AC4,
    nppiAdd_16s_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_i16_ac4_in_place,
    i16,
    AC4,
    nppiAdd_16s_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i16_c4,
    i16,
    C4,
    nppiAdd_16s_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i16_c4_in_place,
    i16,
    C4,
    nppiAdd_16s_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i16_complex_c1,
    ComplexI16,
    C1,
    nppiAdd_16sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i16_complex_c1_in_place,
    ComplexI16,
    C1,
    nppiAdd_16sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i16_complex_c3,
    ComplexI16,
    C3,
    nppiAdd_16sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i16_complex_c3_in_place,
    ComplexI16,
    C3,
    nppiAdd_16sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_i16_complex_ac4,
    ComplexI16,
    AC4,
    nppiAdd_16sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_i16_complex_ac4_in_place,
    ComplexI16,
    AC4,
    nppiAdd_16sc_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i32_c1,
    i32,
    C1,
    nppiAdd_32s_C1RSfs_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_i32_c1_unscaled,
    i32,
    C1,
    nppiAdd_32s_C1R_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i32_c1_in_place,
    i32,
    C1,
    nppiAdd_32s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i32_c3,
    i32,
    C3,
    nppiAdd_32s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i32_c3_in_place,
    i32,
    C3,
    nppiAdd_32s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i32_complex_c1,
    ComplexI32,
    C1,
    nppiAdd_32sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i32_complex_c1_in_place,
    ComplexI32,
    C1,
    nppiAdd_32sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    add_i32_complex_c3,
    ComplexI32,
    C3,
    nppiAdd_32sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    add_i32_complex_c3_in_place,
    ComplexI32,
    C3,
    nppiAdd_32sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_i32_complex_ac4,
    ComplexI32,
    AC4,
    nppiAdd_32sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_i32_complex_ac4_in_place,
    ComplexI32,
    AC4,
    nppiAdd_32sc_AC4IRSfs_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f16_c1,
    f16,
    C1,
    nppiAdd_16f_C1R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f16_c1_in_place,
    f16,
    C1,
    nppiAdd_16f_C1IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f16_c3,
    f16,
    C3,
    nppiAdd_16f_C3R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f16_c3_in_place,
    f16,
    C3,
    nppiAdd_16f_C3IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f16_c4,
    f16,
    C4,
    nppiAdd_16f_C4R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f16_c4_in_place,
    f16,
    C4,
    nppiAdd_16f_C4IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f32_c1,
    f32,
    C1,
    nppiAdd_32f_C1R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f32_c1_in_place,
    f32,
    C1,
    nppiAdd_32f_C1IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f32_c3,
    f32,
    C3,
    nppiAdd_32f_C3R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f32_c3_in_place,
    f32,
    C3,
    nppiAdd_32f_C3IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_f32_ac4,
    f32,
    AC4,
    nppiAdd_32f_AC4R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_f32_ac4_in_place,
    f32,
    AC4,
    nppiAdd_32f_AC4IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f32_c4,
    f32,
    C4,
    nppiAdd_32f_C4R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f32_c4_in_place,
    f32,
    C4,
    nppiAdd_32f_C4IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f32_complex_c1,
    Complex32,
    C1,
    nppiAdd_32fc_C1R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f32_complex_c1_in_place,
    Complex32,
    C1,
    nppiAdd_32fc_C1IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f32_complex_c3,
    Complex32,
    C3,
    nppiAdd_32fc_C3R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f32_complex_c3_in_place,
    Complex32,
    C3,
    nppiAdd_32fc_C3IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    add_f32_complex_ac4,
    Complex32,
    AC4,
    nppiAdd_32fc_AC4R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    add_f32_complex_ac4_in_place,
    Complex32,
    AC4,
    nppiAdd_32fc_AC4IR_Ctx
);

impl_binary!(
    "Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    add_f32_complex_c4,
    Complex32,
    C4,
    nppiAdd_32fc_C4R_Ctx
);

impl_binary_in_place!(
    "Adds `source` to `source_destination`. Updates `source_destination` in place.",
    add_f32_complex_c4_in_place,
    Complex32,
    C4,
    nppiAdd_32fc_C4IR_Ctx
);

impl_generic_binary_operation!("Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",

    AddC1,
    add,
    add_c1,
    C1,
    [
        i32 => add_i32_c1_unscaled,
        f16 => add_f16_c1,
        f32 => add_f32_c1,
        Complex32 => add_f32_complex_c1,
    ]
);

impl_generic_binary_operation_in_place!("Adds `source` to `source_destination`. Updates `source_destination` in place.",

    AddC1InPlace,
    add_in_place,
    add_c1_in_place,
    C1,
    [
        f16 => add_f16_c1_in_place,
        f32 => add_f32_c1_in_place,
        Complex32 => add_f32_complex_c1_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledAddC1,
    add_scaled,
    add_scaled_c1,
    C1,
    [
        u8 => add_u8_c1,
        u16 => add_u16_c1,
        i16 => add_i16_c1,
        ComplexI16 => add_i16_complex_c1,
        i32 => add_i32_c1,
        ComplexI32 => add_i32_complex_c1,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledAddC1InPlace,
    add_scaled_in_place,
    add_scaled_c1_in_place,
    C1,
    [
        u8 => add_u8_c1_in_place,
        u16 => add_u16_c1_in_place,
        i16 => add_i16_c1_in_place,
        ComplexI16 => add_i16_complex_c1_in_place,
        i32 => add_i32_c1_in_place,
        ComplexI32 => add_i32_complex_c1_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",

    MultiplyC1,
    multiply,
    multiply_c1,
    C1,
    [
        i32 => multiply_i32_c1_unscaled,
        f16 => multiply_f16_c1,
        f32 => multiply_f32_c1,
        Complex32 => multiply_f32_complex_c1,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source`. Updates `source_destination` in place.",

    MultiplyC1InPlace,
    multiply_in_place,
    multiply_c1_in_place,
    C1,
    [
        f16 => multiply_f16_c1_in_place,
        f32 => multiply_f32_c1_in_place,
        Complex32 => multiply_f32_complex_c1_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledMultiplyC1,
    multiply_scaled,
    multiply_scaled_c1,
    C1,
    [
        u8 => multiply_u8_c1,
        u16 => multiply_u16_c1,
        i16 => multiply_i16_c1,
        ComplexI16 => multiply_i16_complex_c1,
        i32 => multiply_i32_c1,
        ComplexI32 => multiply_i32_complex_c1,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledMultiplyC1InPlace,
    multiply_scaled_in_place,
    multiply_scaled_c1_in_place,
    C1,
    [
        u8 => multiply_u8_c1_in_place,
        u16 => multiply_u16_c1_in_place,
        i16 => multiply_i16_c1_in_place,
        ComplexI16 => multiply_i16_complex_c1_in_place,
        i32 => multiply_i32_c1_in_place,
        ComplexI32 => multiply_i32_complex_c1_in_place,
    ]
);

impl_generic_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",

    SubtractC1,
    subtract,
    subtract_c1,
    C1,
    [
        i32 => subtract_i32_c1_unscaled,
        f16 => subtract_f16_c1,
        f32 => subtract_f32_c1,
        Complex32 => subtract_f32_complex_c1,
    ]
);

impl_generic_binary_operation_in_place!("Subtracts `source` from `source_destination`. Updates `source_destination` in place.",

    SubtractC1InPlace,
    subtract_in_place,
    subtract_c1_in_place,
    C1,
    [
        f16 => subtract_f16_c1_in_place,
        f32 => subtract_f32_c1_in_place,
        Complex32 => subtract_f32_complex_c1_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledSubtractC1,
    subtract_scaled,
    subtract_scaled_c1,
    C1,
    [
        u8 => subtract_u8_c1,
        u16 => subtract_u16_c1,
        i16 => subtract_i16_c1,
        ComplexI16 => subtract_i16_complex_c1,
        i32 => subtract_i32_c1,
        ComplexI32 => subtract_i32_complex_c1,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledSubtractC1InPlace,
    subtract_scaled_in_place,
    subtract_scaled_c1_in_place,
    C1,
    [
        u8 => subtract_u8_c1_in_place,
        u16 => subtract_u16_c1_in_place,
        i16 => subtract_i16_c1_in_place,
        ComplexI16 => subtract_i16_complex_c1_in_place,
        i32 => subtract_i32_c1_in_place,
        ComplexI32 => subtract_i32_complex_c1_in_place,
    ]
);

impl_generic_binary_operation!("Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",

    DivideC1,
    divide,
    divide_c1,
    C1,
    [
        i32 => divide_i32_c1_unscaled,
        f16 => divide_f16_c1,
        f32 => divide_f32_c1,
        Complex32 => divide_f32_complex_c1,
    ]
);

impl_generic_binary_operation_in_place!("Divides `source_destination` by `source`. Updates `source_destination` in place.",

    DivideC1InPlace,
    divide_in_place,
    divide_c1_in_place,
    C1,
    [
        f16 => divide_f16_c1_in_place,
        f32 => divide_f32_c1_in_place,
        Complex32 => divide_f32_complex_c1_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledDivideC1,
    divide_scaled,
    divide_scaled_c1,
    C1,
    [
        u8 => divide_u8_c1,
        u16 => divide_u16_c1,
        i16 => divide_i16_c1,
        ComplexI16 => divide_i16_complex_c1,
        i32 => divide_i32_c1,
        ComplexI32 => divide_i32_complex_c1,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledDivideC1InPlace,
    divide_scaled_in_place,
    divide_scaled_c1_in_place,
    C1,
    [
        u8 => divide_u8_c1_in_place,
        u16 => divide_u16_c1_in_place,
        i16 => divide_i16_c1_in_place,
        ComplexI16 => divide_i16_complex_c1_in_place,
        i32 => divide_i32_c1_in_place,
        ComplexI32 => divide_i32_complex_c1_in_place,
    ]
);

impl_generic_binary_operation!("Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",

    AddC3,
    add,
    add_c3,
    C3,
    [
        f16 => add_f16_c3,
        f32 => add_f32_c3,
        Complex32 => add_f32_complex_c3,
    ]
);

impl_generic_binary_operation_in_place!("Adds `source` to `source_destination`. Updates `source_destination` in place.",

    AddC3InPlace,
    add_in_place,
    add_c3_in_place,
    C3,
    [
        f16 => add_f16_c3_in_place,
        f32 => add_f32_c3_in_place,
        Complex32 => add_f32_complex_c3_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledAddC3,
    add_scaled,
    add_scaled_c3,
    C3,
    [
        u8 => add_u8_c3,
        u16 => add_u16_c3,
        i16 => add_i16_c3,
        ComplexI16 => add_i16_complex_c3,
        i32 => add_i32_c3,
        ComplexI32 => add_i32_complex_c3,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledAddC3InPlace,
    add_scaled_in_place,
    add_scaled_c3_in_place,
    C3,
    [
        u8 => add_u8_c3_in_place,
        u16 => add_u16_c3_in_place,
        i16 => add_i16_c3_in_place,
        ComplexI16 => add_i16_complex_c3_in_place,
        i32 => add_i32_c3_in_place,
        ComplexI32 => add_i32_complex_c3_in_place,
    ]
);

impl_generic_binary_operation!("Adds `source1` and `source2` for each processed channel. Writes the result to `destination`.",

    AddC4,
    add,
    add_c4,
    C4,
    [
        f16 => add_f16_c4,
        f32 => add_f32_c4,
        Complex32 => add_f32_complex_c4,
    ]
);

impl_generic_binary_operation_in_place!("Adds `source` to `source_destination`. Updates `source_destination` in place.",

    AddC4InPlace,
    add_in_place,
    add_c4_in_place,
    C4,
    [
        f16 => add_f16_c4_in_place,
        f32 => add_f32_c4_in_place,
        Complex32 => add_f32_complex_c4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledAddC4,
    add_scaled,
    add_scaled_c4,
    C4,
    [
        u8 => add_u8_c4,
        u16 => add_u16_c4,
        i16 => add_i16_c4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledAddC4InPlace,
    add_scaled_in_place,
    add_scaled_c4_in_place,
    C4,
    [
        u8 => add_u8_c4_in_place,
        u16 => add_u16_c4_in_place,
        i16 => add_i16_c4_in_place,
    ]
);

impl_generic_binary_operation!("Adds `source1` and `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    AddAc4,
    add,
    add_ac4,
    AC4,
    [
        f32 => add_f32_ac4,
        Complex32 => add_f32_complex_ac4,
    ]
);

impl_generic_binary_operation_in_place!("Adds `source` to `source_destination`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    AddAc4InPlace,
    add_in_place,
    add_ac4_in_place,
    AC4,
    [
        f32 => add_f32_ac4_in_place,
        Complex32 => add_f32_complex_ac4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Adds `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    ScaledAddAc4,
    add_scaled,
    add_scaled_ac4,
    AC4,
    [
        u8 => add_u8_ac4,
        u16 => add_u16_ac4,
        i16 => add_i16_ac4,
        ComplexI16 => add_i16_complex_ac4,
        ComplexI32 => add_i32_complex_ac4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Adds `source` to `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    ScaledAddAc4InPlace,
    add_scaled_in_place,
    add_scaled_ac4_in_place,
    AC4,
    [
        u8 => add_u8_ac4_in_place,
        u16 => add_u16_ac4_in_place,
        i16 => add_i16_ac4_in_place,
        ComplexI16 => add_i16_complex_ac4_in_place,
        ComplexI32 => add_i32_complex_ac4_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",

    MultiplyC3,
    multiply,
    multiply_c3,
    C3,
    [
        f16 => multiply_f16_c3,
        f32 => multiply_f32_c3,
        Complex32 => multiply_f32_complex_c3,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source`. Updates `source_destination` in place.",

    MultiplyC3InPlace,
    multiply_in_place,
    multiply_c3_in_place,
    C3,
    [
        f16 => multiply_f16_c3_in_place,
        f32 => multiply_f32_c3_in_place,
        Complex32 => multiply_f32_complex_c3_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledMultiplyC3,
    multiply_scaled,
    multiply_scaled_c3,
    C3,
    [
        u8 => multiply_u8_c3,
        u16 => multiply_u16_c3,
        i16 => multiply_i16_c3,
        ComplexI16 => multiply_i16_complex_c3,
        i32 => multiply_i32_c3,
        ComplexI32 => multiply_i32_complex_c3,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledMultiplyC3InPlace,
    multiply_scaled_in_place,
    multiply_scaled_c3_in_place,
    C3,
    [
        u8 => multiply_u8_c3_in_place,
        u16 => multiply_u16_c3_in_place,
        i16 => multiply_i16_c3_in_place,
        ComplexI16 => multiply_i16_complex_c3_in_place,
        i32 => multiply_i32_c3_in_place,
        ComplexI32 => multiply_i32_complex_c3_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",

    MultiplyC4,
    multiply,
    multiply_c4,
    C4,
    [
        f16 => multiply_f16_c4,
        f32 => multiply_f32_c4,
        Complex32 => multiply_f32_complex_c4,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source`. Updates `source_destination` in place.",

    MultiplyC4InPlace,
    multiply_in_place,
    multiply_c4_in_place,
    C4,
    [
        f16 => multiply_f16_c4_in_place,
        f32 => multiply_f32_c4_in_place,
        Complex32 => multiply_f32_complex_c4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledMultiplyC4,
    multiply_scaled,
    multiply_scaled_c4,
    C4,
    [
        u8 => multiply_u8_c4,
        u16 => multiply_u16_c4,
        i16 => multiply_i16_c4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledMultiplyC4InPlace,
    multiply_scaled_in_place,
    multiply_scaled_c4_in_place,
    C4,
    [
        u8 => multiply_u8_c4_in_place,
        u16 => multiply_u16_c4_in_place,
        i16 => multiply_i16_c4_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    MultiplyAc4,
    multiply,
    multiply_ac4,
    AC4,
    [
        f32 => multiply_f32_ac4,
        Complex32 => multiply_f32_complex_ac4,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    MultiplyAc4InPlace,
    multiply_in_place,
    multiply_ac4_in_place,
    AC4,
    [
        f32 => multiply_f32_ac4_in_place,
        Complex32 => multiply_f32_complex_ac4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    ScaledMultiplyAc4,
    multiply_scaled,
    multiply_scaled_ac4,
    AC4,
    [
        u8 => multiply_u8_ac4,
        u16 => multiply_u16_ac4,
        i16 => multiply_i16_ac4,
        ComplexI16 => multiply_i16_complex_ac4,
        ComplexI32 => multiply_i32_complex_ac4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    ScaledMultiplyAc4InPlace,
    multiply_scaled_in_place,
    multiply_scaled_ac4_in_place,
    AC4,
    [
        u8 => multiply_u8_ac4_in_place,
        u16 => multiply_u16_ac4_in_place,
        i16 => multiply_i16_ac4_in_place,
        ComplexI16 => multiply_i16_complex_ac4_in_place,
        ComplexI32 => multiply_i32_complex_ac4_in_place,
    ]
);

impl_generic_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",

    SubtractC3,
    subtract,
    subtract_c3,
    C3,
    [
        f16 => subtract_f16_c3,
        f32 => subtract_f32_c3,
        Complex32 => subtract_f32_complex_c3,
    ]
);

impl_generic_binary_operation_in_place!("Subtracts `source` from `source_destination`. Updates `source_destination` in place.",

    SubtractC3InPlace,
    subtract_in_place,
    subtract_c3_in_place,
    C3,
    [
        f16 => subtract_f16_c3_in_place,
        f32 => subtract_f32_c3_in_place,
        Complex32 => subtract_f32_complex_c3_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledSubtractC3,
    subtract_scaled,
    subtract_scaled_c3,
    C3,
    [
        u8 => subtract_u8_c3,
        u16 => subtract_u16_c3,
        i16 => subtract_i16_c3,
        ComplexI16 => subtract_i16_complex_c3,
        i32 => subtract_i32_c3,
        ComplexI32 => subtract_i32_complex_c3,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledSubtractC3InPlace,
    subtract_scaled_in_place,
    subtract_scaled_c3_in_place,
    C3,
    [
        u8 => subtract_u8_c3_in_place,
        u16 => subtract_u16_c3_in_place,
        i16 => subtract_i16_c3_in_place,
        ComplexI16 => subtract_i16_complex_c3_in_place,
        i32 => subtract_i32_c3_in_place,
        ComplexI32 => subtract_i32_complex_c3_in_place,
    ]
);

impl_generic_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",

    SubtractC4,
    subtract,
    subtract_c4,
    C4,
    [
        f16 => subtract_f16_c4,
        f32 => subtract_f32_c4,
        Complex32 => subtract_f32_complex_c4,
    ]
);

impl_generic_binary_operation_in_place!("Subtracts `source` from `source_destination`. Updates `source_destination` in place.",

    SubtractC4InPlace,
    subtract_in_place,
    subtract_c4_in_place,
    C4,
    [
        f16 => subtract_f16_c4_in_place,
        f32 => subtract_f32_c4_in_place,
        Complex32 => subtract_f32_complex_c4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledSubtractC4,
    subtract_scaled,
    subtract_scaled_c4,
    C4,
    [
        u8 => subtract_u8_c4,
        u16 => subtract_u16_c4,
        i16 => subtract_i16_c4,
        i32 => subtract_i32_c4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledSubtractC4InPlace,
    subtract_scaled_in_place,
    subtract_scaled_c4_in_place,
    C4,
    [
        u8 => subtract_u8_c4_in_place,
        u16 => subtract_u16_c4_in_place,
        i16 => subtract_i16_c4_in_place,
        i32 => subtract_i32_c4_in_place,
    ]
);

impl_generic_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    SubtractAc4,
    subtract,
    subtract_ac4,
    AC4,
    [
        f32 => subtract_f32_ac4,
        Complex32 => subtract_f32_complex_ac4,
    ]
);

impl_generic_binary_operation_in_place!("Subtracts `source` from `source_destination`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    SubtractAc4InPlace,
    subtract_in_place,
    subtract_ac4_in_place,
    AC4,
    [
        f32 => subtract_f32_ac4_in_place,
        Complex32 => subtract_f32_complex_ac4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    ScaledSubtractAc4,
    subtract_scaled,
    subtract_scaled_ac4,
    AC4,
    [
        u8 => subtract_u8_ac4,
        u16 => subtract_u16_ac4,
        i16 => subtract_i16_ac4,
        ComplexI16 => subtract_i16_complex_ac4,
        ComplexI32 => subtract_i32_complex_ac4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    ScaledSubtractAc4InPlace,
    subtract_scaled_in_place,
    subtract_scaled_ac4_in_place,
    AC4,
    [
        u8 => subtract_u8_ac4_in_place,
        u16 => subtract_u16_ac4_in_place,
        i16 => subtract_i16_ac4_in_place,
        ComplexI16 => subtract_i16_complex_ac4_in_place,
        ComplexI32 => subtract_i32_complex_ac4_in_place,
    ]
);

impl_generic_binary_operation!("Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",

    DivideC3,
    divide,
    divide_c3,
    C3,
    [
        f16 => divide_f16_c3,
        f32 => divide_f32_c3,
        Complex32 => divide_f32_complex_c3,
    ]
);

impl_generic_binary_operation_in_place!("Divides `source_destination` by `source`. Updates `source_destination` in place.",

    DivideC3InPlace,
    divide_in_place,
    divide_c3_in_place,
    C3,
    [
        f16 => divide_f16_c3_in_place,
        f32 => divide_f32_c3_in_place,
        Complex32 => divide_f32_complex_c3_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledDivideC3,
    divide_scaled,
    divide_scaled_c3,
    C3,
    [
        u8 => divide_u8_c3,
        u16 => divide_u16_c3,
        i16 => divide_i16_c3,
        ComplexI16 => divide_i16_complex_c3,
        i32 => divide_i32_c3,
        ComplexI32 => divide_i32_complex_c3,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledDivideC3InPlace,
    divide_scaled_in_place,
    divide_scaled_c3_in_place,
    C3,
    [
        u8 => divide_u8_c3_in_place,
        u16 => divide_u16_c3_in_place,
        i16 => divide_i16_c3_in_place,
        ComplexI16 => divide_i16_complex_c3_in_place,
        i32 => divide_i32_c3_in_place,
        ComplexI32 => divide_i32_complex_c3_in_place,
    ]
);

impl_generic_binary_operation!("Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",

    DivideC4,
    divide,
    divide_c4,
    C4,
    [
        f16 => divide_f16_c4,
        f32 => divide_f32_c4,
        Complex32 => divide_f32_complex_c4,
    ]
);

impl_generic_binary_operation_in_place!("Divides `source_destination` by `source`. Updates `source_destination` in place.",

    DivideC4InPlace,
    divide_in_place,
    divide_c4_in_place,
    C4,
    [
        f16 => divide_f16_c4_in_place,
        f32 => divide_f32_c4_in_place,
        Complex32 => divide_f32_complex_c4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    ScaledDivideC4,
    divide_scaled,
    divide_scaled_c4,
    C4,
    [
        u8 => divide_u8_c4,
        u16 => divide_u16_c4,
        i16 => divide_i16_c4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    ScaledDivideC4InPlace,
    divide_scaled_in_place,
    divide_scaled_c4_in_place,
    C4,
    [
        u8 => divide_u8_c4_in_place,
        u16 => divide_u16_c4_in_place,
        i16 => divide_i16_c4_in_place,
    ]
);

impl_generic_binary_operation!("Divides `source2` by `source1` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    DivideAc4,
    divide,
    divide_ac4,
    AC4,
    [
        f32 => divide_f32_ac4,
        Complex32 => divide_f32_complex_ac4,
    ]
);

impl_generic_binary_operation_in_place!("Divides `source_destination` by `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    DivideAc4InPlace,
    divide_in_place,
    divide_ac4_in_place,
    AC4,
    [
        f32 => divide_f32_ac4_in_place,
        Complex32 => divide_f32_complex_ac4_in_place,
    ]
);

impl_generic_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    ScaledDivideAc4,
    divide_scaled,
    divide_scaled_ac4,
    AC4,
    [
        u8 => divide_u8_ac4,
        u16 => divide_u16_ac4,
        i16 => divide_i16_ac4,
        ComplexI16 => divide_i16_complex_ac4,
        ComplexI32 => divide_i32_complex_ac4,
    ]
);

impl_generic_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    ScaledDivideAc4InPlace,
    divide_scaled_in_place,
    divide_scaled_ac4_in_place,
    AC4,
    [
        u8 => divide_u8_ac4_in_place,
        u16 => divide_u16_ac4_in_place,
        i16 => divide_i16_ac4_in_place,
        ComplexI16 => divide_i16_complex_ac4_in_place,
        ComplexI32 => divide_i32_complex_ac4_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",

    MultiplyScaleC1,
    multiply_scale,
    multiply_scale_c1,
    C1,
    [
        u8 => multiply_scale_u8_c1,
        u16 => multiply_scale_u16_c1,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",

    MultiplyScaleC1InPlace,
    multiply_scale_in_place,
    multiply_scale_c1_in_place,
    C1,
    [
        u8 => multiply_scale_u8_c1_in_place,
        u16 => multiply_scale_u16_c1_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",

    MultiplyScaleC3,
    multiply_scale,
    multiply_scale_c3,
    C3,
    [
        u8 => multiply_scale_u8_c3,
        u16 => multiply_scale_u16_c3,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",

    MultiplyScaleC3InPlace,
    multiply_scale_in_place,
    multiply_scale_c3_in_place,
    C3,
    [
        u8 => multiply_scale_u8_c3_in_place,
        u16 => multiply_scale_u16_c3_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",

    MultiplyScaleC4,
    multiply_scale,
    multiply_scale_c4,
    C4,
    [
        u8 => multiply_scale_u8_c4,
        u16 => multiply_scale_u16_c4,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",

    MultiplyScaleC4InPlace,
    multiply_scale_in_place,
    multiply_scale_c4_in_place,
    C4,
    [
        u8 => multiply_scale_u8_c4_in_place,
        u16 => multiply_scale_u16_c4_in_place,
    ]
);

impl_generic_binary_operation!("Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    MultiplyScaleAc4,
    multiply_scale,
    multiply_scale_ac4,
    AC4,
    [
        u8 => multiply_scale_u8_ac4,
        u16 => multiply_scale_u16_ac4,
    ]
);

impl_generic_binary_operation_in_place!("Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    MultiplyScaleAc4InPlace,
    multiply_scale_in_place,
    multiply_scale_ac4_in_place,
    AC4,
    [
        u8 => multiply_scale_u8_ac4_in_place,
        u16 => multiply_scale_u16_ac4_in_place,
    ]
);

impl_generic_round_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    DivideRoundC1,
    divide_round,
    divide_round_c1,
    C1,
    [
        u8 => divide_round_u8_c1,
        u16 => divide_round_u16_c1,
        i16 => divide_round_i16_c1,
    ]
);

impl_generic_round_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    DivideRoundC1InPlace,
    divide_round_in_place,
    divide_round_c1_in_place,
    C1,
    [
        u8 => divide_round_u8_c1_in_place,
        u16 => divide_round_u16_c1_in_place,
        i16 => divide_round_i16_c1_in_place,
    ]
);

impl_generic_round_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    DivideRoundC3,
    divide_round,
    divide_round_c3,
    C3,
    [
        u8 => divide_round_u8_c3,
        u16 => divide_round_u16_c3,
        i16 => divide_round_i16_c3,
    ]
);

impl_generic_round_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    DivideRoundC3InPlace,
    divide_round_in_place,
    divide_round_c3_in_place,
    C3,
    [
        u8 => divide_round_u8_c3_in_place,
        u16 => divide_round_u16_c3_in_place,
        i16 => divide_round_i16_c3_in_place,
    ]
);

impl_generic_round_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

    DivideRoundC4,
    divide_round,
    divide_round_c4,
    C4,
    [
        u8 => divide_round_u8_c4,
        u16 => divide_round_u16_c4,
        i16 => divide_round_i16_c4,
    ]
);

impl_generic_round_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

    DivideRoundC4InPlace,
    divide_round_in_place,
    divide_round_c4_in_place,
    C4,
    [
        u8 => divide_round_u8_c4_in_place,
        u16 => divide_round_u16_c4_in_place,
        i16 => divide_round_i16_c4_in_place,
    ]
);

impl_generic_round_scaled_binary_operation!("Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    DivideRoundAc4,
    divide_round,
    divide_round_ac4,
    AC4,
    [
        u8 => divide_round_u8_ac4,
        u16 => divide_round_u16_ac4,
        i16 => divide_round_i16_ac4,
    ]
);

impl_generic_round_scaled_binary_operation_in_place!("Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    DivideRoundAc4InPlace,
    divide_round_in_place,
    divide_round_ac4_in_place,
    AC4,
    [
        u8 => divide_round_u8_ac4_in_place,
        u16 => divide_round_u16_ac4_in_place,
        i16 => divide_round_i16_ac4_in_place,
    ]
);
