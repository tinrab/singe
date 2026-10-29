use singe_cuda::types::{Complex32, f16};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::validate_same_size,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{ComplexI16, ComplexI32, RoundMode},
};

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_u8_c1,
    u8,
    C1,
    nppiMul_8u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_u8_c1_in_place,
    u8,
    C1,
    nppiMul_8u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_u8_c3,
    u8,
    C3,
    nppiMul_8u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_u8_c3_in_place,
    u8,
    C3,
    nppiMul_8u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_u8_ac4,
    u8,
    AC4,
    nppiMul_8u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_u8_ac4_in_place,
    u8,
    AC4,
    nppiMul_8u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_u8_c4,
    u8,
    C4,
    nppiMul_8u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_u8_c4_in_place,
    u8,
    C4,
    nppiMul_8u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_u16_c1,
    u16,
    C1,
    nppiMul_16u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_u16_c1_in_place,
    u16,
    C1,
    nppiMul_16u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_u16_c3,
    u16,
    C3,
    nppiMul_16u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_u16_c3_in_place,
    u16,
    C3,
    nppiMul_16u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_u16_ac4,
    u16,
    AC4,
    nppiMul_16u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_u16_ac4_in_place,
    u16,
    AC4,
    nppiMul_16u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_u16_c4,
    u16,
    C4,
    nppiMul_16u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_u16_c4_in_place,
    u16,
    C4,
    nppiMul_16u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i16_c1,
    i16,
    C1,
    nppiMul_16s_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i16_c1_in_place,
    i16,
    C1,
    nppiMul_16s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i16_c3,
    i16,
    C3,
    nppiMul_16s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i16_c3_in_place,
    i16,
    C3,
    nppiMul_16s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_i16_ac4,
    i16,
    AC4,
    nppiMul_16s_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_i16_ac4_in_place,
    i16,
    AC4,
    nppiMul_16s_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i16_c4,
    i16,
    C4,
    nppiMul_16s_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i16_c4_in_place,
    i16,
    C4,
    nppiMul_16s_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i16_complex_c1,
    ComplexI16,
    C1,
    nppiMul_16sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i16_complex_c1_in_place,
    ComplexI16,
    C1,
    nppiMul_16sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i16_complex_c3,
    ComplexI16,
    C3,
    nppiMul_16sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i16_complex_c3_in_place,
    ComplexI16,
    C3,
    nppiMul_16sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_i16_complex_ac4,
    ComplexI16,
    AC4,
    nppiMul_16sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_i16_complex_ac4_in_place,
    ComplexI16,
    AC4,
    nppiMul_16sc_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i32_c1,
    i32,
    C1,
    nppiMul_32s_C1RSfs_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_i32_c1_unscaled,
    i32,
    C1,
    nppiMul_32s_C1R_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i32_c1_in_place,
    i32,
    C1,
    nppiMul_32s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i32_c3,
    i32,
    C3,
    nppiMul_32s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i32_c3_in_place,
    i32,
    C3,
    nppiMul_32s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i32_complex_c1,
    ComplexI32,
    C1,
    nppiMul_32sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i32_complex_c1_in_place,
    ComplexI32,
    C1,
    nppiMul_32sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_i32_complex_c3,
    ComplexI32,
    C3,
    nppiMul_32sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_i32_complex_c3_in_place,
    ComplexI32,
    C3,
    nppiMul_32sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Multiplies `source1` and `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_i32_complex_ac4,
    ComplexI32,
    AC4,
    nppiMul_32sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Multiplies `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_i32_complex_ac4_in_place,
    ComplexI32,
    AC4,
    nppiMul_32sc_AC4IRSfs_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f16_c1,
    f16,
    C1,
    nppiMul_16f_C1R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f16_c1_in_place,
    f16,
    C1,
    nppiMul_16f_C1IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f16_c3,
    f16,
    C3,
    nppiMul_16f_C3R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f16_c3_in_place,
    f16,
    C3,
    nppiMul_16f_C3IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f16_c4,
    f16,
    C4,
    nppiMul_16f_C4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f16_c4_in_place,
    f16,
    C4,
    nppiMul_16f_C4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f32_c1,
    f32,
    C1,
    nppiMul_32f_C1R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f32_c1_in_place,
    f32,
    C1,
    nppiMul_32f_C1IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f32_c3,
    f32,
    C3,
    nppiMul_32f_C3R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f32_c3_in_place,
    f32,
    C3,
    nppiMul_32f_C3IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_f32_ac4,
    f32,
    AC4,
    nppiMul_32f_AC4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_f32_ac4_in_place,
    f32,
    AC4,
    nppiMul_32f_AC4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f32_c4,
    f32,
    C4,
    nppiMul_32f_C4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f32_c4_in_place,
    f32,
    C4,
    nppiMul_32f_C4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f32_complex_c1,
    Complex32,
    C1,
    nppiMul_32fc_C1R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f32_complex_c1_in_place,
    Complex32,
    C1,
    nppiMul_32fc_C1IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f32_complex_c3,
    Complex32,
    C3,
    nppiMul_32fc_C3R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f32_complex_c3_in_place,
    Complex32,
    C3,
    nppiMul_32fc_C3IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_f32_complex_ac4,
    Complex32,
    AC4,
    nppiMul_32fc_AC4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_f32_complex_ac4_in_place,
    Complex32,
    AC4,
    nppiMul_32fc_AC4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` for each processed channel. Writes the result to `destination`.",
    multiply_f32_complex_c4,
    Complex32,
    C4,
    nppiMul_32fc_C4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source`. Updates `source_destination` in place.",
    multiply_f32_complex_c4_in_place,
    Complex32,
    C4,
    nppiMul_32fc_C4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",
    multiply_scale_u8_c1,
    u8,
    C1,
    nppiMulScale_8u_C1R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",
    multiply_scale_u8_c1_in_place,
    u8,
    C1,
    nppiMulScale_8u_C1IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",
    multiply_scale_u8_c3,
    u8,
    C3,
    nppiMulScale_8u_C3R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",
    multiply_scale_u8_c3_in_place,
    u8,
    C3,
    nppiMulScale_8u_C3IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_scale_u8_ac4,
    u8,
    AC4,
    nppiMulScale_8u_AC4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_scale_u8_ac4_in_place,
    u8,
    AC4,
    nppiMulScale_8u_AC4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",
    multiply_scale_u8_c4,
    u8,
    C4,
    nppiMulScale_8u_C4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",
    multiply_scale_u8_c4_in_place,
    u8,
    C4,
    nppiMulScale_8u_C4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",
    multiply_scale_u16_c1,
    u16,
    C1,
    nppiMulScale_16u_C1R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",
    multiply_scale_u16_c1_in_place,
    u16,
    C1,
    nppiMulScale_16u_C1IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",
    multiply_scale_u16_c3,
    u16,
    C3,
    nppiMulScale_16u_C3R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",
    multiply_scale_u16_c3_in_place,
    u16,
    C3,
    nppiMulScale_16u_C3IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_scale_u16_ac4,
    u16,
    AC4,
    nppiMulScale_16u_AC4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_scale_u16_ac4_in_place,
    u16,
    AC4,
    nppiMulScale_16u_AC4IR_Ctx
);

impl_binary!(
    "Multiplies `source1` and `source2` with NPP's fixed bit-width scaling. Writes the result to `destination`.",
    multiply_scale_u16_c4,
    u16,
    C4,
    nppiMulScale_16u_C4R_Ctx
);

impl_binary_in_place!(
    "Multiplies `source_destination` by `source` with NPP's fixed bit-width scaling. Updates `source_destination` in place.",
    multiply_scale_u16_c4_in_place,
    u16,
    C4,
    nppiMulScale_16u_C4IR_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_u8_c1,
    u8,
    C1,
    nppiSub_8u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_u8_c1_in_place,
    u8,
    C1,
    nppiSub_8u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_u8_c3,
    u8,
    C3,
    nppiSub_8u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_u8_c3_in_place,
    u8,
    C3,
    nppiSub_8u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_u8_ac4,
    u8,
    AC4,
    nppiSub_8u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_u8_ac4_in_place,
    u8,
    AC4,
    nppiSub_8u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_u8_c4,
    u8,
    C4,
    nppiSub_8u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_u8_c4_in_place,
    u8,
    C4,
    nppiSub_8u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_u16_c1,
    u16,
    C1,
    nppiSub_16u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_u16_c1_in_place,
    u16,
    C1,
    nppiSub_16u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_u16_c3,
    u16,
    C3,
    nppiSub_16u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_u16_c3_in_place,
    u16,
    C3,
    nppiSub_16u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_u16_ac4,
    u16,
    AC4,
    nppiSub_16u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_u16_ac4_in_place,
    u16,
    AC4,
    nppiSub_16u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_u16_c4,
    u16,
    C4,
    nppiSub_16u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_u16_c4_in_place,
    u16,
    C4,
    nppiSub_16u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i16_c1,
    i16,
    C1,
    nppiSub_16s_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i16_c1_in_place,
    i16,
    C1,
    nppiSub_16s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i16_c3,
    i16,
    C3,
    nppiSub_16s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i16_c3_in_place,
    i16,
    C3,
    nppiSub_16s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_i16_ac4,
    i16,
    AC4,
    nppiSub_16s_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_i16_ac4_in_place,
    i16,
    AC4,
    nppiSub_16s_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i16_c4,
    i16,
    C4,
    nppiSub_16s_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i16_c4_in_place,
    i16,
    C4,
    nppiSub_16s_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i16_complex_c1,
    ComplexI16,
    C1,
    nppiSub_16sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i16_complex_c1_in_place,
    ComplexI16,
    C1,
    nppiSub_16sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i16_complex_c3,
    ComplexI16,
    C3,
    nppiSub_16sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i16_complex_c3_in_place,
    ComplexI16,
    C3,
    nppiSub_16sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_i16_complex_ac4,
    ComplexI16,
    AC4,
    nppiSub_16sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_i16_complex_ac4_in_place,
    ComplexI16,
    AC4,
    nppiSub_16sc_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i32_c1,
    i32,
    C1,
    nppiSub_32s_C1RSfs_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_i32_c1_unscaled,
    i32,
    C1,
    nppiSub_32s_C1R_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i32_c1_in_place,
    i32,
    C1,
    nppiSub_32s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i32_c3,
    i32,
    C3,
    nppiSub_32s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i32_c3_in_place,
    i32,
    C3,
    nppiSub_32s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i32_c4,
    i32,
    C4,
    nppiSub_32s_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i32_c4_in_place,
    i32,
    C4,
    nppiSub_32s_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i32_complex_c1,
    ComplexI32,
    C1,
    nppiSub_32sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i32_complex_c1_in_place,
    ComplexI32,
    C1,
    nppiSub_32sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    subtract_i32_complex_c3,
    ComplexI32,
    C3,
    nppiSub_32sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    subtract_i32_complex_c3_in_place,
    ComplexI32,
    C3,
    nppiSub_32sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Subtracts `source1` from `source2` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_i32_complex_ac4,
    ComplexI32,
    AC4,
    nppiSub_32sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Subtracts `source` from `source_destination`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_i32_complex_ac4_in_place,
    ComplexI32,
    AC4,
    nppiSub_32sc_AC4IRSfs_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f16_c1,
    f16,
    C1,
    nppiSub_16f_C1R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f16_c1_in_place,
    f16,
    C1,
    nppiSub_16f_C1IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f16_c3,
    f16,
    C3,
    nppiSub_16f_C3R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f16_c3_in_place,
    f16,
    C3,
    nppiSub_16f_C3IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f16_c4,
    f16,
    C4,
    nppiSub_16f_C4R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f16_c4_in_place,
    f16,
    C4,
    nppiSub_16f_C4IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f32_c1,
    f32,
    C1,
    nppiSub_32f_C1R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f32_c1_in_place,
    f32,
    C1,
    nppiSub_32f_C1IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f32_c3,
    f32,
    C3,
    nppiSub_32f_C3R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f32_c3_in_place,
    f32,
    C3,
    nppiSub_32f_C3IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_f32_ac4,
    f32,
    AC4,
    nppiSub_32f_AC4R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_f32_ac4_in_place,
    f32,
    AC4,
    nppiSub_32f_AC4IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f32_c4,
    f32,
    C4,
    nppiSub_32f_C4R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f32_c4_in_place,
    f32,
    C4,
    nppiSub_32f_C4IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f32_complex_c1,
    Complex32,
    C1,
    nppiSub_32fc_C1R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f32_complex_c1_in_place,
    Complex32,
    C1,
    nppiSub_32fc_C1IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f32_complex_c3,
    Complex32,
    C3,
    nppiSub_32fc_C3R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f32_complex_c3_in_place,
    Complex32,
    C3,
    nppiSub_32fc_C3IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    subtract_f32_complex_ac4,
    Complex32,
    AC4,
    nppiSub_32fc_AC4R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    subtract_f32_complex_ac4_in_place,
    Complex32,
    AC4,
    nppiSub_32fc_AC4IR_Ctx
);

impl_binary!(
    "Subtracts `source1` from `source2` for each processed channel. Writes the result to `destination`.",
    subtract_f32_complex_c4,
    Complex32,
    C4,
    nppiSub_32fc_C4R_Ctx
);

impl_binary_in_place!(
    "Subtracts `source` from `source_destination`. Updates `source_destination` in place.",
    subtract_f32_complex_c4_in_place,
    Complex32,
    C4,
    nppiSub_32fc_C4IR_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_u8_c1,
    u8,
    C1,
    nppiDiv_8u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_u8_c1_in_place,
    u8,
    C1,
    nppiDiv_8u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_u8_c3,
    u8,
    C3,
    nppiDiv_8u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_u8_c3_in_place,
    u8,
    C3,
    nppiDiv_8u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_u8_ac4,
    u8,
    AC4,
    nppiDiv_8u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_u8_ac4_in_place,
    u8,
    AC4,
    nppiDiv_8u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_u8_c4,
    u8,
    C4,
    nppiDiv_8u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_u8_c4_in_place,
    u8,
    C4,
    nppiDiv_8u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_u16_c1,
    u16,
    C1,
    nppiDiv_16u_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_u16_c1_in_place,
    u16,
    C1,
    nppiDiv_16u_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_u16_c3,
    u16,
    C3,
    nppiDiv_16u_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_u16_c3_in_place,
    u16,
    C3,
    nppiDiv_16u_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_u16_ac4,
    u16,
    AC4,
    nppiDiv_16u_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_u16_ac4_in_place,
    u16,
    AC4,
    nppiDiv_16u_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_u16_c4,
    u16,
    C4,
    nppiDiv_16u_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_u16_c4_in_place,
    u16,
    C4,
    nppiDiv_16u_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i16_c1,
    i16,
    C1,
    nppiDiv_16s_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i16_c1_in_place,
    i16,
    C1,
    nppiDiv_16s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i16_c3,
    i16,
    C3,
    nppiDiv_16s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i16_c3_in_place,
    i16,
    C3,
    nppiDiv_16s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_i16_ac4,
    i16,
    AC4,
    nppiDiv_16s_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_i16_ac4_in_place,
    i16,
    AC4,
    nppiDiv_16s_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i16_c4,
    i16,
    C4,
    nppiDiv_16s_C4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i16_c4_in_place,
    i16,
    C4,
    nppiDiv_16s_C4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i16_complex_c1,
    ComplexI16,
    C1,
    nppiDiv_16sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i16_complex_c1_in_place,
    ComplexI16,
    C1,
    nppiDiv_16sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i16_complex_c3,
    ComplexI16,
    C3,
    nppiDiv_16sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i16_complex_c3_in_place,
    ComplexI16,
    C3,
    nppiDiv_16sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_i16_complex_ac4,
    ComplexI16,
    AC4,
    nppiDiv_16sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_i16_complex_ac4_in_place,
    ComplexI16,
    AC4,
    nppiDiv_16sc_AC4IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i32_c1,
    i32,
    C1,
    nppiDiv_32s_C1RSfs_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_i32_c1_unscaled,
    i32,
    C1,
    nppiDiv_32s_C1R_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i32_c1_in_place,
    i32,
    C1,
    nppiDiv_32s_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i32_c3,
    i32,
    C3,
    nppiDiv_32s_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i32_c3_in_place,
    i32,
    C3,
    nppiDiv_32s_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i32_complex_c1,
    ComplexI32,
    C1,
    nppiDiv_32sc_C1RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i32_complex_c1_in_place,
    ComplexI32,
    C1,
    nppiDiv_32sc_C1IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_i32_complex_c3,
    ComplexI32,
    C3,
    nppiDiv_32sc_C3RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_i32_complex_c3_in_place,
    ComplexI32,
    C3,
    nppiDiv_32sc_C3IRSfs_Ctx
);

impl_binary_scaled!(
    "Divides `source2` by `source1` for each processed channel. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_i32_complex_ac4,
    ComplexI32,
    AC4,
    nppiDiv_32sc_AC4RSfs_Ctx
);

impl_binary_scaled_in_place!(
    "Divides `source_destination` by `source`. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_i32_complex_ac4_in_place,
    ComplexI32,
    AC4,
    nppiDiv_32sc_AC4IRSfs_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f16_c1,
    f16,
    C1,
    nppiDiv_16f_C1R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f16_c1_in_place,
    f16,
    C1,
    nppiDiv_16f_C1IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f16_c3,
    f16,
    C3,
    nppiDiv_16f_C3R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f16_c3_in_place,
    f16,
    C3,
    nppiDiv_16f_C3IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f16_c4,
    f16,
    C4,
    nppiDiv_16f_C4R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f16_c4_in_place,
    f16,
    C4,
    nppiDiv_16f_C4IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f32_c1,
    f32,
    C1,
    nppiDiv_32f_C1R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f32_c1_in_place,
    f32,
    C1,
    nppiDiv_32f_C1IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f32_c3,
    f32,
    C3,
    nppiDiv_32f_C3R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f32_c3_in_place,
    f32,
    C3,
    nppiDiv_32f_C3IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_f32_ac4,
    f32,
    AC4,
    nppiDiv_32f_AC4R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_f32_ac4_in_place,
    f32,
    AC4,
    nppiDiv_32f_AC4IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f32_c4,
    f32,
    C4,
    nppiDiv_32f_C4R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f32_c4_in_place,
    f32,
    C4,
    nppiDiv_32f_C4IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f32_complex_c1,
    Complex32,
    C1,
    nppiDiv_32fc_C1R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f32_complex_c1_in_place,
    Complex32,
    C1,
    nppiDiv_32fc_C1IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f32_complex_c3,
    Complex32,
    C3,
    nppiDiv_32fc_C3R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f32_complex_c3_in_place,
    Complex32,
    C3,
    nppiDiv_32fc_C3IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_f32_complex_ac4,
    Complex32,
    AC4,
    nppiDiv_32fc_AC4R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_f32_complex_ac4_in_place,
    Complex32,
    AC4,
    nppiDiv_32fc_AC4IR_Ctx
);

impl_binary!(
    "Divides `source2` by `source1` for each processed channel. Writes the result to `destination`.",
    divide_f32_complex_c4,
    Complex32,
    C4,
    nppiDiv_32fc_C4R_Ctx
);

impl_binary_in_place!(
    "Divides `source_destination` by `source`. Updates `source_destination` in place.",
    divide_f32_complex_c4_in_place,
    Complex32,
    C4,
    nppiDiv_32fc_C4IR_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_u8_c1,
    u8,
    C1,
    nppiDiv_Round_8u_C1RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_u8_c1_in_place,
    u8,
    C1,
    nppiDiv_Round_8u_C1IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_u8_c3,
    u8,
    C3,
    nppiDiv_Round_8u_C3RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_u8_c3_in_place,
    u8,
    C3,
    nppiDiv_Round_8u_C3IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_round_u8_ac4,
    u8,
    AC4,
    nppiDiv_Round_8u_AC4RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_round_u8_ac4_in_place,
    u8,
    AC4,
    nppiDiv_Round_8u_AC4IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_u8_c4,
    u8,
    C4,
    nppiDiv_Round_8u_C4RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_u8_c4_in_place,
    u8,
    C4,
    nppiDiv_Round_8u_C4IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_u16_c1,
    u16,
    C1,
    nppiDiv_Round_16u_C1RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_u16_c1_in_place,
    u16,
    C1,
    nppiDiv_Round_16u_C1IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_u16_c3,
    u16,
    C3,
    nppiDiv_Round_16u_C3RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_u16_c3_in_place,
    u16,
    C3,
    nppiDiv_Round_16u_C3IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_round_u16_ac4,
    u16,
    AC4,
    nppiDiv_Round_16u_AC4RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_round_u16_ac4_in_place,
    u16,
    AC4,
    nppiDiv_Round_16u_AC4IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_u16_c4,
    u16,
    C4,
    nppiDiv_Round_16u_C4RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_u16_c4_in_place,
    u16,
    C4,
    nppiDiv_Round_16u_C4IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_i16_c1,
    i16,
    C1,
    nppiDiv_Round_16s_C1RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_i16_c1_in_place,
    i16,
    C1,
    nppiDiv_Round_16s_C1IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_i16_c3,
    i16,
    C3,
    nppiDiv_Round_16s_C3RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_i16_c3_in_place,
    i16,
    C3,
    nppiDiv_Round_16s_C3IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    divide_round_i16_ac4,
    i16,
    AC4,
    nppiDiv_Round_16s_AC4RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    divide_round_i16_ac4_in_place,
    i16,
    AC4,
    nppiDiv_Round_16s_AC4IRSfs_Ctx
);

impl_binary_round_scaled!(
    "Divides `source2` by `source1` for each processed channel. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    divide_round_i16_c4,
    i16,
    C4,
    nppiDiv_Round_16s_C4RSfs_Ctx
);

impl_binary_round_scaled_in_place!(
    "Divides `source_destination` by `source`. Uses `round_mode` for native result rounding. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    divide_round_i16_c4_in_place,
    i16,
    C4,
    nppiDiv_Round_16s_C4IRSfs_Ctx
);
