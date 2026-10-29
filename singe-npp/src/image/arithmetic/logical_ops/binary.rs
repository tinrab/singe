use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::validate_same_size,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
};

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_u8_c1,
    u8,
    C1,
    nppiAnd_8u_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_u8_c1_in_place,
    u8,
    C1,
    nppiAnd_8u_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_u8_c3,
    u8,
    C3,
    nppiAnd_8u_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_u8_c3_in_place,
    u8,
    C3,
    nppiAnd_8u_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_and_u8_ac4,
    u8,
    AC4,
    nppiAnd_8u_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_and_u8_ac4_in_place,
    u8,
    AC4,
    nppiAnd_8u_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_u8_c4,
    u8,
    C4,
    nppiAnd_8u_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_u8_c4_in_place,
    u8,
    C4,
    nppiAnd_8u_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_u16_c1,
    u16,
    C1,
    nppiAnd_16u_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_u16_c1_in_place,
    u16,
    C1,
    nppiAnd_16u_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_u16_c3,
    u16,
    C3,
    nppiAnd_16u_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_u16_c3_in_place,
    u16,
    C3,
    nppiAnd_16u_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_and_u16_ac4,
    u16,
    AC4,
    nppiAnd_16u_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_and_u16_ac4_in_place,
    u16,
    AC4,
    nppiAnd_16u_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_u16_c4,
    u16,
    C4,
    nppiAnd_16u_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_u16_c4_in_place,
    u16,
    C4,
    nppiAnd_16u_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_i32_c1,
    i32,
    C1,
    nppiAnd_32s_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_i32_c1_in_place,
    i32,
    C1,
    nppiAnd_32s_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_i32_c3,
    i32,
    C3,
    nppiAnd_32s_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_i32_c3_in_place,
    i32,
    C3,
    nppiAnd_32s_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_and_i32_ac4,
    i32,
    AC4,
    nppiAnd_32s_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_and_i32_ac4_in_place,
    i32,
    AC4,
    nppiAnd_32s_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise AND to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_and_i32_c4,
    i32,
    C4,
    nppiAnd_32s_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise AND to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_and_i32_c4_in_place,
    i32,
    C4,
    nppiAnd_32s_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_u8_c1,
    u8,
    C1,
    nppiOr_8u_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_u8_c1_in_place,
    u8,
    C1,
    nppiOr_8u_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_u8_c3,
    u8,
    C3,
    nppiOr_8u_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_u8_c3_in_place,
    u8,
    C3,
    nppiOr_8u_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_or_u8_ac4,
    u8,
    AC4,
    nppiOr_8u_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_or_u8_ac4_in_place,
    u8,
    AC4,
    nppiOr_8u_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_u8_c4,
    u8,
    C4,
    nppiOr_8u_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_u8_c4_in_place,
    u8,
    C4,
    nppiOr_8u_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_u16_c1,
    u16,
    C1,
    nppiOr_16u_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_u16_c1_in_place,
    u16,
    C1,
    nppiOr_16u_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_u16_c3,
    u16,
    C3,
    nppiOr_16u_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_u16_c3_in_place,
    u16,
    C3,
    nppiOr_16u_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_or_u16_ac4,
    u16,
    AC4,
    nppiOr_16u_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_or_u16_ac4_in_place,
    u16,
    AC4,
    nppiOr_16u_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_u16_c4,
    u16,
    C4,
    nppiOr_16u_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_u16_c4_in_place,
    u16,
    C4,
    nppiOr_16u_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_i32_c1,
    i32,
    C1,
    nppiOr_32s_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_i32_c1_in_place,
    i32,
    C1,
    nppiOr_32s_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_i32_c3,
    i32,
    C3,
    nppiOr_32s_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_i32_c3_in_place,
    i32,
    C3,
    nppiOr_32s_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_or_i32_ac4,
    i32,
    AC4,
    nppiOr_32s_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_or_i32_ac4_in_place,
    i32,
    AC4,
    nppiOr_32s_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_or_i32_c4,
    i32,
    C4,
    nppiOr_32s_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_or_i32_c4_in_place,
    i32,
    C4,
    nppiOr_32s_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_u8_c1,
    u8,
    C1,
    nppiXor_8u_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_u8_c1_in_place,
    u8,
    C1,
    nppiXor_8u_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_u8_c3,
    u8,
    C3,
    nppiXor_8u_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_u8_c3_in_place,
    u8,
    C3,
    nppiXor_8u_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_xor_u8_ac4,
    u8,
    AC4,
    nppiXor_8u_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_xor_u8_ac4_in_place,
    u8,
    AC4,
    nppiXor_8u_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_u8_c4,
    u8,
    C4,
    nppiXor_8u_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_u8_c4_in_place,
    u8,
    C4,
    nppiXor_8u_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_u16_c1,
    u16,
    C1,
    nppiXor_16u_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_u16_c1_in_place,
    u16,
    C1,
    nppiXor_16u_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_u16_c3,
    u16,
    C3,
    nppiXor_16u_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_u16_c3_in_place,
    u16,
    C3,
    nppiXor_16u_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_xor_u16_ac4,
    u16,
    AC4,
    nppiXor_16u_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_xor_u16_ac4_in_place,
    u16,
    AC4,
    nppiXor_16u_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_u16_c4,
    u16,
    C4,
    nppiXor_16u_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_u16_c4_in_place,
    u16,
    C4,
    nppiXor_16u_C4IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_i32_c1,
    i32,
    C1,
    nppiXor_32s_C1R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_i32_c1_in_place,
    i32,
    C1,
    nppiXor_32s_C1IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_i32_c3,
    i32,
    C3,
    nppiXor_32s_C3R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_i32_c3_in_place,
    i32,
    C3,
    nppiXor_32s_C3IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_xor_i32_ac4,
    i32,
    AC4,
    nppiXor_32s_AC4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_xor_i32_ac4_in_place,
    i32,
    AC4,
    nppiXor_32s_AC4IR_Ctx
);

impl_binary!(
    "Applies bitwise exclusive OR to corresponding channels of `source1` and `source2`. Writes the result to `destination`.",
    logical_xor_i32_c4,
    i32,
    C4,
    nppiXor_32s_C4R_Ctx
);

impl_binary_in_place!(
    "Applies bitwise exclusive OR to `source_destination` and `source`. Updates `source_destination` in place.",
    logical_xor_i32_c4_in_place,
    i32,
    C4,
    nppiXor_32s_C4IR_Ctx
);
