use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::validate_same_size,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::IntoNpp,
};

impl_add_constant_scalar!(
    "Shifts each processed channel right by its constant bit count. Writes the result to `destination`.",
    right_shift_constant_u8_c1,
    u8,
    u32,
    C1,
    nppiRShiftC_8u_C1R_Ctx
);

impl_add_constant_scalar_in_place!(
    "Shifts each processed channel right by its constant bit count. Updates `source_destination` in place.",
    right_shift_constant_u8_c1_in_place,
    u8,
    u32,
    C1,
    nppiRShiftC_8u_C1IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_u8_c3,
    u8,
    u32,
    C3,
    3,
    nppiRShiftC_8u_C3R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_u8_c3_in_place,
    u8,
    u32,
    C3,
    3,
    nppiRShiftC_8u_C3IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    right_shift_constant_u8_ac4,
    u8,
    u32,
    AC4,
    3,
    nppiRShiftC_8u_AC4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    right_shift_constant_u8_ac4_in_place,
    u8,
    u32,
    AC4,
    3,
    nppiRShiftC_8u_AC4IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_u8_c4,
    u8,
    u32,
    C4,
    4,
    nppiRShiftC_8u_C4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_u8_c4_in_place,
    u8,
    u32,
    C4,
    4,
    nppiRShiftC_8u_C4IR_Ctx
);

impl_add_constant_scalar!(
    "Shifts each processed channel right by its constant bit count. Writes the result to `destination`.",
    right_shift_constant_i8_c1,
    i8,
    u32,
    C1,
    nppiRShiftC_8s_C1R_Ctx
);

impl_add_constant_scalar_in_place!(
    "Shifts each processed channel right by its constant bit count. Updates `source_destination` in place.",
    right_shift_constant_i8_c1_in_place,
    i8,
    u32,
    C1,
    nppiRShiftC_8s_C1IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_i8_c3,
    i8,
    u32,
    C3,
    3,
    nppiRShiftC_8s_C3R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_i8_c3_in_place,
    i8,
    u32,
    C3,
    3,
    nppiRShiftC_8s_C3IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    right_shift_constant_i8_ac4,
    i8,
    u32,
    AC4,
    3,
    nppiRShiftC_8s_AC4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    right_shift_constant_i8_ac4_in_place,
    i8,
    u32,
    AC4,
    3,
    nppiRShiftC_8s_AC4IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_i8_c4,
    i8,
    u32,
    C4,
    4,
    nppiRShiftC_8s_C4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_i8_c4_in_place,
    i8,
    u32,
    C4,
    4,
    nppiRShiftC_8s_C4IR_Ctx
);

impl_add_constant_scalar!(
    "Shifts each processed channel right by its constant bit count. Writes the result to `destination`.",
    right_shift_constant_u16_c1,
    u16,
    u32,
    C1,
    nppiRShiftC_16u_C1R_Ctx
);

impl_add_constant_scalar_in_place!(
    "Shifts each processed channel right by its constant bit count. Updates `source_destination` in place.",
    right_shift_constant_u16_c1_in_place,
    u16,
    u32,
    C1,
    nppiRShiftC_16u_C1IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_u16_c3,
    u16,
    u32,
    C3,
    3,
    nppiRShiftC_16u_C3R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_u16_c3_in_place,
    u16,
    u32,
    C3,
    3,
    nppiRShiftC_16u_C3IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    right_shift_constant_u16_ac4,
    u16,
    u32,
    AC4,
    3,
    nppiRShiftC_16u_AC4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    right_shift_constant_u16_ac4_in_place,
    u16,
    u32,
    AC4,
    3,
    nppiRShiftC_16u_AC4IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_u16_c4,
    u16,
    u32,
    C4,
    4,
    nppiRShiftC_16u_C4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_u16_c4_in_place,
    u16,
    u32,
    C4,
    4,
    nppiRShiftC_16u_C4IR_Ctx
);

impl_add_constant_scalar!(
    "Shifts each processed channel right by its constant bit count. Writes the result to `destination`.",
    right_shift_constant_i16_c1,
    i16,
    u32,
    C1,
    nppiRShiftC_16s_C1R_Ctx
);

impl_add_constant_scalar_in_place!(
    "Shifts each processed channel right by its constant bit count. Updates `source_destination` in place.",
    right_shift_constant_i16_c1_in_place,
    i16,
    u32,
    C1,
    nppiRShiftC_16s_C1IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_i16_c3,
    i16,
    u32,
    C3,
    3,
    nppiRShiftC_16s_C3R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_i16_c3_in_place,
    i16,
    u32,
    C3,
    3,
    nppiRShiftC_16s_C3IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    right_shift_constant_i16_ac4,
    i16,
    u32,
    AC4,
    3,
    nppiRShiftC_16s_AC4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    right_shift_constant_i16_ac4_in_place,
    i16,
    u32,
    AC4,
    3,
    nppiRShiftC_16s_AC4IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_i16_c4,
    i16,
    u32,
    C4,
    4,
    nppiRShiftC_16s_C4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_i16_c4_in_place,
    i16,
    u32,
    C4,
    4,
    nppiRShiftC_16s_C4IR_Ctx
);

impl_add_constant_scalar!(
    "Shifts each processed channel right by its constant bit count. Writes the result to `destination`.",
    right_shift_constant_i32_c1,
    i32,
    u32,
    C1,
    nppiRShiftC_32s_C1R_Ctx
);

impl_add_constant_scalar_in_place!(
    "Shifts each processed channel right by its constant bit count. Updates `source_destination` in place.",
    right_shift_constant_i32_c1_in_place,
    i32,
    u32,
    C1,
    nppiRShiftC_32s_C1IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_i32_c3,
    i32,
    u32,
    C3,
    3,
    nppiRShiftC_32s_C3R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_i32_c3_in_place,
    i32,
    u32,
    C3,
    3,
    nppiRShiftC_32s_C3IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    right_shift_constant_i32_ac4,
    i32,
    u32,
    AC4,
    3,
    nppiRShiftC_32s_AC4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    right_shift_constant_i32_ac4_in_place,
    i32,
    u32,
    AC4,
    3,
    nppiRShiftC_32s_AC4IR_Ctx
);

impl_add_constant_array!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Writes the result to `destination`.",
    right_shift_constant_i32_c4,
    i32,
    u32,
    C4,
    4,
    nppiRShiftC_32s_C4R_Ctx
);

impl_add_constant_array_in_place!(
    "Shifts each processed channel right by its constant bit count. Uses one constant per processed channel. Updates `source_destination` in place.",
    right_shift_constant_i32_c4_in_place,
    i32,
    u32,
    C4,
    4,
    nppiRShiftC_32s_C4IR_Ctx
);
