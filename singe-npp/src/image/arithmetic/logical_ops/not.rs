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

impl_unary_operation!(
    "Inverts the bits of each processed channel value. Writes the result to `destination`.",
    logical_not_u8_c1,
    u8,
    C1,
    nppiNot_8u_C1R_Ctx
);

impl_unary_operation_in_place!(
    "Inverts the bits of each processed channel value. Updates `source_destination` in place.",
    logical_not_u8_c1_in_place,
    u8,
    C1,
    nppiNot_8u_C1IR_Ctx
);

impl_unary_operation!(
    "Inverts the bits of each processed channel value. Writes the result to `destination`.",
    logical_not_u8_c3,
    u8,
    C3,
    nppiNot_8u_C3R_Ctx
);

impl_unary_operation_in_place!(
    "Inverts the bits of each processed channel value. Updates `source_destination` in place.",
    logical_not_u8_c3_in_place,
    u8,
    C3,
    nppiNot_8u_C3IR_Ctx
);

impl_unary_operation!(
    "Inverts the bits of each processed channel value. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    logical_not_u8_ac4,
    u8,
    AC4,
    nppiNot_8u_AC4R_Ctx
);

impl_unary_operation_in_place!(
    "Inverts the bits of each processed channel value. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    logical_not_u8_ac4_in_place,
    u8,
    AC4,
    nppiNot_8u_AC4IR_Ctx
);

impl_unary_operation!(
    "Inverts the bits of each processed channel value. Writes the result to `destination`.",
    logical_not_u8_c4,
    u8,
    C4,
    nppiNot_8u_C4R_Ctx
);

impl_unary_operation_in_place!(
    "Inverts the bits of each processed channel value. Updates `source_destination` in place.",
    logical_not_u8_c4_in_place,
    u8,
    C4,
    nppiNot_8u_C4IR_Ctx
);
