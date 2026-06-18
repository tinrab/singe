use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::LogicalConstantImage};

impl_logical_constant_image!(
    u8,
    C1,
    u8,
    arithmetic::and_constant_u8_c1,
    arithmetic::and_constant_u8_c1_in_place,
    arithmetic::or_constant_u8_c1,
    arithmetic::or_constant_u8_c1_in_place,
    arithmetic::xor_constant_u8_c1,
    arithmetic::xor_constant_u8_c1_in_place
);
impl_logical_constant_image!(
    u8,
    C3,
    [u8; 3],
    arithmetic::and_constant_u8_c3,
    arithmetic::and_constant_u8_c3_in_place,
    arithmetic::or_constant_u8_c3,
    arithmetic::or_constant_u8_c3_in_place,
    arithmetic::xor_constant_u8_c3,
    arithmetic::xor_constant_u8_c3_in_place
);
impl_logical_constant_image!(
    u8,
    C4,
    [u8; 4],
    arithmetic::and_constant_u8_c4,
    arithmetic::and_constant_u8_c4_in_place,
    arithmetic::or_constant_u8_c4,
    arithmetic::or_constant_u8_c4_in_place,
    arithmetic::xor_constant_u8_c4,
    arithmetic::xor_constant_u8_c4_in_place
);
impl_logical_constant_image!(
    u8,
    AC4,
    [u8; 3],
    arithmetic::and_constant_u8_ac4,
    arithmetic::and_constant_u8_ac4_in_place,
    arithmetic::or_constant_u8_ac4,
    arithmetic::or_constant_u8_ac4_in_place,
    arithmetic::xor_constant_u8_ac4,
    arithmetic::xor_constant_u8_ac4_in_place
);
impl_logical_constant_image!(
    u16,
    C1,
    u16,
    arithmetic::and_constant_u16_c1,
    arithmetic::and_constant_u16_c1_in_place,
    arithmetic::or_constant_u16_c1,
    arithmetic::or_constant_u16_c1_in_place,
    arithmetic::xor_constant_u16_c1,
    arithmetic::xor_constant_u16_c1_in_place
);
impl_logical_constant_image!(
    u16,
    C3,
    [u16; 3],
    arithmetic::and_constant_u16_c3,
    arithmetic::and_constant_u16_c3_in_place,
    arithmetic::or_constant_u16_c3,
    arithmetic::or_constant_u16_c3_in_place,
    arithmetic::xor_constant_u16_c3,
    arithmetic::xor_constant_u16_c3_in_place
);
impl_logical_constant_image!(
    u16,
    C4,
    [u16; 4],
    arithmetic::and_constant_u16_c4,
    arithmetic::and_constant_u16_c4_in_place,
    arithmetic::or_constant_u16_c4,
    arithmetic::or_constant_u16_c4_in_place,
    arithmetic::xor_constant_u16_c4,
    arithmetic::xor_constant_u16_c4_in_place
);
impl_logical_constant_image!(
    u16,
    AC4,
    [u16; 3],
    arithmetic::and_constant_u16_ac4,
    arithmetic::and_constant_u16_ac4_in_place,
    arithmetic::or_constant_u16_ac4,
    arithmetic::or_constant_u16_ac4_in_place,
    arithmetic::xor_constant_u16_ac4,
    arithmetic::xor_constant_u16_ac4_in_place
);
impl_logical_constant_image!(
    i32,
    C1,
    i32,
    arithmetic::and_constant_i32_c1,
    arithmetic::and_constant_i32_c1_in_place,
    arithmetic::or_constant_i32_c1,
    arithmetic::or_constant_i32_c1_in_place,
    arithmetic::xor_constant_i32_c1,
    arithmetic::xor_constant_i32_c1_in_place
);
impl_logical_constant_image!(
    i32,
    C3,
    [i32; 3],
    arithmetic::and_constant_i32_c3,
    arithmetic::and_constant_i32_c3_in_place,
    arithmetic::or_constant_i32_c3,
    arithmetic::or_constant_i32_c3_in_place,
    arithmetic::xor_constant_i32_c3,
    arithmetic::xor_constant_i32_c3_in_place
);
impl_logical_constant_image!(
    i32,
    C4,
    [i32; 4],
    arithmetic::and_constant_i32_c4,
    arithmetic::and_constant_i32_c4_in_place,
    arithmetic::or_constant_i32_c4,
    arithmetic::or_constant_i32_c4_in_place,
    arithmetic::xor_constant_i32_c4,
    arithmetic::xor_constant_i32_c4_in_place
);
impl_logical_constant_image!(
    i32,
    AC4,
    [i32; 3],
    arithmetic::and_constant_i32_ac4,
    arithmetic::and_constant_i32_ac4_in_place,
    arithmetic::or_constant_i32_ac4,
    arithmetic::or_constant_i32_ac4_in_place,
    arithmetic::xor_constant_i32_ac4,
    arithmetic::xor_constant_i32_ac4_in_place
);
