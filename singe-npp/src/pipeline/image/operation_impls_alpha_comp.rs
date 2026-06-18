use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::AlphaOperation,
};

use super::super::{ImagePipeline, operation_traits::*};

impl_alpha_comp_constant_image!(u8, C1, u8, arithmetic::alpha_comp_constant_u8_c1);
impl_alpha_comp_constant_image!(u8, C3, u8, arithmetic::alpha_comp_constant_u8_c3);
impl_alpha_comp_constant_image!(u8, C4, u8, arithmetic::alpha_comp_constant_u8_c4);
impl_alpha_comp_constant_image!(u8, AC4, u8, arithmetic::alpha_comp_constant_u8_ac4);
impl_alpha_comp_constant_image!(i8, C1, i8, arithmetic::alpha_comp_constant_i8_c1);
impl_alpha_comp_constant_image!(u16, C1, u16, arithmetic::alpha_comp_constant_u16_c1);
impl_alpha_comp_constant_image!(u16, C3, u16, arithmetic::alpha_comp_constant_u16_c3);
impl_alpha_comp_constant_image!(u16, C4, u16, arithmetic::alpha_comp_constant_u16_c4);
impl_alpha_comp_constant_image!(u16, AC4, u16, arithmetic::alpha_comp_constant_u16_ac4);
impl_alpha_comp_constant_image!(i16, C1, i16, arithmetic::alpha_comp_constant_i16_c1);
impl_alpha_comp_constant_image!(u32, C1, u32, arithmetic::alpha_comp_constant_u32_c1);
impl_alpha_comp_constant_image!(i32, C1, i32, arithmetic::alpha_comp_constant_i32_c1);
impl_alpha_comp_constant_image!(f32, C1, f32, arithmetic::alpha_comp_constant_f32_c1);

impl_alpha_comp_image!(u8, C1, arithmetic::alpha_comp_u8_ac1);
impl_alpha_comp_image!(u8, AC4, arithmetic::alpha_comp_u8_ac4);
impl_alpha_comp_image!(i8, C1, arithmetic::alpha_comp_i8_ac1);
impl_alpha_comp_image!(u16, C1, arithmetic::alpha_comp_u16_ac1);
impl_alpha_comp_image!(u16, AC4, arithmetic::alpha_comp_u16_ac4);
impl_alpha_comp_image!(i16, C1, arithmetic::alpha_comp_i16_ac1);
impl_alpha_comp_image!(u32, C1, arithmetic::alpha_comp_u32_ac1);
impl_alpha_comp_image!(u32, AC4, arithmetic::alpha_comp_u32_ac4);
impl_alpha_comp_image!(i32, C1, arithmetic::alpha_comp_i32_ac1);
impl_alpha_comp_image!(i32, AC4, arithmetic::alpha_comp_i32_ac4);
impl_alpha_comp_image!(f32, C1, arithmetic::alpha_comp_f32_ac1);
impl_alpha_comp_image!(f32, AC4, arithmetic::alpha_comp_f32_ac4);
