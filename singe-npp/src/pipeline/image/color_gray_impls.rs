use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    },
    types::{BayerGridPosition, ImageNormalization, Rectangle},
};

use super::ImagePipeline;
use super::color_dispatch::*;

impl_gamma_image!(
    C3,
    color::gamma_forward_u8_c3,
    color::gamma_forward_u8_c3_in_place,
    color::gamma_inverse_u8_c3,
    color::gamma_inverse_u8_c3_in_place
);
impl_gamma_image!(
    AC4,
    color::gamma_forward_u8_ac4,
    color::gamma_forward_u8_ac4_in_place,
    color::gamma_inverse_u8_ac4,
    color::gamma_inverse_u8_ac4_in_place
);
impl_planar_gamma_image!(
    color::gamma_forward_u8_p3,
    color::gamma_forward_u8_p3_in_place,
    color::gamma_inverse_u8_p3,
    color::gamma_inverse_u8_p3_in_place
);

impl_rgb_to_gray_image!(u8, C3, color::rgb_to_gray_u8_c3);
impl_rgb_to_gray_image!(u8, AC4, color::rgb_to_gray_u8_ac4);
impl_rgb_to_gray_image!(u16, C3, color::rgb_to_gray_u16_c3);
impl_rgb_to_gray_image!(u16, AC4, color::rgb_to_gray_u16_ac4);
impl_rgb_to_gray_image!(i16, C3, color::rgb_to_gray_i16_c3);
impl_rgb_to_gray_image!(i16, AC4, color::rgb_to_gray_i16_ac4);
impl_rgb_to_gray_image!(f32, C3, color::rgb_to_gray_f32_c3);
impl_rgb_to_gray_image!(f32, AC4, color::rgb_to_gray_f32_ac4);

impl_color_to_gray_image!(u8, C3, 3, color::color_to_gray_u8_c3);
impl_color_to_gray_image!(u8, AC4, 3, color::color_to_gray_u8_ac4);
impl_color_to_gray_image!(u8, C4, 4, color::color_to_gray_u8_c4);
impl_color_to_gray_image!(u16, C3, 3, color::color_to_gray_u16_c3);
impl_color_to_gray_image!(u16, AC4, 3, color::color_to_gray_u16_ac4);
impl_color_to_gray_image!(u16, C4, 4, color::color_to_gray_u16_c4);
impl_color_to_gray_image!(i16, C3, 3, color::color_to_gray_i16_c3);
impl_color_to_gray_image!(i16, AC4, 3, color::color_to_gray_i16_ac4);
impl_color_to_gray_image!(i16, C4, 4, color::color_to_gray_i16_c4);
impl_color_to_gray_image!(f32, C3, 3, color::color_to_gray_f32_c3);
impl_color_to_gray_image!(f32, AC4, 3, color::color_to_gray_f32_ac4);
impl_color_to_gray_image!(f32, C4, 4, color::color_to_gray_f32_c4);

impl_gradient_color_to_gray_image!(u8, color::gradient_color_to_gray_u8_c3);
impl_gradient_color_to_gray_image!(u16, color::gradient_color_to_gray_u16_c3);
impl_gradient_color_to_gray_image!(i16, color::gradient_color_to_gray_i16_c3);
impl_gradient_color_to_gray_image!(f32, color::gradient_color_to_gray_f32_c3);

impl_cfa_to_rgb_image!(u8, color::cfa_to_rgb_u8_c3);
impl_cfa_to_rgba_image!(u8, color::cfa_to_rgba_u8_ac4);
impl_cfa_to_rgb_image!(u16, color::cfa_to_rgb_u16_c3);
impl_cfa_to_rgba_image!(u16, color::cfa_to_rgba_u16_ac4);
impl_cfa_to_rgb_image!(u32, color::cfa_to_rgb_u32_c3);
impl_cfa_to_rgba_image!(u32, color::cfa_to_rgba_u32_ac4);
