use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{
            AC4, C1, C2, C3, C4, ImageOutputView, ImageView, ImageViewMut, PlanarImageOutputView,
            PlanarImageView, PlanarImageViewMut,
        },
    },
    pipeline::image::{
        ImagePipeline,
        color_dispatch::{ColorTwistImage, PlanarColorTwistImage},
    },
};

impl_color_twist_image!(
    u8,
    C1,
    color::color_twist_u8_c1,
    color::color_twist_u8_c1_in_place
);

impl_color_twist_image!(
    u8,
    C2,
    color::color_twist_u8_c2,
    color::color_twist_u8_c2_in_place
);

impl_color_twist_image!(
    u8,
    C3,
    color::color_twist_u8_c3,
    color::color_twist_u8_c3_in_place
);

impl_color_twist_image!(
    u8,
    C4,
    color::color_twist_u8_c4,
    color::color_twist_u8_c4_in_place
);

impl_color_twist_image!(
    u8,
    AC4,
    color::color_twist_u8_ac4,
    color::color_twist_u8_ac4_in_place
);

impl_planar_color_twist_image!(
    u8,
    color::color_twist_u8_p3,
    color::color_twist_u8_p3_in_place
);

impl_color_twist_image!(
    u16,
    C1,
    color::color_twist_u16_c1,
    color::color_twist_u16_c1_in_place
);

impl_color_twist_image!(
    u16,
    C2,
    color::color_twist_u16_c2,
    color::color_twist_u16_c2_in_place
);

impl_color_twist_image!(
    u16,
    C3,
    color::color_twist_u16_c3,
    color::color_twist_u16_c3_in_place
);

impl_color_twist_image!(
    u16,
    AC4,
    color::color_twist_u16_ac4,
    color::color_twist_u16_ac4_in_place
);

impl_planar_color_twist_image!(
    u16,
    color::color_twist_u16_p3,
    color::color_twist_u16_p3_in_place
);

impl_color_twist_image!(
    i16,
    C1,
    color::color_twist_i16_c1,
    color::color_twist_i16_c1_in_place
);

impl_color_twist_image!(
    i16,
    C2,
    color::color_twist_i16_c2,
    color::color_twist_i16_c2_in_place
);

impl_color_twist_image!(
    i16,
    C3,
    color::color_twist_i16_c3,
    color::color_twist_i16_c3_in_place
);

impl_color_twist_image!(
    i16,
    AC4,
    color::color_twist_i16_ac4,
    color::color_twist_i16_ac4_in_place
);

impl_planar_color_twist_image!(
    i16,
    color::color_twist_i16_p3,
    color::color_twist_i16_p3_in_place
);

mod f16_impls;

mod i8_impls;

mod constants_impls;

mod batch_impls;

mod f32_impls;
