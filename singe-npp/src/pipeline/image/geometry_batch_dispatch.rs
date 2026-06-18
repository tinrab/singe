use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::ImagePipeline;

#[macro_use]
#[path = "geometry_batch_dispatch_macros.rs"]
mod macros;

impl_mirror_batch_image!(
    f32,
    C1,
    geometry::mirror_batch_f32_c1,
    geometry::mirror_batch_f32_c1_in_place
);
impl_mirror_batch_image!(
    f32,
    C3,
    geometry::mirror_batch_f32_c3,
    geometry::mirror_batch_f32_c3_in_place
);
impl_mirror_batch_image!(
    f32,
    C4,
    geometry::mirror_batch_f32_c4,
    geometry::mirror_batch_f32_c4_in_place
);
impl_mirror_batch_image!(
    f32,
    AC4,
    geometry::mirror_batch_f32_ac4,
    geometry::mirror_batch_f32_ac4_in_place
);

impl_resize_batch_image!(
    u8,
    C1,
    geometry::resize_batch_u8_c1,
    geometry::resize_batch_u8_c1_advanced
);
impl_resize_batch_image!(
    u8,
    C3,
    geometry::resize_batch_u8_c3,
    geometry::resize_batch_u8_c3_advanced
);
impl_resize_batch_image!(
    u8,
    C4,
    geometry::resize_batch_u8_c4,
    geometry::resize_batch_u8_c4_advanced
);
impl_resize_batch_image!(
    u8,
    AC4,
    geometry::resize_batch_u8_ac4,
    geometry::resize_batch_u8_ac4_advanced
);
impl_resize_batch_advanced_image!(f16, C1, geometry::resize_batch_f16_c1_advanced);
impl_resize_batch_advanced_image!(f16, C3, geometry::resize_batch_f16_c3_advanced);
impl_resize_batch_advanced_image!(f16, C4, geometry::resize_batch_f16_c4_advanced);
impl_resize_batch_image!(
    f32,
    C1,
    geometry::resize_batch_f32_c1,
    geometry::resize_batch_f32_c1_advanced
);
impl_resize_batch_image!(
    f32,
    C3,
    geometry::resize_batch_f32_c3,
    geometry::resize_batch_f32_c3_advanced
);
impl_resize_batch_image!(
    f32,
    C4,
    geometry::resize_batch_f32_c4,
    geometry::resize_batch_f32_c4_advanced
);
impl_resize_batch_image!(
    f32,
    AC4,
    geometry::resize_batch_f32_ac4,
    geometry::resize_batch_f32_ac4_advanced
);

#[path = "geometry_warp_batch_dispatch.rs"]
mod warp_batch_dispatch;
