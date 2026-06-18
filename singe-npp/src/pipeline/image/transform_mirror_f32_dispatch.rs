use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::Axis,
};

use super::{ImagePipeline, MirrorImage};

impl_mirror_image!(
    f32,
    C1,
    geometry::mirror_f32_c1,
    geometry::mirror_f32_c1_in_place
);
impl_mirror_image!(
    f32,
    C3,
    geometry::mirror_f32_c3,
    geometry::mirror_f32_c3_in_place
);
impl_mirror_image!(
    f32,
    C4,
    geometry::mirror_f32_c4,
    geometry::mirror_f32_c4_in_place
);
impl_mirror_image!(
    f32,
    AC4,
    geometry::mirror_f32_ac4,
    geometry::mirror_f32_ac4_in_place
);
