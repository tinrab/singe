use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{
    ImagePipeline, WarpAffineBackImage, WarpAffineImage, WarpAffineQuadImage,
    WarpPerspectiveBackImage, WarpPerspectiveImage, WarpPerspectiveQuadImage,
};

impl_warp_for_layout!(
    i32,
    C1,
    geometry::warp_affine_i32_c1,
    geometry::warp_affine_back_i32_c1,
    geometry::warp_affine_quad_i32_c1,
    geometry::warp_perspective_i32_c1,
    geometry::warp_perspective_back_i32_c1,
    geometry::warp_perspective_quad_i32_c1
);
impl_warp_for_layout!(
    i32,
    C3,
    geometry::warp_affine_i32_c3,
    geometry::warp_affine_back_i32_c3,
    geometry::warp_affine_quad_i32_c3,
    geometry::warp_perspective_i32_c3,
    geometry::warp_perspective_back_i32_c3,
    geometry::warp_perspective_quad_i32_c3
);
impl_warp_for_layout!(
    i32,
    C4,
    geometry::warp_affine_i32_c4,
    geometry::warp_affine_back_i32_c4,
    geometry::warp_affine_quad_i32_c4,
    geometry::warp_perspective_i32_c4,
    geometry::warp_perspective_back_i32_c4,
    geometry::warp_perspective_quad_i32_c4
);
impl_warp_for_layout!(
    i32,
    AC4,
    geometry::warp_affine_i32_ac4,
    geometry::warp_affine_back_i32_ac4,
    geometry::warp_affine_quad_i32_ac4,
    geometry::warp_perspective_i32_ac4,
    geometry::warp_perspective_back_i32_ac4,
    geometry::warp_perspective_quad_i32_ac4
);
impl_affine_perspective_for_layout!(
    f16,
    C1,
    geometry::warp_affine_f16_c1,
    geometry::warp_perspective_f16_c1
);
impl_affine_perspective_for_layout!(
    f16,
    C3,
    geometry::warp_affine_f16_c3,
    geometry::warp_perspective_f16_c3
);
impl_affine_perspective_for_layout!(
    f16,
    C4,
    geometry::warp_affine_f16_c4,
    geometry::warp_perspective_f16_c4
);
impl_warp_affine_image!(f64, C1, geometry::warp_affine_f64_c1);
impl_warp_affine_image!(f64, C3, geometry::warp_affine_f64_c3);
impl_warp_affine_image!(f64, C4, geometry::warp_affine_f64_c4);
impl_warp_affine_image!(f64, AC4, geometry::warp_affine_f64_ac4);
