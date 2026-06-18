use crate::{context::StreamContext, error::Result, image::geometry};

use super::super::planar::{PlanarImage, PlanarWarpAffine, PlanarWarpPerspective, PlanarWarpQuad};

impl_planar_warp_geometry!(
    f32,
    3,
    geometry::warp_affine_f32_p3,
    geometry::warp_affine_back_f32_p3,
    geometry::warp_affine_quad_f32_p3,
    geometry::warp_perspective_f32_p3,
    geometry::warp_perspective_back_f32_p3,
    geometry::warp_perspective_quad_f32_p3
);
impl_planar_warp_geometry!(
    f32,
    4,
    geometry::warp_affine_f32_p4,
    geometry::warp_affine_back_f32_p4,
    geometry::warp_affine_quad_f32_p4,
    geometry::warp_perspective_f32_p4,
    geometry::warp_perspective_back_f32_p4,
    geometry::warp_perspective_quad_f32_p4
);
