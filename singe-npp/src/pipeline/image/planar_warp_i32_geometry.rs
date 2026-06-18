use crate::{context::StreamContext, error::Result, image::geometry};

use super::super::planar::{PlanarImage, PlanarWarpAffine, PlanarWarpPerspective, PlanarWarpQuad};

impl_planar_warp_geometry!(
    i32,
    3,
    geometry::warp_affine_i32_p3,
    geometry::warp_affine_back_i32_p3,
    geometry::warp_affine_quad_i32_p3,
    geometry::warp_perspective_i32_p3,
    geometry::warp_perspective_back_i32_p3,
    geometry::warp_perspective_quad_i32_p3
);
impl_planar_warp_geometry!(
    i32,
    4,
    geometry::warp_affine_i32_p4,
    geometry::warp_affine_back_i32_p4,
    geometry::warp_affine_quad_i32_p4,
    geometry::warp_perspective_i32_p4,
    geometry::warp_perspective_back_i32_p4,
    geometry::warp_perspective_quad_i32_p4
);
