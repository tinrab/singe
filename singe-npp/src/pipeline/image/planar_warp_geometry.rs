use crate::{context::StreamContext, error::Result, image::geometry};

use super::planar::{PlanarImage, PlanarWarpAffine, PlanarWarpPerspective, PlanarWarpQuad};

#[path = "planar_warp_f64_affine_geometry.rs"]
mod f64_affine_geometry;

macro_rules! impl_planar_warp_geometry {
    ($ty:ty, $planes:literal, $affine:path, $affine_back:path, $affine_quad:path, $perspective:path, $perspective_back:path, $perspective_quad:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn warp_affine(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpAffine,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    warp.destination_roi.size(),
                    &warp,
                    $affine as PlanarWarpAffine<$ty, $planes>,
                )
            }

            pub fn warp_affine_back(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpAffine,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    warp.destination_roi.size(),
                    &warp,
                    $affine_back as PlanarWarpAffine<$ty, $planes>,
                )
            }

            pub fn warp_affine_quad(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpQuad,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    warp.destination_roi.size(),
                    &warp,
                    $affine_quad as PlanarWarpQuad<$ty, $planes>,
                )
            }

            pub fn warp_perspective(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpPerspective,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    warp.destination_roi.size(),
                    &warp,
                    $perspective as PlanarWarpPerspective<$ty, $planes>,
                )
            }

            pub fn warp_perspective_back(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpPerspective,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    warp.destination_roi.size(),
                    &warp,
                    $perspective_back as PlanarWarpPerspective<$ty, $planes>,
                )
            }

            pub fn warp_perspective_quad(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpQuad,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    warp.destination_roi.size(),
                    &warp,
                    $perspective_quad as PlanarWarpQuad<$ty, $planes>,
                )
            }
        }
    };
}

#[path = "planar_warp_f32_geometry.rs"]
mod f32_geometry;
#[path = "planar_warp_i32_geometry.rs"]
mod i32_geometry;

impl_planar_warp_geometry!(
    u8,
    3,
    geometry::warp_affine_u8_p3,
    geometry::warp_affine_back_u8_p3,
    geometry::warp_affine_quad_u8_p3,
    geometry::warp_perspective_u8_p3,
    geometry::warp_perspective_back_u8_p3,
    geometry::warp_perspective_quad_u8_p3
);
impl_planar_warp_geometry!(
    u8,
    4,
    geometry::warp_affine_u8_p4,
    geometry::warp_affine_back_u8_p4,
    geometry::warp_affine_quad_u8_p4,
    geometry::warp_perspective_u8_p4,
    geometry::warp_perspective_back_u8_p4,
    geometry::warp_perspective_quad_u8_p4
);
impl_planar_warp_geometry!(
    u16,
    3,
    geometry::warp_affine_u16_p3,
    geometry::warp_affine_back_u16_p3,
    geometry::warp_affine_quad_u16_p3,
    geometry::warp_perspective_u16_p3,
    geometry::warp_perspective_back_u16_p3,
    geometry::warp_perspective_quad_u16_p3
);
impl_planar_warp_geometry!(
    u16,
    4,
    geometry::warp_affine_u16_p4,
    geometry::warp_affine_back_u16_p4,
    geometry::warp_affine_quad_u16_p4,
    geometry::warp_perspective_u16_p4,
    geometry::warp_perspective_back_u16_p4,
    geometry::warp_perspective_quad_u16_p4
);
