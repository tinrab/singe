use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{PlanarImageView, PlanarImageViewMut},
    },
};

use super::PlanarImage;

macro_rules! impl_planar_warp_affine_geometry_into {
    ($ty:ty, $planes:literal, $affine:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn warp_affine_into(
                stream_context: &StreamContext,
                warp: &geometry::WarpAffine,
                source: &PlanarImageView<'_, $ty, $planes>,
                destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
            ) -> Result<()> {
                $affine(stream_context, warp, source, destination)
            }
        }
    };
}

impl_planar_warp_affine_geometry_into!(f64, 3, geometry::warp_affine_f64_p3);
impl_planar_warp_affine_geometry_into!(f64, 4, geometry::warp_affine_f64_p4);
