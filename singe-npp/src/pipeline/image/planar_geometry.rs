use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{C1, ImageView, PlanarImageView, PlanarImageViewMut},
    },
};

use super::planar::{PlanarImage, PlanarRemap, PlanarResize, PlanarResizeSqrPixel};

macro_rules! impl_planar_resize_geometry {
    ($ty:ty, $planes:literal, $resize:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn resize(
                self,
                stream_context: &StreamContext,
                resize: geometry::Resize,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    resize.destination_roi.size(),
                    &resize,
                    $resize as PlanarResize<$ty, $planes>,
                )
            }
        }
    };
}

macro_rules! impl_planar_resize_sqr_pixel_geometry {
    ($ty:ty, $planes:literal, $resize:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn resize_sqr_pixel(
                self,
                stream_context: &StreamContext,
                resize: geometry::ResizeSqrPixel,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    resize.destination_roi.size(),
                    &resize,
                    $resize as PlanarResizeSqrPixel<$ty, $planes>,
                )
            }
        }
    };
}

macro_rules! impl_planar_remap_geometry {
    ($ty:ty, $map_ty:ty, $planes:literal, $remap:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn remap(
                self,
                stream_context: &StreamContext,
                remap: geometry::Remap,
                x_map: &ImageView<'_, $map_ty, C1>,
                y_map: &ImageView<'_, $map_ty, C1>,
            ) -> Result<Self> {
                self.remap_geometry(
                    stream_context,
                    &remap,
                    x_map,
                    y_map,
                    $remap as PlanarRemap<$ty, $map_ty, $planes>,
                )
            }
        }
    };
}

macro_rules! impl_planar_resize_sqr_pixel_geometry_into {
    ($ty:ty, $planes:literal, $resize:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn resize_sqr_pixel_into(
                stream_context: &StreamContext,
                resize: &geometry::ResizeSqrPixel,
                source: &PlanarImageView<'_, $ty, $planes>,
                destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
            ) -> Result<()> {
                $resize(stream_context, resize, source, destination)
            }
        }
    };
}

macro_rules! impl_planar_remap_geometry_into {
    ($ty:ty, $map_ty:ty, $planes:literal, $remap:path) => {
        impl PlanarImage<$ty, $planes> {
            pub fn remap_into(
                stream_context: &StreamContext,
                remap: &geometry::Remap,
                source: &PlanarImageView<'_, $ty, $planes>,
                x_map: &ImageView<'_, $map_ty, C1>,
                y_map: &ImageView<'_, $map_ty, C1>,
                destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
            ) -> Result<()> {
                $remap(stream_context, remap, source, x_map, y_map, destination)
            }
        }
    };
}

impl_planar_resize_geometry!(u8, 3, geometry::resize_u8_p3);
impl_planar_resize_geometry!(u8, 4, geometry::resize_u8_p4);
impl_planar_resize_geometry!(u16, 3, geometry::resize_u16_p3);
impl_planar_resize_geometry!(u16, 4, geometry::resize_u16_p4);
impl_planar_resize_geometry!(i16, 3, geometry::resize_i16_p3);
impl_planar_resize_geometry!(i16, 4, geometry::resize_i16_p4);
impl_planar_resize_geometry!(f32, 3, geometry::resize_f32_p3);
impl_planar_resize_geometry!(f32, 4, geometry::resize_f32_p4);

impl_planar_resize_sqr_pixel_geometry!(u8, 3, geometry::resize_sqr_pixel_u8_p3);
impl_planar_resize_sqr_pixel_geometry!(u8, 4, geometry::resize_sqr_pixel_u8_p4);
impl_planar_resize_sqr_pixel_geometry!(u16, 3, geometry::resize_sqr_pixel_u16_p3);
impl_planar_resize_sqr_pixel_geometry!(u16, 4, geometry::resize_sqr_pixel_u16_p4);
impl_planar_resize_sqr_pixel_geometry!(i16, 3, geometry::resize_sqr_pixel_i16_p3);
impl_planar_resize_sqr_pixel_geometry!(i16, 4, geometry::resize_sqr_pixel_i16_p4);
impl_planar_resize_sqr_pixel_geometry!(f32, 3, geometry::resize_sqr_pixel_f32_p3);
impl_planar_resize_sqr_pixel_geometry!(f32, 4, geometry::resize_sqr_pixel_f32_p4);
impl_planar_resize_sqr_pixel_geometry_into!(f64, 3, geometry::resize_sqr_pixel_f64_p3);
impl_planar_resize_sqr_pixel_geometry_into!(f64, 4, geometry::resize_sqr_pixel_f64_p4);

impl_planar_remap_geometry!(u8, f32, 3, geometry::remap_u8_p3);
impl_planar_remap_geometry!(u8, f32, 4, geometry::remap_u8_p4);
impl_planar_remap_geometry!(u16, f32, 3, geometry::remap_u16_p3);
impl_planar_remap_geometry!(u16, f32, 4, geometry::remap_u16_p4);
impl_planar_remap_geometry!(i16, f32, 3, geometry::remap_i16_p3);
impl_planar_remap_geometry!(i16, f32, 4, geometry::remap_i16_p4);
impl_planar_remap_geometry!(f32, f32, 3, geometry::remap_f32_p3);
impl_planar_remap_geometry!(f32, f32, 4, geometry::remap_f32_p4);
impl_planar_remap_geometry_into!(f64, f64, 3, geometry::remap_f64_p3);
impl_planar_remap_geometry_into!(f64, f64, 4, geometry::remap_f64_p4);
