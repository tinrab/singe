use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{
    ImagePipeline, RemapImage, ResizeSqrPixelAdvancedImage, ResizeSqrPixelImage, RotateImage,
    WarpAffineBackImage, WarpAffineImage, WarpAffineQuadImage, WarpPerspectiveBackImage,
    WarpPerspectiveImage, WarpPerspectiveQuadImage,
};

impl_geometry_for_layout!(
    f32,
    C1,
    geometry::rotate_f32_c1,
    geometry::resize_sqr_pixel_f32_c1,
    geometry::remap_f32_c1,
    geometry::warp_affine_f32_c1,
    geometry::warp_affine_back_f32_c1,
    geometry::warp_affine_quad_f32_c1,
    geometry::warp_perspective_f32_c1,
    geometry::warp_perspective_back_f32_c1,
    geometry::warp_perspective_quad_f32_c1
);
impl_geometry_for_layout!(
    f32,
    C3,
    geometry::rotate_f32_c3,
    geometry::resize_sqr_pixel_f32_c3,
    geometry::remap_f32_c3,
    geometry::warp_affine_f32_c3,
    geometry::warp_affine_back_f32_c3,
    geometry::warp_affine_quad_f32_c3,
    geometry::warp_perspective_f32_c3,
    geometry::warp_perspective_back_f32_c3,
    geometry::warp_perspective_quad_f32_c3
);
impl_geometry_for_layout!(
    f32,
    C4,
    geometry::rotate_f32_c4,
    geometry::resize_sqr_pixel_f32_c4,
    geometry::remap_f32_c4,
    geometry::warp_affine_f32_c4,
    geometry::warp_affine_back_f32_c4,
    geometry::warp_affine_quad_f32_c4,
    geometry::warp_perspective_f32_c4,
    geometry::warp_perspective_back_f32_c4,
    geometry::warp_perspective_quad_f32_c4
);
impl_geometry_for_layout!(
    f32,
    AC4,
    geometry::rotate_f32_ac4,
    geometry::resize_sqr_pixel_f32_ac4,
    geometry::remap_f32_ac4,
    geometry::warp_affine_f32_ac4,
    geometry::warp_affine_back_f32_ac4,
    geometry::warp_affine_quad_f32_ac4,
    geometry::warp_perspective_f32_ac4,
    geometry::warp_perspective_back_f32_ac4,
    geometry::warp_perspective_quad_f32_ac4
);

impl_resize_sqr_pixel_image!(f64, C1, geometry::resize_sqr_pixel_f64_c1);
impl_resize_sqr_pixel_image!(f64, C3, geometry::resize_sqr_pixel_f64_c3);
impl_resize_sqr_pixel_image!(f64, C4, geometry::resize_sqr_pixel_f64_c4);
impl_resize_sqr_pixel_image!(f64, AC4, geometry::resize_sqr_pixel_f64_ac4);
impl_remap_image!(f64, f64, C1, geometry::remap_f64_c1);
impl_remap_image!(f64, f64, C3, geometry::remap_f64_c3);
impl_remap_image!(f64, f64, C4, geometry::remap_f64_c4);
impl_remap_image!(f64, f64, AC4, geometry::remap_f64_ac4);

impl_resize_sqr_pixel_image!(i16, C1, geometry::resize_sqr_pixel_i16_c1);
impl_resize_sqr_pixel_image!(i16, C4, geometry::resize_sqr_pixel_i16_c4);
impl_resize_sqr_pixel_image!(i16, AC4, geometry::resize_sqr_pixel_i16_ac4);
impl_resize_sqr_pixel_advanced_image!(u8, C1, geometry::resize_sqr_pixel_c1_advanced);
impl_remap_image!(i16, f32, C1, geometry::remap_i16_c1);
impl_remap_image!(i16, f32, C4, geometry::remap_i16_c4);
impl_remap_image!(i16, f32, AC4, geometry::remap_i16_ac4);
