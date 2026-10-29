pub use crate::pipeline::image::ImagePipeline;
use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
};

mod traits;

#[macro_use]
mod macros;

pub use traits::{
    RemapImage, ResizeSqrPixelAdvancedImage, ResizeSqrPixelImage, RotateImage, WarpAffineBackImage,
    WarpAffineImage, WarpAffineQuadImage, WarpPerspectiveBackImage, WarpPerspectiveImage,
    WarpPerspectiveQuadImage,
};

impl_geometry_for_layout!(
    u8,
    C1,
    geometry::rotate_u8_c1,
    geometry::resize_sqr_pixel_u8_c1,
    geometry::remap_u8_c1,
    geometry::warp_affine_u8_c1,
    geometry::warp_affine_back_u8_c1,
    geometry::warp_affine_quad_u8_c1,
    geometry::warp_perspective_u8_c1,
    geometry::warp_perspective_back_u8_c1,
    geometry::warp_perspective_quad_u8_c1
);

impl_geometry_for_layout!(
    u8,
    C3,
    geometry::rotate_u8_c3,
    geometry::resize_sqr_pixel_u8_c3,
    geometry::remap_u8_c3,
    geometry::warp_affine_u8_c3,
    geometry::warp_affine_back_u8_c3,
    geometry::warp_affine_quad_u8_c3,
    geometry::warp_perspective_u8_c3,
    geometry::warp_perspective_back_u8_c3,
    geometry::warp_perspective_quad_u8_c3
);

impl_geometry_for_layout!(
    u8,
    C4,
    geometry::rotate_u8_c4,
    geometry::resize_sqr_pixel_u8_c4,
    geometry::remap_u8_c4,
    geometry::warp_affine_u8_c4,
    geometry::warp_affine_back_u8_c4,
    geometry::warp_affine_quad_u8_c4,
    geometry::warp_perspective_u8_c4,
    geometry::warp_perspective_back_u8_c4,
    geometry::warp_perspective_quad_u8_c4
);

impl_geometry_for_layout!(
    u8,
    AC4,
    geometry::rotate_u8_ac4,
    geometry::resize_sqr_pixel_u8_ac4,
    geometry::remap_u8_ac4,
    geometry::warp_affine_u8_ac4,
    geometry::warp_affine_back_u8_ac4,
    geometry::warp_affine_quad_u8_ac4,
    geometry::warp_perspective_u8_ac4,
    geometry::warp_perspective_back_u8_ac4,
    geometry::warp_perspective_quad_u8_ac4
);

impl_geometry_for_layout!(
    u16,
    C1,
    geometry::rotate_u16_c1,
    geometry::resize_sqr_pixel_u16_c1,
    geometry::remap_u16_c1,
    geometry::warp_affine_u16_c1,
    geometry::warp_affine_back_u16_c1,
    geometry::warp_affine_quad_u16_c1,
    geometry::warp_perspective_u16_c1,
    geometry::warp_perspective_back_u16_c1,
    geometry::warp_perspective_quad_u16_c1
);

impl_geometry_for_layout!(
    u16,
    C3,
    geometry::rotate_u16_c3,
    geometry::resize_sqr_pixel_u16_c3,
    geometry::remap_u16_c3,
    geometry::warp_affine_u16_c3,
    geometry::warp_affine_back_u16_c3,
    geometry::warp_affine_quad_u16_c3,
    geometry::warp_perspective_u16_c3,
    geometry::warp_perspective_back_u16_c3,
    geometry::warp_perspective_quad_u16_c3
);

impl_geometry_for_layout!(
    u16,
    C4,
    geometry::rotate_u16_c4,
    geometry::resize_sqr_pixel_u16_c4,
    geometry::remap_u16_c4,
    geometry::warp_affine_u16_c4,
    geometry::warp_affine_back_u16_c4,
    geometry::warp_affine_quad_u16_c4,
    geometry::warp_perspective_u16_c4,
    geometry::warp_perspective_back_u16_c4,
    geometry::warp_perspective_quad_u16_c4
);

impl_geometry_for_layout!(
    u16,
    AC4,
    geometry::rotate_u16_ac4,
    geometry::resize_sqr_pixel_u16_ac4,
    geometry::remap_u16_ac4,
    geometry::warp_affine_u16_ac4,
    geometry::warp_affine_back_u16_ac4,
    geometry::warp_affine_quad_u16_ac4,
    geometry::warp_perspective_u16_ac4,
    geometry::warp_perspective_back_u16_ac4,
    geometry::warp_perspective_quad_u16_ac4
);

impl_resize_sqr_pixel_image!(i16, C3, geometry::resize_sqr_pixel_i16_c3);

impl_remap_image!(i16, f32, C3, geometry::remap_i16_c3);

mod warp_impls;

mod float_impls;
