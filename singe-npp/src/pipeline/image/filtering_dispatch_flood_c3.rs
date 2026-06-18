use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C3, ImageViewMut},
    },
    types::{ConnectedRegion, ImageNormalization, Point},
};

use super::super::{ImagePipeline, filtering_traits::FloodFillImage};

impl_flood_fill_image!(
    u8,
    C3,
    [u8; 3],
    filtering::flood_fill_u8_c3_in_place,
    filtering::flood_fill_boundary_u8_c3_in_place,
    filtering::flood_fill_range_u8_c3_in_place,
    filtering::flood_fill_range_boundary_u8_c3_in_place,
    filtering::flood_fill_gradient_u8_c3_in_place,
    filtering::flood_fill_gradient_boundary_u8_c3_in_place
);
impl_flood_fill_image!(
    u16,
    C3,
    [u16; 3],
    filtering::flood_fill_u16_c3_in_place,
    filtering::flood_fill_boundary_u16_c3_in_place,
    filtering::flood_fill_range_u16_c3_in_place,
    filtering::flood_fill_range_boundary_u16_c3_in_place,
    filtering::flood_fill_gradient_u16_c3_in_place,
    filtering::flood_fill_gradient_boundary_u16_c3_in_place
);
impl_flood_fill_image!(
    u32,
    C3,
    [u32; 3],
    filtering::flood_fill_u32_c3_in_place,
    filtering::flood_fill_boundary_u32_c3_in_place,
    filtering::flood_fill_range_u32_c3_in_place,
    filtering::flood_fill_range_boundary_u32_c3_in_place,
    filtering::flood_fill_gradient_u32_c3_in_place,
    filtering::flood_fill_gradient_boundary_u32_c3_in_place
);
