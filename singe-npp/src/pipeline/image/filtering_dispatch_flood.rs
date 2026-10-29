use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, filtering_traits::FloodFillImage},
    types::{ConnectedRegion, ImageNormalization, Point},
};

#[macro_use]
mod macros;

mod c3;

impl_flood_fill_image!(
    u8,
    C1,
    u8,
    filtering::flood_fill_u8_c1_in_place,
    filtering::flood_fill_boundary_u8_c1_in_place,
    filtering::flood_fill_range_u8_c1_in_place,
    filtering::flood_fill_range_boundary_u8_c1_in_place,
    filtering::flood_fill_gradient_u8_c1_in_place,
    filtering::flood_fill_gradient_boundary_u8_c1_in_place
);

impl_flood_fill_image!(
    u16,
    C1,
    u16,
    filtering::flood_fill_u16_c1_in_place,
    filtering::flood_fill_boundary_u16_c1_in_place,
    filtering::flood_fill_range_u16_c1_in_place,
    filtering::flood_fill_range_boundary_u16_c1_in_place,
    filtering::flood_fill_gradient_u16_c1_in_place,
    filtering::flood_fill_gradient_boundary_u16_c1_in_place
);

impl_flood_fill_image!(
    u32,
    C1,
    u32,
    filtering::flood_fill_u32_c1_in_place,
    filtering::flood_fill_boundary_u32_c1_in_place,
    filtering::flood_fill_range_u32_c1_in_place,
    filtering::flood_fill_range_boundary_u32_c1_in_place,
    filtering::flood_fill_gradient_u32_c1_in_place,
    filtering::flood_fill_gradient_boundary_u32_c1_in_place
);
