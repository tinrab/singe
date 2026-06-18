use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::{super::ImagePipeline, LookupTableImage};

impl_lookup_table_image!(
    u8,
    i32,
    color::lookup_table_c1,
    color::lookup_table_c1_in_place,
    color::lookup_table_linear_u8_c1,
    color::lookup_table_linear_u8_c1_in_place,
    color::lookup_table_cubic_u8_c1,
    color::lookup_table_cubic_u8_c1_in_place
);
impl_lookup_table_image!(
    u16,
    i32,
    color::lookup_table_c1,
    color::lookup_table_c1_in_place,
    color::lookup_table_linear_u16_c1,
    color::lookup_table_linear_u16_c1_in_place,
    color::lookup_table_cubic_u16_c1,
    color::lookup_table_cubic_u16_c1_in_place
);
impl_lookup_table_image!(
    i16,
    i32,
    color::lookup_table_c1,
    color::lookup_table_c1_in_place,
    color::lookup_table_linear_i16_c1,
    color::lookup_table_linear_i16_c1_in_place,
    color::lookup_table_cubic_i16_c1,
    color::lookup_table_cubic_i16_c1_in_place
);
impl_lookup_table_image!(
    f32,
    f32,
    color::lookup_table_c1,
    color::lookup_table_c1_in_place,
    color::lookup_table_linear_f32_c1,
    color::lookup_table_linear_f32_c1_in_place,
    color::lookup_table_cubic_f32_c1,
    color::lookup_table_cubic_f32_c1_in_place
);
