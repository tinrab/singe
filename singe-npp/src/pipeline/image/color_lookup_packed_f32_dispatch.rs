use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, PackedLookupTableImage};

impl_packed_lookup_table_image!(
    f32,
    C3,
    3,
    f32,
    color::lookup_table_c3,
    color::lookup_table_c3_in_place,
    color::lookup_table_linear_f32_c3,
    color::lookup_table_linear_f32_c3_in_place,
    color::lookup_table_cubic_f32_c3,
    color::lookup_table_cubic_f32_c3_in_place
);
impl_packed_lookup_table_image!(
    f32,
    C4,
    4,
    f32,
    color::lookup_table_c4,
    color::lookup_table_c4_in_place,
    color::lookup_table_linear_f32_c4,
    color::lookup_table_linear_f32_c4_in_place,
    color::lookup_table_cubic_f32_c4,
    color::lookup_table_cubic_f32_c4_in_place
);
impl_packed_lookup_table_image!(
    f32,
    AC4,
    3,
    f32,
    color::lookup_table_ac4,
    color::lookup_table_ac4_in_place,
    color::lookup_table_linear_f32_ac4,
    color::lookup_table_linear_f32_ac4_in_place,
    color::lookup_table_cubic_f32_ac4,
    color::lookup_table_cubic_f32_ac4_in_place
);
