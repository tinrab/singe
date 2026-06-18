use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::ImagePipeline;
use super::{
    PackedPaletteLookupTableImage, PaletteLookupTableImage, PaletteLookupTableSwapImage,
    PaletteLookupTableToImage, TrilinearLookupTableImage,
};

impl_trilinear_lookup_table_image!(u8, C4, color::lookup_table_trilinear_c4);
impl_trilinear_lookup_table_image!(u8, AC4, color::lookup_table_trilinear_ac4);

impl_palette_lookup_table_image!(u8, C1, color::lookup_table_palette_c1);
impl_palette_lookup_table_image!(u16, C1, color::lookup_table_palette_c1);
impl_packed_palette_lookup_table_image!(u8, C3, 3, color::lookup_table_palette_c3);
impl_packed_palette_lookup_table_image!(u8, C4, 4, color::lookup_table_palette_c4);
impl_packed_palette_lookup_table_image!(u8, AC4, 3, color::lookup_table_palette_ac4);
impl_packed_palette_lookup_table_image!(u16, C3, 3, color::lookup_table_palette_c3);
impl_packed_palette_lookup_table_image!(u16, C4, 4, color::lookup_table_palette_c4);
impl_packed_palette_lookup_table_image!(u16, AC4, 3, color::lookup_table_palette_ac4);
impl_palette_lookup_table_to_image!(u8, u8, C3, u8, color::lookup_table_palette_u8_to_u8_c3);
impl_palette_lookup_table_to_image!(u8, u32, C1, u32, color::lookup_table_palette_u8_to_u32_c1);
impl_palette_lookup_table_to_image!(u16, u8, C1, u8, color::lookup_table_palette_u16_to_u8_c1);
impl_palette_lookup_table_to_image!(u16, u8, C3, u8, color::lookup_table_palette_u16_to_u8_c3);
impl_palette_lookup_table_to_image!(u16, u32, C1, u32, color::lookup_table_palette_u16_to_u32_c1);
impl_palette_lookup_table_swap_image!(u8, color::lookup_table_palette_swap_c3_to_c4);
impl_palette_lookup_table_swap_image!(u16, color::lookup_table_palette_swap_c3_to_c4);
