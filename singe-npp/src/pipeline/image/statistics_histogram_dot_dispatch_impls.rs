use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{AC4, C1, C3, C4, ImageView},
    },
    pipeline::image::ImagePipeline,
};

use super::{
    DotProductImage, HistogramEvenChannelsImage, HistogramEvenImage, HistogramRangeChannelsImage,
    HistogramRangeImage,
};

impl_histogram_even_image!(u8, C1, statistics::histogram_even_u8_c1);
impl_histogram_even_image!(u16, C1, statistics::histogram_even_u16_c1);
impl_histogram_even_image!(i16, C1, statistics::histogram_even_i16_c1);

impl_histogram_even_channels_image!(u8, C3, 3, statistics::histogram_even_u8_c3);
impl_histogram_even_channels_image!(u16, C3, 3, statistics::histogram_even_u16_c3);
impl_histogram_even_channels_image!(i16, C3, 3, statistics::histogram_even_i16_c3);
impl_histogram_even_channels_image!(u8, C4, 4, statistics::histogram_even_u8_c4);
impl_histogram_even_channels_image!(u16, C4, 4, statistics::histogram_even_u16_c4);
impl_histogram_even_channels_image!(i16, C4, 4, statistics::histogram_even_i16_c4);
impl_histogram_even_channels_image!(u8, AC4, 3, statistics::histogram_even_u8_ac4);
impl_histogram_even_channels_image!(u16, AC4, 3, statistics::histogram_even_u16_ac4);
impl_histogram_even_channels_image!(i16, AC4, 3, statistics::histogram_even_i16_ac4);

impl_histogram_range_image!(u8, i32, C1, statistics::histogram_range_u8_c1);
impl_histogram_range_image!(u16, i32, C1, statistics::histogram_range_u16_c1);
impl_histogram_range_image!(i16, i32, C1, statistics::histogram_range_i16_c1);
impl_histogram_range_image!(f32, f32, C1, statistics::histogram_range_f32_c1);
impl_histogram_range_channels_image!(u8, i32, C3, 3, statistics::histogram_range_u8_c3);
impl_histogram_range_channels_image!(u16, i32, C3, 3, statistics::histogram_range_u16_c3);
impl_histogram_range_channels_image!(i16, i32, C3, 3, statistics::histogram_range_i16_c3);
impl_histogram_range_channels_image!(f32, f32, C3, 3, statistics::histogram_range_f32_c3);
impl_histogram_range_channels_image!(u8, i32, C4, 4, statistics::histogram_range_u8_c4);
impl_histogram_range_channels_image!(u16, i32, C4, 4, statistics::histogram_range_u16_c4);
impl_histogram_range_channels_image!(i16, i32, C4, 4, statistics::histogram_range_i16_c4);
impl_histogram_range_channels_image!(f32, f32, C4, 4, statistics::histogram_range_f32_c4);
impl_histogram_range_channels_image!(u8, i32, AC4, 3, statistics::histogram_range_u8_ac4);
impl_histogram_range_channels_image!(u16, i32, AC4, 3, statistics::histogram_range_u16_ac4);
impl_histogram_range_channels_image!(i16, i32, AC4, 3, statistics::histogram_range_i16_ac4);
impl_histogram_range_channels_image!(f32, f32, AC4, 3, statistics::histogram_range_f32_ac4);

impl_dot_product_image!(u8, C1, 1, statistics::dot_prod_u8_c1);
impl_dot_product_image!(i8, C1, 1, statistics::dot_prod_i8_c1);
impl_dot_product_image!(u16, C1, 1, statistics::dot_prod_u16_c1);
impl_dot_product_image!(i16, C1, 1, statistics::dot_prod_i16_c1);
impl_dot_product_image!(u32, C1, 1, statistics::dot_prod_u32_c1);
impl_dot_product_image!(i32, C1, 1, statistics::dot_prod_i32_c1);
impl_dot_product_image!(f32, C1, 1, statistics::dot_prod_f32_c1);
impl_dot_product_image!(u8, C3, 3, statistics::dot_prod_u8_c3);
impl_dot_product_image!(i8, C3, 3, statistics::dot_prod_i8_c3);
impl_dot_product_image!(u16, C3, 3, statistics::dot_prod_u16_c3);
impl_dot_product_image!(i16, C3, 3, statistics::dot_prod_i16_c3);
impl_dot_product_image!(u32, C3, 3, statistics::dot_prod_u32_c3);
impl_dot_product_image!(i32, C3, 3, statistics::dot_prod_i32_c3);
impl_dot_product_image!(f32, C3, 3, statistics::dot_prod_f32_c3);
impl_dot_product_image!(u8, C4, 4, statistics::dot_prod_u8_c4);
impl_dot_product_image!(i8, C4, 4, statistics::dot_prod_i8_c4);
impl_dot_product_image!(u16, C4, 4, statistics::dot_prod_u16_c4);
impl_dot_product_image!(i16, C4, 4, statistics::dot_prod_i16_c4);
impl_dot_product_image!(u32, C4, 4, statistics::dot_prod_u32_c4);
impl_dot_product_image!(i32, C4, 4, statistics::dot_prod_i32_c4);
impl_dot_product_image!(f32, C4, 4, statistics::dot_prod_f32_c4);
impl_dot_product_image!(u8, AC4, 3, statistics::dot_prod_u8_ac4);
impl_dot_product_image!(i8, AC4, 3, statistics::dot_prod_i8_ac4);
impl_dot_product_image!(u16, AC4, 3, statistics::dot_prod_u16_ac4);
impl_dot_product_image!(i16, AC4, 3, statistics::dot_prod_i16_ac4);
impl_dot_product_image!(u32, AC4, 3, statistics::dot_prod_u32_ac4);
impl_dot_product_image!(i32, AC4, 3, statistics::dot_prod_i32_ac4);
impl_dot_product_image!(f32, AC4, 3, statistics::dot_prod_f32_ac4);
