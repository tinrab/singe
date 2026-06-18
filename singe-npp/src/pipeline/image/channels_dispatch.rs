use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{C1, C2, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ExtractChannelImage, ImagePipeline, InsertChannelImage, SetChannelImage};

#[path = "channels_ac4_swap_dispatch.rs"]
mod ac4_swap_dispatch;
#[path = "channels_copy_channel_dispatch.rs"]
mod copy_channel_dispatch;

macro_rules! impl_extract_channel_image {
    ($ty:ty, $layout:ty, $extract:path) => {
        impl<'a> ExtractChannelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn extract_channel_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                channel: usize,
            ) -> Result<()> {
                $extract(stream_context, source, destination, channel)
            }
        }
    };
}

macro_rules! impl_insert_channel_image {
    ($ty:ty, $layout:ty, $insert:path) => {
        impl<'a> InsertChannelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn insert_channel_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                channel: usize,
            ) -> Result<()> {
                $insert(stream_context, source, destination, channel)
            }
        }
    };
}

macro_rules! impl_set_channel_image {
    ($ty:ty, $layout:ty, $set:path) => {
        impl<'a> SetChannelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn set_channel_image(
                stream_context: &StreamContext,
                value: $ty,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                channel: usize,
            ) -> Result<()> {
                $set(stream_context, destination, value, channel)
            }
        }
    };
}

impl_extract_channel_image!(u8, C3, exchange::extract_channel_c3);
impl_extract_channel_image!(u8, C4, exchange::extract_channel_c4);
impl_extract_channel_image!(u16, C3, exchange::extract_channel_c3);
impl_extract_channel_image!(u16, C4, exchange::extract_channel_c4);
impl_extract_channel_image!(i16, C3, exchange::extract_channel_c3);
impl_extract_channel_image!(i16, C4, exchange::extract_channel_c4);
impl_extract_channel_image!(i32, C3, exchange::extract_channel_c3);
impl_extract_channel_image!(i32, C4, exchange::extract_channel_c4);
impl_extract_channel_image!(f32, C2, exchange::extract_channel_c2);
impl_extract_channel_image!(f32, C3, exchange::extract_channel_c3);
impl_extract_channel_image!(f32, C4, exchange::extract_channel_c4);

impl_insert_channel_image!(u8, C3, exchange::insert_channel_c3);
impl_insert_channel_image!(u8, C4, exchange::insert_channel_c4);
impl_insert_channel_image!(u16, C3, exchange::insert_channel_c3);
impl_insert_channel_image!(u16, C4, exchange::insert_channel_c4);
impl_insert_channel_image!(i16, C3, exchange::insert_channel_c3);
impl_insert_channel_image!(i16, C4, exchange::insert_channel_c4);
impl_insert_channel_image!(i32, C3, exchange::insert_channel_c3);
impl_insert_channel_image!(i32, C4, exchange::insert_channel_c4);
impl_insert_channel_image!(f32, C2, exchange::insert_channel_c2);
impl_insert_channel_image!(f32, C3, exchange::insert_channel_c3);
impl_insert_channel_image!(f32, C4, exchange::insert_channel_c4);

impl_set_channel_image!(u8, C3, exchange::set_channel_c3);
impl_set_channel_image!(u8, C4, exchange::set_channel_c4);
impl_set_channel_image!(u16, C3, exchange::set_channel_c3);
impl_set_channel_image!(u16, C4, exchange::set_channel_c4);
impl_set_channel_image!(i16, C3, exchange::set_channel_c3);
impl_set_channel_image!(i16, C4, exchange::set_channel_c4);
impl_set_channel_image!(i32, C3, exchange::set_channel_c3);
impl_set_channel_image!(i32, C4, exchange::set_channel_c4);
impl_set_channel_image!(f32, C3, exchange::set_channel_c3);
impl_set_channel_image!(f32, C4, exchange::set_channel_c4);
