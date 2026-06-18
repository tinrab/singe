use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, SwapChannelsAc4Image};

macro_rules! impl_swap_channels_ac4_image {
    ($ty:ty, $swap:path) => {
        impl<'a> SwapChannelsAc4Image<$ty> for ImagePipeline<'a, $ty, AC4> {
            fn swap_channels_ac4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, AC4>,
                destination: &mut ImageViewMut<'_, $ty, AC4>,
                order: [i32; 3],
            ) -> Result<()> {
                $swap(stream_context, source, destination, order)
            }
        }
    };
}

impl_swap_channels_ac4_image!(u8, exchange::swap_channels_ac4);
impl_swap_channels_ac4_image!(u16, exchange::swap_channels_ac4);
impl_swap_channels_ac4_image!(i16, exchange::swap_channels_ac4);
impl_swap_channels_ac4_image!(i32, exchange::swap_channels_ac4);
impl_swap_channels_ac4_image!(f32, exchange::swap_channels_ac4);
