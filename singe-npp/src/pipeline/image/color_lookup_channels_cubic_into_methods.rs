use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, color_lookup_dispatch::PackedLookupTableImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn lookup_table_channels_cubic_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<()>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_cubic_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    pub fn lookup_table_channels_cubic_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<()>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_cubic_image_in_place(
            stream_context,
            image,
            values,
            levels,
        )
    }
}
