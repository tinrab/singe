use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, color_lookup_dispatch::PackedLookupTableImage};

#[path = "color_lookup_channels_cubic_into_methods.rs"]
mod cubic_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn lookup_table_channels_into<const CHANNELS: usize>(
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
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    pub fn lookup_table_channels_in_place<const CHANNELS: usize>(
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
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_image_in_place(
            stream_context,
            image,
            values,
            levels,
        )
    }

    pub fn lookup_table_channels_linear_into<const CHANNELS: usize>(
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
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_linear_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    pub fn lookup_table_channels_linear_in_place<const CHANNELS: usize>(
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
        <Self as PackedLookupTableImage<
            T,
            L,
            CHANNELS,
        >>::lookup_table_channels_linear_image_in_place(stream_context, image, values, levels)
    }
}
