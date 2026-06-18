use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub trait PackedLookupTableImage<T, L, const CHANNELS: usize> {
    type TableValue;

    fn lookup_table_channels_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<Self::TableValue>; CHANNELS],
        levels: &[&DeviceMemory<Self::TableValue>; CHANNELS],
    ) -> Result<()>;

    fn lookup_table_channels_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<Self::TableValue>; CHANNELS],
        levels: &[&DeviceMemory<Self::TableValue>; CHANNELS],
    ) -> Result<()>;

    fn lookup_table_channels_linear_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<Self::TableValue>; CHANNELS],
        levels: &[&DeviceMemory<Self::TableValue>; CHANNELS],
    ) -> Result<()>;

    fn lookup_table_channels_linear_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<Self::TableValue>; CHANNELS],
        levels: &[&DeviceMemory<Self::TableValue>; CHANNELS],
    ) -> Result<()>;

    fn lookup_table_channels_cubic_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<Self::TableValue>; CHANNELS],
        levels: &[&DeviceMemory<Self::TableValue>; CHANNELS],
    ) -> Result<()>;

    fn lookup_table_channels_cubic_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<Self::TableValue>; CHANNELS],
        levels: &[&DeviceMemory<Self::TableValue>; CHANNELS],
    ) -> Result<()>;
}
