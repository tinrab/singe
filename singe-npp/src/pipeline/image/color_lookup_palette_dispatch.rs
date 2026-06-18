use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C3, C4, ImageView, ImageViewMut},
};

pub trait TrilinearLookupTableImage<T, L> {
    fn lookup_table_trilinear_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        values: &DeviceMemory<u32>,
        levels: [&[T]; 3],
    ) -> Result<()>;
}

pub trait PaletteLookupTableImage<T, L> {
    fn lookup_table_palette_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        table: &DeviceMemory<T>,
        bit_size: i32,
    ) -> Result<()>;
}

pub trait PackedPaletteLookupTableImage<T, L, const CHANNELS: usize> {
    fn lookup_table_palette_channels_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        tables: &[&DeviceMemory<T>; CHANNELS],
        bit_size: i32,
    ) -> Result<()>;
}

pub trait PaletteLookupTableToImage<T, D, L> {
    type TableValue;

    fn lookup_table_palette_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, D, L>,
        table: &DeviceMemory<Self::TableValue>,
        bit_size: i32,
    ) -> Result<()>;
}

pub trait PaletteLookupTableSwapImage<T> {
    fn lookup_table_palette_swap_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageViewMut<'_, T, C4>,
        alpha: i32,
        tables: &[&DeviceMemory<T>; 3],
        bit_size: i32,
    ) -> Result<()>;
}

macro_rules! impl_trilinear_lookup_table_image {
    ($ty:ty, $layout:ty, $lookup:path) => {
        impl<'a> TrilinearLookupTableImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn lookup_table_trilinear_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                values: &DeviceMemory<u32>,
                levels: [&[$ty]; 3],
            ) -> Result<()> {
                $lookup(stream_context, source, destination, values, levels)
            }
        }
    };
}

macro_rules! impl_palette_lookup_table_image {
    ($ty:ty, $layout:ty, $lookup:path) => {
        impl<'a> PaletteLookupTableImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn lookup_table_palette_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                table: &DeviceMemory<$ty>,
                bit_size: i32,
            ) -> Result<()> {
                $lookup(stream_context, source, destination, table, bit_size)
            }
        }
    };
}

macro_rules! impl_packed_palette_lookup_table_image {
    ($ty:ty, $layout:ty, $channels:literal, $lookup:path) => {
        impl<'a> PackedPaletteLookupTableImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn lookup_table_palette_channels_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                tables: &[&DeviceMemory<$ty>; $channels],
                bit_size: i32,
            ) -> Result<()> {
                $lookup(stream_context, source, destination, tables, bit_size)
            }
        }
    };
}

#[macro_use]
#[path = "color_lookup_palette_to_dispatch_macros.rs"]
mod to_dispatch_macros;

#[path = "color_lookup_palette_dispatch_impls.rs"]
mod impls;
