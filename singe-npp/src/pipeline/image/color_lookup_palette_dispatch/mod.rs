use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C3, C4, ImageOutputView, ImageView},
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Out-of-place dispatch initializes every destination channel except native ignored AC4 alpha.
pub unsafe trait TrilinearLookupTableImage<T, L> {
    fn lookup_table_trilinear_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        values: &DeviceMemory<u32>,
        levels: [&[T]; 3],
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Out-of-place dispatch initializes every destination channel except native ignored AC4 alpha.
pub unsafe trait PaletteLookupTableImage<T, L> {
    fn lookup_table_palette_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        table: &DeviceMemory<T>,
        bit_size: i32,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Out-of-place dispatch initializes every destination channel except native ignored AC4 alpha.
pub unsafe trait PackedPaletteLookupTableImage<T, L, const CHANNELS: usize> {
    fn lookup_table_palette_channels_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        tables: &[&DeviceMemory<T>; CHANNELS],
        bit_size: i32,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Out-of-place dispatch initializes every destination channel except native ignored AC4 alpha.
pub unsafe trait PaletteLookupTableToImage<T, D, L> {
    type TableValue;

    fn lookup_table_palette_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, D, L>,
        table: &DeviceMemory<Self::TableValue>,
        bit_size: i32,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Out-of-place dispatch initializes every destination channel except native ignored AC4 alpha.
pub unsafe trait PaletteLookupTableSwapImage<T> {
    fn lookup_table_palette_swap_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageOutputView<'_, T, C4>,
        alpha: i32,
        tables: &[&DeviceMemory<T>; 3],
        bit_size: i32,
    ) -> Result<()>;
}

macro_rules! impl_trilinear_lookup_table_image {
    ($ty:ty, $layout:ty, $lookup:path) => {
        unsafe impl<'a> TrilinearLookupTableImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn lookup_table_trilinear_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                values: &DeviceMemory<u32>,
                levels: [&[$ty]; 3],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(values)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup(stream_context, source, destination, values, levels)
                })
            }
        }
    };
}

macro_rules! impl_palette_lookup_table_image {
    ($ty:ty, $layout:ty, $lookup:path) => {
        unsafe impl<'a> PaletteLookupTableImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn lookup_table_palette_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                table: &DeviceMemory<$ty>,
                bit_size: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(table)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup(stream_context, source, destination, table, bit_size)
                })
            }
        }
    };
}

macro_rules! impl_packed_palette_lookup_table_image {
    ($ty:ty, $layout:ty, $channels:literal, $lookup:path) => {
        unsafe impl<'a> PackedPaletteLookupTableImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn lookup_table_palette_channels_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                tables: &[&DeviceMemory<$ty>; $channels],
                bit_size: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup(stream_context, source, destination, tables, bit_size)
                })
            }
        }
    };
}

#[macro_use]
mod to_dispatch_macros;

mod impls;
