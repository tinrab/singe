use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline,
    color_lookup_dispatch::{
        PackedPaletteLookupTableImage, PaletteLookupTableImage, PaletteLookupTableToImage,
    },
};

#[path = "color_lookup_palette_swap_methods.rs"]
mod palette_swap_methods;
#[path = "color_lookup_palette_to_methods.rs"]
mod palette_to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: PaletteLookupTableImage<T, L>,
{
    pub fn lookup_table_palette_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        table: &DeviceMemory<T>,
        bit_size: i32,
    ) -> Result<()> {
        <Self as PaletteLookupTableImage<T, L>>::lookup_table_palette_image(
            stream_context,
            source,
            destination,
            table,
            bit_size,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: PaletteLookupTableImage<T, L>,
{
    pub fn lookup_table_palette(self, table: &DeviceMemory<T>, bit_size: i32) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PaletteLookupTableImage<T, L>>::lookup_table_palette_image(
                self.stream_context,
                &source,
                &mut destination_view,
                table,
                bit_size,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn lookup_table_palette_channels_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        tables: &[&DeviceMemory<T>; CHANNELS],
        bit_size: i32,
    ) -> Result<()>
    where
        Self: PackedPaletteLookupTableImage<T, L, CHANNELS>,
    {
        <Self as PackedPaletteLookupTableImage<T, L, CHANNELS>>::lookup_table_palette_channels_image(
            stream_context,
            source,
            destination,
            tables,
            bit_size,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn lookup_table_palette_channels<const CHANNELS: usize>(
        self,
        tables: &[&DeviceMemory<T>; CHANNELS],
        bit_size: i32,
    ) -> Result<Self>
    where
        Self: PackedPaletteLookupTableImage<T, L, CHANNELS>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PackedPaletteLookupTableImage<T, L, CHANNELS>>::lookup_table_palette_channels_image(
                self.stream_context,
                &source,
                &mut destination_view,
                tables,
                bit_size,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
