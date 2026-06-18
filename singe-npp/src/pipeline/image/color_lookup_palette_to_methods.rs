use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, ImagePipeline, Workspace},
};

use super::{ImageBacking, PaletteLookupTableToImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn lookup_table_palette_to_into<D, L>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, D, L>,
        table: &DeviceMemory<<Self as PaletteLookupTableToImage<T, D, L>>::TableValue>,
        bit_size: i32,
    ) -> Result<()>
    where
        D: Copy,
        L: ChannelLayout,
        Self: PaletteLookupTableToImage<T, D, L>,
    {
        <Self as PaletteLookupTableToImage<T, D, L>>::lookup_table_palette_to_image(
            stream_context,
            source,
            destination,
            table,
            bit_size,
        )
    }

    pub fn lookup_table_palette_to<D, L>(
        self,
        table: &DeviceMemory<<Self as PaletteLookupTableToImage<T, D, L>>::TableValue>,
        bit_size: i32,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy,
        L: ChannelLayout,
        Workspace: ImageAllocator<D, L>,
        Self: PaletteLookupTableToImage<T, D, L>,
    {
        let mut destination = self.workspace.image::<D, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PaletteLookupTableToImage<T, D, L>>::lookup_table_palette_to_image(
                self.stream_context,
                &source,
                &mut destination_view,
                table,
                bit_size,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
