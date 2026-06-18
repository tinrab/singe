use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, C4, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ImageBacking, ImagePipeline, color_lookup_dispatch::PaletteLookupTableSwapImage,
};

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: PaletteLookupTableSwapImage<T>,
{
    pub fn lookup_table_palette_swap_to_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageViewMut<'_, T, C4>,
        alpha: i32,
        tables: &[&DeviceMemory<T>; 3],
        bit_size: i32,
    ) -> Result<()> {
        <Self as PaletteLookupTableSwapImage<T>>::lookup_table_palette_swap_image(
            stream_context,
            source,
            destination,
            alpha,
            tables,
            bit_size,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Workspace: ImageAllocator<T, C4>,
    Self: PaletteLookupTableSwapImage<T>,
{
    pub fn lookup_table_palette_swap_to_c4(
        self,
        alpha: i32,
        tables: &[&DeviceMemory<T>; 3],
        bit_size: i32,
    ) -> Result<ImagePipeline<'a, T, C4>> {
        let mut destination = self.workspace.image::<T, C4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PaletteLookupTableSwapImage<T>>::lookup_table_palette_swap_image(
                self.stream_context,
                &source,
                &mut destination_view,
                alpha,
                tables,
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
