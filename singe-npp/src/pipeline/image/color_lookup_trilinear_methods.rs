use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    CopyImage, ImageBacking, ImagePipeline, color_lookup_dispatch::TrilinearLookupTableImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: TrilinearLookupTableImage<T, L>,
{
    pub fn lookup_table_trilinear_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        values: &DeviceMemory<u32>,
        levels: [&[T]; 3],
    ) -> Result<()> {
        <Self as TrilinearLookupTableImage<T, L>>::lookup_table_trilinear_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }
}

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Workspace: ImageAllocator<u8, AC4>,
    Self: CopyImage<u8, AC4>,
{
    pub fn lookup_table_trilinear_in_place(
        mut self,
        values: &DeviceMemory<u32>,
        levels: [&[u8]; 3],
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                color::lookup_table_trilinear_ac4_in_place(
                    self.stream_context,
                    &mut image_view,
                    values,
                    levels,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<u8, AC4>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<u8, AC4>>::copy(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                color::lookup_table_trilinear_ac4_in_place(
                    self.stream_context,
                    &mut destination_view,
                    values,
                    levels,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: TrilinearLookupTableImage<T, L>,
{
    pub fn lookup_table_trilinear(
        self,
        values: &DeviceMemory<u32>,
        levels: [&[T]; 3],
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as TrilinearLookupTableImage<T, L>>::lookup_table_trilinear_image(
                self.stream_context,
                &source,
                &mut destination_view,
                values,
                levels,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
