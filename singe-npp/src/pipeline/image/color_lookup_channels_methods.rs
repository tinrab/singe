use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, color_lookup_dispatch::PackedLookupTableImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn lookup_table_channels<const CHANNELS: usize>(
        self,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_image(
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

    pub fn lookup_table_channels_linear<const CHANNELS: usize>(
        self,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_linear_image(
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

    pub fn lookup_table_channels_cubic<const CHANNELS: usize>(
        self,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_cubic_image(
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
