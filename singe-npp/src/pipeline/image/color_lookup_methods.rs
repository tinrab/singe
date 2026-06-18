use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::view::C1,
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, color_lookup_dispatch::LookupTableImage};

#[path = "color_lookup_into_methods.rs"]
mod into_methods;

#[path = "color_lookup_channels_into_methods.rs"]
mod channels_into_methods;

#[path = "color_lookup_channels_methods.rs"]
mod channels_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: LookupTableImage<T>,
{
    pub fn lookup_table(
        self,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as LookupTableImage<T>>::lookup_table_image(
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

    pub fn lookup_table_linear(
        self,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as LookupTableImage<T>>::lookup_table_linear_image(
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

    pub fn lookup_table_cubic(
        self,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as LookupTableImage<T>>::lookup_table_cubic_image(
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
