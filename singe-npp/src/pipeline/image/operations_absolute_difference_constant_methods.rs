use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::view::C1,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ImageBacking, ImagePipeline, operation_traits::AbsoluteDifferenceConstantImage,
};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: AbsoluteDifferenceConstantImage<T>,
{
    pub fn absolute_difference_constant(self, constant: T) -> Result<Self> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AbsoluteDifferenceConstantImage<T>>::absolute_difference_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub fn absolute_difference_device_constant(self, constant: &DeviceMemory<T>) -> Result<Self> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AbsoluteDifferenceConstantImage<T>>::absolute_difference_device_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
