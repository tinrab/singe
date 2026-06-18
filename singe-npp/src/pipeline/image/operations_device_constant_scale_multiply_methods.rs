use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::{
    ImageBacking, ImagePipeline,
    operation_shapes::{DeviceConstantInPlaceOperation, DeviceConstantOperation},
    operation_traits::MultiplyDeviceConstantScaleImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiplyDeviceConstantScaleImage<T, L>,
{
    pub fn multiply_device_constant_scale_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    pub fn multiply_device_constant_scale_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image_in_place(
            stream_context,
            constant,
            source_destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MultiplyDeviceConstantScaleImage<T, L>,
{
    pub fn multiply_device_constant_scale(
        self,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
    ) -> Result<Self> {
        self.multiply_device_constant_scale_operation(
            constant,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image_in_place,
        )
    }

    fn multiply_device_constant_scale_operation(
        mut self,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
        operation: DeviceConstantOperation<
            T,
            L,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant,
        >,
        operation_in_place: DeviceConstantInPlaceOperation<
            T,
            L,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, constant, &mut destination_view)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
