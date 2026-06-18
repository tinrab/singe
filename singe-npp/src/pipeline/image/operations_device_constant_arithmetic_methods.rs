use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ImageBacking, ImagePipeline,
    operation_shapes::{DeviceConstantScaleInPlaceOperation, DeviceConstantScaleOperation},
    operation_traits::DeviceConstantArithmeticImage,
};

#[path = "operations_device_constant_arithmetic_into_methods.rs"]
mod into_methods;
#[path = "operations_device_constant_scale_multiply_methods.rs"]
mod scale_multiply_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: DeviceConstantArithmeticImage<T, L>,
{
    pub fn add_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image_in_place,
        )
    }

    pub fn subtract_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image_in_place,
        )
    }

    pub fn multiply_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::multiply_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::multiply_device_constant_image_in_place,
        )
    }

    pub fn divide_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::divide_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::divide_device_constant_image_in_place,
        )
    }

    fn device_constant_arithmetic(
        mut self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
        operation: DeviceConstantScaleOperation<
            T,
            L,
            <Self as DeviceConstantArithmeticImage<T, L>>::Constant,
        >,
        operation_in_place: DeviceConstantScaleInPlaceOperation<
            T,
            L,
            <Self as DeviceConstantArithmeticImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view, scale_factor)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    constant,
                    &mut destination_view,
                    scale_factor,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
