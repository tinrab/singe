use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline,
    operation_shapes::{ConstantScaleInPlaceOperation, ConstantScaleOperation},
    operation_traits::ConstantArithmeticImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ConstantArithmeticImage<T, L>,
    <Self as ConstantArithmeticImage<T, L>>::Constant: Copy,
{
    pub fn add_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::add_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::add_constant_image_in_place,
        )
    }

    pub fn subtract_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::subtract_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::subtract_constant_image_in_place,
        )
    }

    pub fn multiply_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::multiply_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::multiply_constant_image_in_place,
        )
    }

    pub fn divide_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::divide_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::divide_constant_image_in_place,
        )
    }

    fn constant_arithmetic(
        mut self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
        operation: ConstantScaleOperation<T, L, <Self as ConstantArithmeticImage<T, L>>::Constant>,
        operation_in_place: ConstantScaleInPlaceOperation<
            T,
            L,
            <Self as ConstantArithmeticImage<T, L>>::Constant,
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
