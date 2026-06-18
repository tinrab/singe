use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline, operation_shapes::ConstantShiftImageOperation, operation_traits::*,
};

#[path = "operations_logical_not_methods.rs"]
mod logical_not_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: RightShiftConstantImage<T, L>,
    <Self as RightShiftConstantImage<T, L>>::Constant: Copy,
{
    pub fn right_shift_constant(
        self,
        constant: <Self as RightShiftConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        shift_constant_image(
            self,
            constant,
            <Self as RightShiftConstantImage<T, L>>::right_shift_constant_image,
            <Self as RightShiftConstantImage<T, L>>::right_shift_constant_image_in_place,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: LeftShiftConstantImage<T, L>,
    <Self as LeftShiftConstantImage<T, L>>::Constant: Copy,
{
    pub fn left_shift_constant(
        self,
        constant: <Self as LeftShiftConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        shift_constant_image(
            self,
            constant,
            <Self as LeftShiftConstantImage<T, L>>::left_shift_constant_image,
            <Self as LeftShiftConstantImage<T, L>>::left_shift_constant_image_in_place,
        )
    }
}

fn shift_constant_image<'a, T, L, C>(
    mut pipeline: ImagePipeline<'a, T, L>,
    constant: C,
    operation: ConstantShiftImageOperation<T, L, C>,
    operation_in_place: for<'destination> fn(
        &StreamContext,
        C,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>,
) -> Result<ImagePipeline<'a, T, L>>
where
    T: Copy,
    L: ChannelLayout,
    C: Copy,
    Workspace: ImageAllocator<T, L>,
{
    match &mut pipeline.backing {
        ImageBacking::Owned(image) => {
            let mut image_view = image.view_mut()?;
            operation_in_place(pipeline.stream_context, constant, &mut image_view)?;
        }
        ImageBacking::Borrowed(source) => {
            let mut destination = pipeline.workspace.image::<T, L>(source.size())?;
            let mut destination_view = destination.view_mut()?;
            operation(
                pipeline.stream_context,
                source,
                constant,
                &mut destination_view,
            )?;
            pipeline.backing = ImageBacking::Owned(destination);
        }
    }

    Ok(pipeline)
}
