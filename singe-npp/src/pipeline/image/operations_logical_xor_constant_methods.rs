use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImagePipeline, LogicalConstantImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: LogicalConstantImage<T, L>,
    <Self as LogicalConstantImage<T, L>>::Constant: Copy,
{
    pub fn logical_xor_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_xor_constant_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    pub fn logical_xor_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_xor_constant_image_in_place(
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
    Self: LogicalConstantImage<T, L>,
    <Self as LogicalConstantImage<T, L>>::Constant: Copy,
{
    pub fn logical_xor_constant(
        self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        self.logical_constant(
            constant,
            <Self as LogicalConstantImage<T, L>>::logical_xor_constant_image,
            <Self as LogicalConstantImage<T, L>>::logical_xor_constant_image_in_place,
        )
    }
}
