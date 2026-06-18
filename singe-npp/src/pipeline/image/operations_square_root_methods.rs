use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::{ImagePipeline, operation_traits::SquareRootImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareRootImage<T, L>,
{
    pub fn square_root_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareRootImage<T, L>>::square_root_image(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn square_root_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareRootImage<T, L>>::square_root_image_in_place(
            stream_context,
            source_destination,
            scale_factor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SquareRootImage<T, L>,
{
    pub fn square_root(self, scale_factor: i32) -> Result<Self> {
        super::unary_power_image(
            self,
            scale_factor,
            <Self as SquareRootImage<T, L>>::square_root_image,
            <Self as SquareRootImage<T, L>>::square_root_image_in_place,
        )
    }
}
