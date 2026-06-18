use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::super::{ImagePipeline, operation_traits::UnscaledBinaryArithmeticImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    pub fn multiply_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::multiply_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn divide_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::divide_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    pub fn multiply_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::multiply_unscaled_image,
        )
    }

    pub fn divide_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::divide_unscaled_image,
        )
    }
}
