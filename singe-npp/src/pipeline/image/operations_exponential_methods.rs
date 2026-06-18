use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImagePipeline, operation_traits::UnaryTranscendentalImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnaryTranscendentalImage<T, L>,
{
    pub fn exponential_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as UnaryTranscendentalImage<T, L>>::exponential_image(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn exponential_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as UnaryTranscendentalImage<T, L>>::exponential_image_in_place(
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
    Self: UnaryTranscendentalImage<T, L>,
{
    pub fn exponential(self, scale_factor: i32) -> Result<Self> {
        self.unary_transcendental(
            scale_factor,
            <Self as UnaryTranscendentalImage<T, L>>::exponential_image,
            <Self as UnaryTranscendentalImage<T, L>>::exponential_image_in_place,
        )
    }
}
