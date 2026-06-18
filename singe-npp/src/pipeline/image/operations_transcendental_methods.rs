use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline, operation_shapes::ScaledUnaryImageOperation,
    operation_traits::UnaryTranscendentalImage,
};

#[path = "operations_exponential_methods.rs"]
mod exponential_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnaryTranscendentalImage<T, L>,
{
    pub fn natural_logarithm_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as UnaryTranscendentalImage<T, L>>::natural_logarithm_image(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn natural_logarithm_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as UnaryTranscendentalImage<T, L>>::natural_logarithm_image_in_place(
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
    pub fn natural_logarithm(self, scale_factor: i32) -> Result<Self> {
        self.unary_transcendental(
            scale_factor,
            <Self as UnaryTranscendentalImage<T, L>>::natural_logarithm_image,
            <Self as UnaryTranscendentalImage<T, L>>::natural_logarithm_image_in_place,
        )
    }

    pub(super) fn unary_transcendental(
        mut self,
        scale_factor: i32,
        operation: ScaledUnaryImageOperation<T, L>,
        operation_in_place: for<'destination> fn(
            &StreamContext,
            &mut ImageViewMut<'destination, T, L>,
            i32,
        ) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, &mut image_view, scale_factor)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    scale_factor,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
