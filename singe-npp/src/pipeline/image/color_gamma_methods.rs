use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::color::{GammaImage, SameLayoutColorOperation};
use super::{ImageBacking, ImagePipeline};

#[path = "color_planar_gamma_methods.rs"]
mod planar_methods;

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: GammaImage<L>,
{
    pub fn gamma_forward_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_forward_image(stream_context, source, destination)
    }

    pub fn gamma_forward_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_forward_image_in_place(stream_context, image)
    }

    pub fn gamma_inverse_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_inverse_image(stream_context, source, destination)
    }

    pub fn gamma_inverse_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_inverse_image_in_place(stream_context, image)
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, L>,
    Self: GammaImage<L>,
{
    pub fn gamma_forward(self) -> Result<Self> {
        self.gamma_with(
            <Self as GammaImage<L>>::gamma_forward_image,
            <Self as GammaImage<L>>::gamma_forward_image_in_place,
        )
    }

    pub fn gamma_inverse(self) -> Result<Self> {
        self.gamma_with(
            <Self as GammaImage<L>>::gamma_inverse_image,
            <Self as GammaImage<L>>::gamma_inverse_image_in_place,
        )
    }

    fn gamma_with(
        mut self,
        operation: SameLayoutColorOperation<L>,
        operation_in_place: for<'destination> fn(
            &StreamContext,
            &mut ImageViewMut<'destination, u8, L>,
        ) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<u8, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, &mut destination_view)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
