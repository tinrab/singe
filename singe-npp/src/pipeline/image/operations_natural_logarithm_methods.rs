use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, NaturalLogarithmImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: NaturalLogarithmImage<T, L>,
{
    pub fn natural_logarithm_unscaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as NaturalLogarithmImage<T, L>>::natural_logarithm_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn natural_logarithm_unscaled_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as NaturalLogarithmImage<T, L>>::natural_logarithm_image_in_place(
            stream_context,
            source_destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: NaturalLogarithmImage<T, L>,
{
    pub fn natural_logarithm_unscaled(mut self) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as NaturalLogarithmImage<T, L>>::natural_logarithm_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as NaturalLogarithmImage<T, L>>::natural_logarithm_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
