use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

use super::super::scale_to_u8_dispatch::ScaleToU8Image;

#[path = "scale_integer_to_u8_methods.rs"]
mod integer_methods;

impl<'a, L> ImagePipeline<'a, f32, L>
where
    L: ChannelLayout,
    Self: ScaleToU8Image<L>,
{
    pub fn scale_to_u8_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ScaleToU8Image<L>>::scale_to_u8_image(
            stream_context,
            source,
            destination,
            min,
            max,
        )
    }
}

impl<'a, L> ImagePipeline<'a, f32, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, L>,
    Self: ScaleToU8Image<L>,
{
    pub fn scale_to_u8(self, min: f32, max: f32) -> Result<ImagePipeline<'a, u8, L>> {
        let mut destination = self.workspace.image::<u8, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleToU8Image<L>>::scale_to_u8_image(
                self.stream_context,
                &source,
                &mut destination_view,
                min,
                max,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
