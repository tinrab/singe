use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::scale_dispatch::ScaleToI16Image;
use super::super::{ImageBacking, ImagePipeline};

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: ScaleToI16Image<L>,
{
    pub fn scale_to_i16_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, i16, L>,
    ) -> Result<()> {
        <Self as ScaleToI16Image<L>>::scale_to_i16_image(stream_context, source, destination)
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<i16, L>,
    Self: ScaleToI16Image<L>,
{
    pub fn scale_to_i16(self) -> Result<ImagePipeline<'a, i16, L>> {
        let mut destination = self.workspace.image::<i16, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleToI16Image<L>>::scale_to_i16_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
