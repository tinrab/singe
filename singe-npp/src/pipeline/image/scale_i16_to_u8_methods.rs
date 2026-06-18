use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::HintAlgorithm,
};

use super::super::super::super::{
    ImageBacking, ImagePipeline, scale_to_u8_dispatch::ScaleI16ToU8Image,
};

impl<'a, L> ImagePipeline<'a, i16, L>
where
    L: ChannelLayout,
    Self: ScaleI16ToU8Image<L>,
{
    pub fn scale_to_u8_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, i16, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()> {
        <Self as ScaleI16ToU8Image<L>>::scale_i16_to_u8_image(
            stream_context,
            source,
            destination,
            hint,
        )
    }
}

impl<'a, L> ImagePipeline<'a, i16, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, L>,
    Self: ScaleI16ToU8Image<L>,
{
    pub fn scale_to_u8(self, hint: HintAlgorithm) -> Result<ImagePipeline<'a, u8, L>> {
        let mut destination = self.workspace.image::<u8, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleI16ToU8Image<L>>::scale_i16_to_u8_image(
                self.stream_context,
                &source,
                &mut destination_view,
                hint,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
