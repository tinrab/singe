use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

use super::super::scale_dispatch::ScaleToF32Image;

#[path = "scale_to_i16_methods.rs"]
mod scale_to_i16_methods;
#[path = "scale_to_integer_methods.rs"]
mod scale_to_integer_methods;

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: ScaleToF32Image<L>,
{
    pub fn scale_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ScaleToF32Image<L>>::scale_to_f32_image(
            stream_context,
            source,
            destination,
            min,
            max,
        )
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<f32, L>,
    Self: ScaleToF32Image<L>,
{
    pub fn scale_to_f32(self, min: f32, max: f32) -> Result<ImagePipeline<'a, f32, L>> {
        let mut destination = self.workspace.image::<f32, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleToF32Image<L>>::scale_to_f32_image(
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
