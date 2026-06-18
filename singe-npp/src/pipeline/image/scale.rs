use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, scale_dispatch::ScaleImage};

#[path = "scale_conversion_methods.rs"]
mod conversion_methods;
#[path = "scale_hint_methods.rs"]
mod hint_methods;
#[path = "scale_range_methods.rs"]
mod range_methods;
#[path = "scale_to_u8_methods.rs"]
mod to_u8_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn scale_to_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaleImage<T, U, L>,
    {
        <Self as ScaleImage<T, U, L>>::scale_image(stream_context, source, destination)
    }

    pub fn scale_to<U>(self) -> Result<ImagePipeline<'a, U, L>>
    where
        U: Copy,
        Workspace: ImageAllocator<U, L>,
        Self: ScaleImage<T, U, L>,
    {
        let mut destination = self.workspace.image::<U, L>(self.size())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleImage<T, U, L>>::scale_image(
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
