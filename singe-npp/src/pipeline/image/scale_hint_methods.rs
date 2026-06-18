use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::HintAlgorithm,
};

use super::super::{ImageBacking, ImagePipeline, scale_dispatch::ScaleHintImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn scale_hint_to_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        hint: HintAlgorithm,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaleHintImage<T, U, L>,
    {
        <Self as ScaleHintImage<T, U, L>>::scale_hint_image(
            stream_context,
            source,
            destination,
            hint,
        )
    }

    pub fn scale_hint_to<U>(self, hint: HintAlgorithm) -> Result<ImagePipeline<'a, U, L>>
    where
        U: Copy,
        Workspace: ImageAllocator<U, L>,
        Self: ScaleHintImage<T, U, L>,
    {
        let mut destination = self.workspace.image::<U, L>(self.size())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleHintImage<T, U, L>>::scale_hint_image(
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
