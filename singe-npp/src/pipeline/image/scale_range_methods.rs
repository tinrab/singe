use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, scale_dispatch::ScaleRangeImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn scale_range_to_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        min: f32,
        max: f32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaleRangeImage<T, U, L>,
    {
        <Self as ScaleRangeImage<T, U, L>>::scale_range_image(
            stream_context,
            source,
            destination,
            min,
            max,
        )
    }

    pub fn scale_range_to<U>(self, min: f32, max: f32) -> Result<ImagePipeline<'a, U, L>>
    where
        U: Copy,
        Workspace: ImageAllocator<U, L>,
        Self: ScaleRangeImage<T, U, L>,
    {
        let mut destination = self.workspace.image::<U, L>(self.size())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleRangeImage<T, U, L>>::scale_range_image(
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
