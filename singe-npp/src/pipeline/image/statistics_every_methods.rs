use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImageBacking, ImagePipeline, statistics::EveryStatisticImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + EveryStatisticImage<T, L>,
{
    pub fn max_every_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EveryStatisticImage<T, L>>::max_every(stream_context, source, source_destination)
    }

    pub fn min_every_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EveryStatisticImage<T, L>>::min_every(stream_context, source, source_destination)
    }

    pub fn max_every(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.every_statistic(other, <Self as EveryStatisticImage<T, L>>::max_every)
    }

    pub fn min_every(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.every_statistic(other, <Self as EveryStatisticImage<T, L>>::min_every)
    }

    fn every_statistic(
        mut self,
        other: &ImageView<'_, T, L>,
        statistic: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
        ) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                statistic(self.stream_context, other, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                {
                    let mut destination_view = destination.view_mut()?;
                    <Self as CopyImage<T, L>>::copy(
                        self.stream_context,
                        source,
                        &mut destination_view,
                    )?;
                    statistic(self.stream_context, other, &mut destination_view)?;
                }
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
