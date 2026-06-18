use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, Remap, geometry_dispatch::RemapImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn remap_into<M>(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &ImageView<'_, T, L>,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        M: Copy,
        Self: RemapImage<T, M, L>,
    {
        <Self as RemapImage<T, M, L>>::remap_image(
            stream_context,
            remap,
            source,
            x_map,
            y_map,
            destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn remap<M>(
        self,
        remap: Remap,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
    ) -> Result<Self>
    where
        M: Copy,
        Self: RemapImage<T, M, L>,
    {
        let mut destination = self.workspace.image::<T, L>(x_map.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as RemapImage<T, M, L>>::remap_image(
                self.stream_context,
                &remap,
                &source,
                x_map,
                y_map,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
