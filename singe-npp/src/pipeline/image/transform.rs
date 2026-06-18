use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Axis,
};

use super::{ImageBacking, ImagePipeline};

#[path = "transform_mirror_dispatch.rs"]
mod mirror_dispatch;

pub trait MirrorImage<T, L> {
    fn mirror_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        axis: Axis,
    ) -> Result<()>;

    fn mirror_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        axis: Axis,
    ) -> Result<()>;
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MirrorImage<T, L>,
{
    pub fn mirror_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        axis: Axis,
    ) -> Result<()> {
        <Self as MirrorImage<T, L>>::mirror_image(stream_context, source, destination, axis)
    }

    pub fn mirror_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        axis: Axis,
    ) -> Result<()> {
        <Self as MirrorImage<T, L>>::mirror_image_in_place(stream_context, image, axis)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MirrorImage<T, L>,
{
    pub fn mirror(mut self, axis: Axis) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as MirrorImage<T, L>>::mirror_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    axis,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as MirrorImage<T, L>>::mirror_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    axis,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
