use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{ConstantBorderImage, ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ConstantBorderImage<T, L>,
    <Self as ConstantBorderImage<T, L>>::Value: Copy,
{
    pub fn constant_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        top: usize,
        left: usize,
        value: <Self as ConstantBorderImage<T, L>>::Value,
    ) -> Result<()> {
        <Self as ConstantBorderImage<T, L>>::copy_constant_border_image(
            stream_context,
            source,
            destination,
            top,
            left,
            value,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ConstantBorderImage<T, L>,
    <Self as ConstantBorderImage<T, L>>::Value: Copy,
{
    pub fn constant_border(
        self,
        size: Size,
        top: usize,
        left: usize,
        value: <Self as ConstantBorderImage<T, L>>::Value,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ConstantBorderImage<T, L>>::copy_constant_border_image(
                self.stream_context,
                &source,
                &mut destination_view,
                top,
                left,
                value,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
