use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{ImageBacking, ImagePipeline};

#[path = "border_constant_methods.rs"]
mod constant_methods;
#[path = "border_dispatch.rs"]
mod dispatch;

pub(super) use dispatch::{BorderImage, ConstantBorderImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BorderImage<T, L>,
{
    pub fn replicate_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        top: usize,
        left: usize,
    ) -> Result<()> {
        <Self as BorderImage<T, L>>::copy_replicate_border_image(
            stream_context,
            source,
            destination,
            top,
            left,
        )
    }

    pub fn wrap_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        top: usize,
        left: usize,
    ) -> Result<()> {
        <Self as BorderImage<T, L>>::copy_wrap_border_image(
            stream_context,
            source,
            destination,
            top,
            left,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: BorderImage<T, L>,
{
    pub fn replicate_border(self, size: Size, top: usize, left: usize) -> Result<Self> {
        self.border(
            size,
            top,
            left,
            <Self as BorderImage<T, L>>::copy_replicate_border_image,
        )
    }

    pub fn wrap_border(self, size: Size, top: usize, left: usize) -> Result<Self> {
        self.border(
            size,
            top,
            left,
            <Self as BorderImage<T, L>>::copy_wrap_border_image,
        )
    }

    fn border(
        self,
        size: Size,
        top: usize,
        left: usize,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            usize,
            usize,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                &mut destination_view,
                top,
                left,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
