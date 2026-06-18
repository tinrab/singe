use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        memory::Image,
        view::{C1, ChannelLayout, ImageView},
    },
    pipeline::{ImageAllocator, Workspace, memory_context::validate_device_memory_stream_context},
    types::Size,
};

/// Current storage behind a fluent image pipeline.
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageBacking<'a, T, L = C1> {
    Borrowed(ImageView<'a, T, L>),
    Owned(Image<T, L>),
}

/// Eager, stream-aware image processing pipeline.
pub struct ImagePipeline<'a, T, L = C1> {
    pub(super) stream_context: &'a StreamContext,
    pub(super) workspace: &'a mut Workspace,
    pub(super) backing: ImageBacking<'a, T, L>,
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn create(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        size: Size,
    ) -> Result<Self>
    where
        Workspace: ImageAllocator<T, L>,
    {
        let image = workspace.image::<T, L>(size)?;
        Ok(Self::from_owned(stream_context, workspace, image))
    }

    pub fn from_view(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        source: ImageView<'a, T, L>,
    ) -> Self {
        Self {
            stream_context,
            workspace,
            backing: ImageBacking::Borrowed(source),
        }
    }

    pub fn from_owned(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        image: Image<T, L>,
    ) -> Self {
        Self {
            stream_context,
            workspace,
            backing: ImageBacking::Owned(image),
        }
    }

    pub fn from_memory(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        memory: &'a DeviceMemory<T>,
        size: Size,
    ) -> Result<Self> {
        validate_device_memory_stream_context(stream_context, memory)?;
        Ok(Self::from_view(
            stream_context,
            workspace,
            ImageView::from_memory(memory, size)?,
        ))
    }

    pub fn from_memory_with_step(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        memory: &'a DeviceMemory<T>,
        size: Size,
        step: i32,
    ) -> Result<Self> {
        validate_device_memory_stream_context(stream_context, memory)?;
        Ok(Self::from_view(
            stream_context,
            workspace,
            ImageView::from_memory_with_step(memory, size, step)?,
        ))
    }

    pub const fn size(&self) -> Size {
        match &self.backing {
            ImageBacking::Borrowed(view) => view.size(),
            ImageBacking::Owned(image) => image.size(),
        }
    }

    pub const fn is_owned(&self) -> bool {
        matches!(self.backing, ImageBacking::Owned(_))
    }

    pub(super) fn view(&self) -> Result<ImageView<'_, T, L>> {
        match &self.backing {
            ImageBacking::Borrowed(view) => Ok(*view),
            ImageBacking::Owned(image) => image.view(),
        }
    }

    pub fn into_backing(self) -> ImageBacking<'a, T, L> {
        self.backing
    }
}
