use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::DataTypeLike,
};

use super::{ImageBacking, ImagePipeline};

#[path = "convert_dispatch.rs"]
mod convert_dispatch;
#[path = "convert_extended_methods.rs"]
mod extended_methods;
#[path = "convert_signed_methods.rs"]
mod signed_methods;

use convert_dispatch::ConvertImage;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: DataTypeLike,
    L: ChannelLayout,
{
    pub fn convert_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
    ) -> Result<()>
    where
        U: DataTypeLike,
        Self: ConvertImage<T, U, L>,
    {
        <Self as ConvertImage<T, U, L>>::convert_image(stream_context, source, destination)
    }

    pub fn convert_to<U>(self) -> Result<ImagePipeline<'a, U, L>>
    where
        U: DataTypeLike,
        Workspace: ImageAllocator<U, L>,
        Self: ConvertImage<T, U, L>,
    {
        let mut destination = self.workspace.image::<U, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ConvertImage<T, U, L>>::convert_image(
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

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_u16(self) -> Result<ImagePipeline<'a, u16, L>>
    where
        Workspace: ImageAllocator<u16, L>,
        Self: ConvertImage<u8, u16, L>,
    {
        self.convert_to::<u16>()
    }

    pub fn convert_to_i16(self) -> Result<ImagePipeline<'a, i16, L>>
    where
        Workspace: ImageAllocator<i16, L>,
        Self: ConvertImage<u8, i16, L>,
    {
        self.convert_to::<i16>()
    }

    pub fn convert_to_i32(self) -> Result<ImagePipeline<'a, i32, L>>
    where
        Workspace: ImageAllocator<i32, L>,
        Self: ConvertImage<u8, i32, L>,
    {
        self.convert_to::<i32>()
    }

    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<u8, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, u16, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_u8(self) -> Result<ImagePipeline<'a, u8, L>>
    where
        Workspace: ImageAllocator<u8, L>,
        Self: ConvertImage<u16, u8, L>,
    {
        self.convert_to::<u8>()
    }

    pub fn convert_to_i32(self) -> Result<ImagePipeline<'a, i32, L>>
    where
        Workspace: ImageAllocator<i32, L>,
        Self: ConvertImage<u16, i32, L>,
    {
        self.convert_to::<i32>()
    }

    pub fn convert_to_u32(self) -> Result<ImagePipeline<'a, u32, L>>
    where
        Workspace: ImageAllocator<u32, L>,
        Self: ConvertImage<u16, u32, L>,
    {
        self.convert_to::<u32>()
    }

    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<u16, f32, L>,
    {
        self.convert_to::<f32>()
    }
}
