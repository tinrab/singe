use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::ImagePipeline;
use super::convert_dispatch::ConvertImage;

impl<'a, L> ImagePipeline<'a, i16, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_u8(self) -> Result<ImagePipeline<'a, u8, L>>
    where
        Workspace: ImageAllocator<u8, L>,
        Self: ConvertImage<i16, u8, L>,
    {
        self.convert_to::<u8>()
    }

    pub fn convert_to_u16(self) -> Result<ImagePipeline<'a, u16, L>>
    where
        Workspace: ImageAllocator<u16, L>,
        Self: ConvertImage<i16, u16, L>,
    {
        self.convert_to::<u16>()
    }

    pub fn convert_to_i32(self) -> Result<ImagePipeline<'a, i32, L>>
    where
        Workspace: ImageAllocator<i32, L>,
        Self: ConvertImage<i16, i32, L>,
    {
        self.convert_to::<i32>()
    }

    pub fn convert_to_u32(self) -> Result<ImagePipeline<'a, u32, L>>
    where
        Workspace: ImageAllocator<u32, L>,
        Self: ConvertImage<i16, u32, L>,
    {
        self.convert_to::<u32>()
    }

    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<i16, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, i32, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_u8(self) -> Result<ImagePipeline<'a, u8, L>>
    where
        Workspace: ImageAllocator<u8, L>,
        Self: ConvertImage<i32, u8, L>,
    {
        self.convert_to::<u8>()
    }

    pub fn convert_to_u32(self) -> Result<ImagePipeline<'a, u32, L>>
    where
        Workspace: ImageAllocator<u32, L>,
        Self: ConvertImage<i32, u32, L>,
    {
        self.convert_to::<u32>()
    }

    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<i32, f32, L>,
    {
        self.convert_to::<f32>()
    }
}
