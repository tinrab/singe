use singe_cuda::types::f16;

use crate::{
    error::Result,
    image::view::{C1, ChannelLayout},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ConvertImage, ImagePipeline};

impl<'a, L> ImagePipeline<'a, u32, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<u32, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, i32, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_i8(self) -> Result<ImagePipeline<'a, i8, L>>
    where
        Workspace: ImageAllocator<i8, L>,
        Self: ConvertImage<i32, i8, L>,
    {
        self.convert_to::<i8>()
    }
}

impl<'a> ImagePipeline<'a, i8, C1> {
    pub fn convert_to_u8(self) -> Result<ImagePipeline<'a, u8, C1>> {
        self.convert_to::<u8>()
    }

    pub fn convert_to_u16(self) -> Result<ImagePipeline<'a, u16, C1>> {
        self.convert_to::<u16>()
    }

    pub fn convert_to_i16(self) -> Result<ImagePipeline<'a, i16, C1>> {
        self.convert_to::<i16>()
    }

    pub fn convert_to_u32(self) -> Result<ImagePipeline<'a, u32, C1>> {
        self.convert_to::<u32>()
    }
}

impl<'a, L> ImagePipeline<'a, i8, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_i32(self) -> Result<ImagePipeline<'a, i32, L>>
    where
        Workspace: ImageAllocator<i32, L>,
        Self: ConvertImage<i8, i32, L>,
    {
        self.convert_to::<i32>()
    }

    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<i8, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, f16, L>
where
    L: ChannelLayout,
{
    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        Workspace: ImageAllocator<f32, L>,
        Self: ConvertImage<f16, f32, L>,
    {
        self.convert_to::<f32>()
    }
}
