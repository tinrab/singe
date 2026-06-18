use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{DataTypeLike, RoundMode},
};

use super::super::{ImageBacking, ImagePipeline};
use super::round_dispatch::RoundConvertImage;

impl<'a, L> ImagePipeline<'a, f32, L>
where
    L: ChannelLayout,
{
    pub fn convert_round_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        round_mode: RoundMode,
    ) -> Result<()>
    where
        U: DataTypeLike,
        Self: RoundConvertImage<f32, U, L>,
    {
        <Self as RoundConvertImage<f32, U, L>>::round_convert_image(
            stream_context,
            source,
            destination,
            round_mode,
        )
    }

    pub fn convert_round_to<U>(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, U, L>>
    where
        U: DataTypeLike,
        Workspace: ImageAllocator<U, L>,
        Self: RoundConvertImage<f32, U, L>,
    {
        let mut destination = self.workspace.image::<U, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as RoundConvertImage<f32, U, L>>::round_convert_image(
                self.stream_context,
                &source,
                &mut destination_view,
                round_mode,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    fn round_convert_output<U>(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, U, L>>
    where
        U: DataTypeLike,
        Workspace: ImageAllocator<U, L>,
        Self: RoundConvertImage<f32, U, L>,
    {
        self.convert_round_to::<U>(round_mode)
    }

    pub fn convert_to_u8(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, u8, L>>
    where
        Workspace: ImageAllocator<u8, L>,
        Self: RoundConvertImage<f32, u8, L>,
    {
        self.round_convert_output::<u8>(round_mode)
    }

    pub fn convert_to_i8(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, i8, L>>
    where
        Workspace: ImageAllocator<i8, L>,
        Self: RoundConvertImage<f32, i8, L>,
    {
        self.round_convert_output::<i8>(round_mode)
    }

    pub fn convert_to_u16(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, u16, L>>
    where
        Workspace: ImageAllocator<u16, L>,
        Self: RoundConvertImage<f32, u16, L>,
    {
        self.round_convert_output::<u16>(round_mode)
    }

    pub fn convert_to_i16(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, i16, L>>
    where
        Workspace: ImageAllocator<i16, L>,
        Self: RoundConvertImage<f32, i16, L>,
    {
        self.round_convert_output::<i16>(round_mode)
    }

    pub fn convert_to_f16(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, f16, L>>
    where
        Workspace: ImageAllocator<f16, L>,
        Self: RoundConvertImage<f32, f16, L>,
    {
        self.round_convert_output::<f16>(round_mode)
    }
}
