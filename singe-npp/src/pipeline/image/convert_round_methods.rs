use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{DataTypeLike, RoundMode},
};

use super::{ImageBacking, ImagePipeline};

#[path = "convert_round_dispatch.rs"]
mod round_dispatch;

#[path = "convert_round_to_methods.rs"]
mod round_methods;

#[path = "convert_round_typed_methods.rs"]
mod typed_methods;

use round_dispatch::ScaledRoundConvertImage;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: DataTypeLike,
{
    pub fn convert_scaled_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, U, C1>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: DataTypeLike,
        Self: ScaledRoundConvertImage<T, U, C1>,
    {
        <Self as ScaledRoundConvertImage<T, U, C1>>::scaled_round_convert_image(
            stream_context,
            source,
            destination,
            round_mode,
            scale_factor,
        )
    }

    pub fn convert_scaled_to<U>(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, U, C1>>
    where
        U: DataTypeLike,
        Workspace: ImageAllocator<U, C1>,
        Self: ScaledRoundConvertImage<T, U, C1>,
    {
        let mut destination = self.workspace.image::<U, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaledRoundConvertImage<T, U, C1>>::scaled_round_convert_image(
                self.stream_context,
                &source,
                &mut destination_view,
                round_mode,
                scale_factor,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub(super) fn scaled_round_convert_output<U>(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, U, C1>>
    where
        U: DataTypeLike,
        Workspace: ImageAllocator<U, C1>,
        Self: ScaledRoundConvertImage<T, U, C1>,
    {
        self.convert_scaled_to::<U>(round_mode, scale_factor)
    }
}
