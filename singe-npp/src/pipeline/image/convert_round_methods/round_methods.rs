use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, convert_round_methods::round_dispatch::RoundConvertImage,
    },
    types::{DataTypeLike, RoundMode},
};

impl<'a, L> ImagePipeline<'a, f32, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `f32` to `U` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Writes the converted samples into the equally sized `destination`.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Source and destination dimensions must match.
    pub fn convert_round_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        destination: &mut ImageOutputView<'_, U, L>,
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

    /// Converts each image sample from `f32` to `U` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Allocating the output image, calculating its size, or checking the source producer
    /// for capture can also fail.
    pub fn convert_round_to<U>(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, U, L>>
    where
        U: DataTypeLike + SupportedImage<L>,
        Self: RoundConvertImage<f32, U, L>,
    {
        self.ensure_eager()?;

        let mut destination =
            Image::<U, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
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
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    fn round_convert_output<U>(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, U, L>>
    where
        U: DataTypeLike + SupportedImage<L>,
        Self: RoundConvertImage<f32, U, L>,
    {
        self.convert_round_to::<U>(round_mode)
    }

    /// Converts each image sample from `f32` to `u8` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Allocating the output image, calculating its size, or checking the source producer
    /// for capture can also fail.
    pub fn convert_to_u8(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, u8, L>>
    where
        u8: SupportedImage<L>,
        Self: RoundConvertImage<f32, u8, L>,
    {
        self.round_convert_output::<u8>(round_mode)
    }

    /// Converts each image sample from `f32` to `i8` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Allocating the output image, calculating its size, or checking the source producer
    /// for capture can also fail.
    pub fn convert_to_i8(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, i8, L>>
    where
        i8: SupportedImage<L>,
        Self: RoundConvertImage<f32, i8, L>,
    {
        self.round_convert_output::<i8>(round_mode)
    }

    /// Converts each image sample from `f32` to `u16` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Allocating the output image, calculating its size, or checking the source producer
    /// for capture can also fail.
    pub fn convert_to_u16(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, u16, L>>
    where
        u16: SupportedImage<L>,
        Self: RoundConvertImage<f32, u16, L>,
    {
        self.round_convert_output::<u16>(round_mode)
    }

    /// Converts each image sample from `f32` to `i16` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Allocating the output image, calculating its size, or checking the source producer
    /// for capture can also fail.
    pub fn convert_to_i16(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, i16, L>>
    where
        i16: SupportedImage<L>,
        Self: RoundConvertImage<f32, i16, L>,
    {
        self.round_convert_output::<i16>(round_mode)
    }

    /// Converts each image sample from `f32` to `f16` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Allocating the output image, calculating its size, or checking the source producer
    /// for capture can also fail.
    pub fn convert_to_f16(self, round_mode: RoundMode) -> Result<ImagePipeline<'a, f16, L>>
    where
        f16: SupportedImage<L>,
        Self: RoundConvertImage<f32, f16, L>,
    {
        self.round_convert_output::<f16>(round_mode)
    }
}
