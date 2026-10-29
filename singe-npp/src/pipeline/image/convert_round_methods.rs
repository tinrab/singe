use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
    types::{DataTypeLike, RoundMode},
};

mod round_dispatch;

mod round_methods;

mod typed_methods;

use round_dispatch::ScaledRoundConvertImage;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: DataTypeLike,
{
    /// Converts each image sample to `U` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Passes `scale_factor` to the native scaled conversion without validating its numeric range.
    /// Writes the converted samples into the equally sized `destination`.
    /// Processes every physical channel. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Source and destination dimensions must match.
    pub fn convert_scaled_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, U, C1>,
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

    /// Converts each image sample to `U` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Passes `scale_factor` to the native scaled conversion without validating its numeric range.
    /// Allocates a new owned image with the same pixel dimensions and channel layout.
    /// Processes every physical channel. Row padding remains unwritten.
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
    pub fn convert_scaled_to<U>(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, U, C1>>
    where
        U: DataTypeLike + SupportedImage<C1>,
        Self: ScaledRoundConvertImage<T, U, C1>,
    {
        self.ensure_eager()?;

        let mut destination =
            Image::<U, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
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
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn scaled_round_convert_output<U>(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, U, C1>>
    where
        U: DataTypeLike + SupportedImage<C1>,
        Self: ScaledRoundConvertImage<T, U, C1>,
    {
        self.convert_scaled_to::<U>(round_mode, scale_factor)
    }
}
