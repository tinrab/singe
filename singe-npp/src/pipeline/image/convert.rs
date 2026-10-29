use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
    types::DataTypeLike,
};

mod convert_dispatch;
mod extended_methods;
mod signed_methods;

use convert_dispatch::ConvertImage;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: DataTypeLike,
    L: ChannelLayout,
{
    /// Converts each image sample to `U` using native conversion semantics.
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
    pub fn convert_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, U, L>,
    ) -> Result<()>
    where
        U: DataTypeLike,
        Self: ConvertImage<T, U, L>,
    {
        <Self as ConvertImage<T, U, L>>::convert_image(stream_context, source, destination)
    }

    /// Converts each image sample to `U` using native conversion semantics.
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
    pub fn convert_to<U>(self) -> Result<ImagePipeline<'a, U, L>>
    where
        U: DataTypeLike + SupportedImage<L>,
        Self: ConvertImage<T, U, L>,
    {
        self.ensure_eager()?;

        let mut destination =
            Image::<U, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ConvertImage<T, U, L>>::convert_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `u8` to `u16` using native conversion semantics.
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
    pub fn convert_to_u16(self) -> Result<ImagePipeline<'a, u16, L>>
    where
        u16: SupportedImage<L>,
        Self: ConvertImage<u8, u16, L>,
    {
        self.convert_to::<u16>()
    }

    /// Converts each image sample from `u8` to `i16` using native conversion semantics.
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
    pub fn convert_to_i16(self) -> Result<ImagePipeline<'a, i16, L>>
    where
        i16: SupportedImage<L>,
        Self: ConvertImage<u8, i16, L>,
    {
        self.convert_to::<i16>()
    }

    /// Converts each image sample from `u8` to `i32` using native conversion semantics.
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
    pub fn convert_to_i32(self) -> Result<ImagePipeline<'a, i32, L>>
    where
        i32: SupportedImage<L>,
        Self: ConvertImage<u8, i32, L>,
    {
        self.convert_to::<i32>()
    }

    /// Converts each image sample from `u8` to `f32` using native conversion semantics.
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
    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        f32: SupportedImage<L>,
        Self: ConvertImage<u8, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, u16, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `u16` to `u8` using native conversion semantics.
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
    pub fn convert_to_u8(self) -> Result<ImagePipeline<'a, u8, L>>
    where
        u8: SupportedImage<L>,
        Self: ConvertImage<u16, u8, L>,
    {
        self.convert_to::<u8>()
    }

    /// Converts each image sample from `u16` to `i32` using native conversion semantics.
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
    pub fn convert_to_i32(self) -> Result<ImagePipeline<'a, i32, L>>
    where
        i32: SupportedImage<L>,
        Self: ConvertImage<u16, i32, L>,
    {
        self.convert_to::<i32>()
    }

    /// Converts each image sample from `u16` to `u32` using native conversion semantics.
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
    pub fn convert_to_u32(self) -> Result<ImagePipeline<'a, u32, L>>
    where
        u32: SupportedImage<L>,
        Self: ConvertImage<u16, u32, L>,
    {
        self.convert_to::<u32>()
    }

    /// Converts each image sample from `u16` to `f32` using native conversion semantics.
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
    pub fn convert_to_f32(self) -> Result<ImagePipeline<'a, f32, L>>
    where
        f32: SupportedImage<L>,
        Self: ConvertImage<u16, f32, L>,
    {
        self.convert_to::<f32>()
    }
}
