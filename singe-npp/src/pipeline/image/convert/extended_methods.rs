use singe_cuda::types::f16;

use crate::{
    error::Result,
    image::{
        SupportedImage,
        view::{C1, ChannelLayout},
    },
    pipeline::image::convert::{ConvertImage, ImagePipeline},
};

impl<'a, L> ImagePipeline<'a, u32, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `u32` to `f32` using native conversion semantics.
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
        Self: ConvertImage<u32, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, i32, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `i32` to `i8` using native conversion semantics.
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
    pub fn convert_to_i8(self) -> Result<ImagePipeline<'a, i8, L>>
    where
        i8: SupportedImage<L>,
        Self: ConvertImage<i32, i8, L>,
    {
        self.convert_to::<i8>()
    }
}

impl<'a> ImagePipeline<'a, i8, C1> {
    /// Converts each image sample from `i8` to `u8` using native conversion semantics.
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
    pub fn convert_to_u8(self) -> Result<ImagePipeline<'a, u8, C1>> {
        self.convert_to::<u8>()
    }

    /// Converts each image sample from `i8` to `u16` using native conversion semantics.
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
    pub fn convert_to_u16(self) -> Result<ImagePipeline<'a, u16, C1>> {
        self.convert_to::<u16>()
    }

    /// Converts each image sample from `i8` to `i16` using native conversion semantics.
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
    pub fn convert_to_i16(self) -> Result<ImagePipeline<'a, i16, C1>> {
        self.convert_to::<i16>()
    }

    /// Converts each image sample from `i8` to `u32` using native conversion semantics.
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
    pub fn convert_to_u32(self) -> Result<ImagePipeline<'a, u32, C1>> {
        self.convert_to::<u32>()
    }
}

impl<'a, L> ImagePipeline<'a, i8, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `i8` to `i32` using native conversion semantics.
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
        Self: ConvertImage<i8, i32, L>,
    {
        self.convert_to::<i32>()
    }

    /// Converts each image sample from `i8` to `f32` using native conversion semantics.
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
        Self: ConvertImage<i8, f32, L>,
    {
        self.convert_to::<f32>()
    }
}

impl<'a, L> ImagePipeline<'a, f16, L>
where
    L: ChannelLayout,
{
    /// Converts each image sample from `f16` to `f32` using native conversion semantics.
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
        Self: ConvertImage<f16, f32, L>,
    {
        self.convert_to::<f32>()
    }
}
