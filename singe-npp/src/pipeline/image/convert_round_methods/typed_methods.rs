use crate::{error::Result, image::view::C1, pipeline::image::ImagePipeline, types::RoundMode};

mod f32_methods;

impl<'a> ImagePipeline<'a, u16, C1> {
    /// Converts each image sample from `u16` to `i8` using native conversion semantics.
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
    pub fn convert_scaled_to_i8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i8, C1>> {
        self.scaled_round_convert_output::<i8>(round_mode, scale_factor)
    }

    /// Converts each image sample from `u16` to `i16` using native conversion semantics.
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
    pub fn convert_scaled_to_i16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i16, C1>> {
        self.scaled_round_convert_output::<i16>(round_mode, scale_factor)
    }
}

impl<'a> ImagePipeline<'a, i16, C1> {
    /// Converts each image sample from `i16` to `i8` using native conversion semantics.
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
    pub fn convert_scaled_to_i8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i8, C1>> {
        self.scaled_round_convert_output::<i8>(round_mode, scale_factor)
    }
}

impl<'a> ImagePipeline<'a, u8, C1> {
    /// Converts each image sample from `u8` to `i8` using native conversion semantics.
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
    pub fn convert_scaled_to_i8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i8, C1>> {
        self.scaled_round_convert_output::<i8>(round_mode, scale_factor)
    }
}

impl<'a> ImagePipeline<'a, u32, C1> {
    /// Converts each image sample from `u32` to `u8` using native conversion semantics.
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
    pub fn convert_scaled_to_u8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, u8, C1>> {
        self.scaled_round_convert_output::<u8>(round_mode, scale_factor)
    }

    /// Converts each image sample from `u32` to `u16` using native conversion semantics.
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
    pub fn convert_scaled_to_u16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, u16, C1>> {
        self.scaled_round_convert_output::<u16>(round_mode, scale_factor)
    }

    /// Converts each image sample from `u32` to `i8` using native conversion semantics.
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
    pub fn convert_scaled_to_i8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i8, C1>> {
        self.scaled_round_convert_output::<i8>(round_mode, scale_factor)
    }

    /// Converts each image sample from `u32` to `i16` using native conversion semantics.
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
    pub fn convert_scaled_to_i16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i16, C1>> {
        self.scaled_round_convert_output::<i16>(round_mode, scale_factor)
    }

    /// Converts each image sample from `u32` to `i32` using native conversion semantics.
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
    pub fn convert_scaled_to_i32(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i32, C1>> {
        self.scaled_round_convert_output::<i32>(round_mode, scale_factor)
    }
}

impl<'a> ImagePipeline<'a, i32, C1> {
    /// Converts each image sample from `i32` to `u16` using native conversion semantics.
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
    pub fn convert_scaled_to_u16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, u16, C1>> {
        self.scaled_round_convert_output::<u16>(round_mode, scale_factor)
    }

    /// Converts each image sample from `i32` to `i16` using native conversion semantics.
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
    pub fn convert_scaled_to_i16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i16, C1>> {
        self.scaled_round_convert_output::<i16>(round_mode, scale_factor)
    }
}
