use crate::{
    context::StreamContext,
    error::Result,
    image::{SupportedImage, color, memory::Image, view::C3},
    pipeline::image::planar::SubsampledPlanarImage,
};

impl SubsampledPlanarImage<u8> {
    /// 3 channel planar `YCbCr420` to 3 channel packed RGB color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`ycbcr420_to_rgb_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr420_to_rgb_jpeg_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr420_to_rgb_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::ycbcr420_to_rgb_jpeg_u8_p3_to_c3,
            None,
        )
    }

    /// 3 channel planar `YCbCr422` to 3 channel packed RGB color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`ycbcr422_to_rgb_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr422_to_rgb_jpeg_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr422_to_rgb_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::ycbcr422_to_rgb_jpeg_u8_p3_to_c3,
            None,
        )
    }

    /// 3 channel planar `YCbCr411` to 3 channel packed RGB color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`ycbcr411_to_rgb_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr411_to_rgb_jpeg_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr411_to_rgb_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::ycbcr411_to_rgb_jpeg_u8_p3_to_c3,
            None,
        )
    }

    /// 3 channel planar `YCbCr420` to 3 channel packed BGR color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`ycbcr420_to_bgr_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr420_to_bgr_jpeg_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr420_to_bgr_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::ycbcr420_to_bgr_jpeg_u8_p3_to_c3,
            None,
        )
    }

    /// 3 channel planar `YCbCr422` to 3 channel packed BGR color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`ycbcr422_to_bgr_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr422_to_bgr_jpeg_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr422_to_bgr_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::ycbcr422_to_bgr_jpeg_u8_p3_to_c3,
            None,
        )
    }

    /// 3 channel planar `YCbCr411` to 3 channel packed BGR color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`ycbcr411_to_bgr_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr411_to_bgr_jpeg_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr411_to_bgr_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::ycbcr411_to_bgr_jpeg_u8_p3_to_c3,
            None,
        )
    }
}
