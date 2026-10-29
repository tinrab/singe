use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{planar::SubsampledPlanarImage, planar_pipeline_color::ImagePipeline},
};

pub type PackedToSubsampledPlanarColorConvert<L> =
    for<'source, 'y, 'cb, 'cr> unsafe fn(
        &StreamContext,
        &ImageView<'source, u8, L>,
        &mut ImageOutputView<'y, u8, C1>,
        &mut ImageOutputView<'cb, u8, C1>,
        &mut ImageOutputView<'cr, u8, C1>,
    ) -> Result<()>;

pub type Nv12PackedForward<T, L> = for<'source, 'y, 'uv> unsafe fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageOutputView<'y, T, C1>,
    &mut ImageOutputView<'uv, T, C2>,
    color::ColorTwistMatrix,
) -> Result<()>;

mod nv12_color;

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C1>,
{
    /// 3 channel packed RGB to 3 channel planar `YCbCr420` color conversion.
    ///
    /// Uses [`rgb_to_ycbcr420_jpeg_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr420_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr420_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_ycbcr420_jpeg_u8_c3_to_p3)
    }

    /// 3 channel packed RGB to 3 channel planar `YCbCr422` color conversion.
    ///
    /// Uses [`rgb_to_ycbcr422_jpeg_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr422_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr422_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::rgb_to_ycbcr422_jpeg_u8_c3_to_p3)
    }

    /// 3 channel packed RGB to 3 channel planar `YCbCr411` color conversion.
    ///
    /// Uses [`rgb_to_ycbcr411_jpeg_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr411_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 4 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr411_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::rgb_to_ycbcr411_jpeg_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr420` color conversion.
    ///
    /// Uses [`bgr_to_ycbcr420_jpeg_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr420_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr420_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_jpeg_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr422` color conversion.
    ///
    /// Uses [`bgr_to_ycbcr422_jpeg_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr422_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr422_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::bgr_to_ycbcr422_jpeg_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr411` color conversion.
    ///
    /// Uses [`bgr_to_ycbcr411_jpeg_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr411_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 4 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr411_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::bgr_to_ycbcr411_jpeg_u8_c3_to_p3)
    }

    /// 3 channel packed RGB to 3 channel planar `YCbCr420` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`rgb_to_ycbcr420_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr420_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_ycbcr420_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr420` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`bgr_to_ycbcr420_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr420_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCrCb420` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`bgr_to_ycrcb420_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycrcb420_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycrcb420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycrcb420_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr420_709CSC` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`bgr_to_ycbcr420_709csc_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr420_709csc_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr420_709csc_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_709csc_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCrCb420_709CSC` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`bgr_to_ycrcb420_709csc_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycrcb420_709csc_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycrcb420_709csc_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycrcb420_709csc_u8_c3_to_p3)
    }

    /// RGB to `YCbCr411` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`rgb_to_ycbcr411_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr411_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 4 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr411_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::rgb_to_ycbcr411_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr411` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`bgr_to_ycbcr411_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr411_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 4 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr411_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::bgr_to_ycbcr411_u8_c3_to_p3)
    }

    /// 3 channel packed RGB to 3 channel planar `YCbCr422` color conversion.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`rgb_to_ycbcr422_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr422_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::rgb_to_ycbcr422_u8_c3_to_p3)
    }

    /// 3 channel packed BGR to 3 channel planar `YCbCr422` color conversion. images.
    ///
    /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
    /// Each plane's byte stride is passed independently.
    ///
    /// Uses [`bgr_to_ycbcr422_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr422_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Destination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Destination full-resolution dimensions must be divisible by those subsampling factors.
    /// Separate Cb and Cr planes must have equal sizes.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::bgr_to_ycbcr422_u8_c3_to_p3)
    }

    /// 3 channel packed RGB to 3 channel planar YUV420 color conversion.
    ///
    /// Uses [`rgb_to_yuv420_u8_c3_to_p3`](crate::image::raw::color::rgb_to_yuv420_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 2. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_yuv420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_yuv420_u8_c3_to_p3)
    }

    /// 3 channel packed RGB to 3 channel planar YUV422 color conversion.
    ///
    /// Uses [`rgb_to_yuv422_u8_c3_to_p3`](crate::image::raw::color::rgb_to_yuv422_u8_c3_to_p3).
    ///
    /// Allocates three owned planes: a full-resolution first plane and two chroma planes
    /// with width divided by 2 and height divided by 1. Full-resolution dimensions
    /// must be divisible by these factors. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_yuv422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::rgb_to_yuv422_u8_c3_to_p3)
    }

    fn color_convert_to_subsampled_planar(
        self,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
        operation: PackedToSubsampledPlanarColorConvert<C3>,
    ) -> Result<SubsampledPlanarImage<u8>> {
        self.ensure_eager()?;

        let mut destination = SubsampledPlanarImage::create_uninit(
            self.stream_context.cuda_context(),
            self.size(),
            horizontal_subsampling,
            vertical_subsampling,
        )?;

        {
            let source = self.view()?;
            let mut y = destination.y.output()?;
            let mut cb = destination.cb.output()?;
            let mut cr = destination.cr.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(self.stream_context, &source, &mut y, &mut cb, &mut cr) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}

mod ac4_subsampled_color;
mod u16_nv12_color;
