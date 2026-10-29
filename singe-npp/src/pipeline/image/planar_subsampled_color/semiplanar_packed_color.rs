use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        memory::Image,
        view::{C3, C4},
    },
    pipeline::image::planar::SemiplanarImage,
};

impl SemiplanarImage<u8> {
    /// 2 channel planar NV12 to 3 channel packed RGB color conversion.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv12_to_rgb_u8_p2_to_c3`](crate::image::raw::color::nv12_to_rgb_u8_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_rgb_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_rgb_u8_p2_to_c3)
    }

    /// 2 channel planar NV12 to 3 channel packed RGB 709 HDTV full color conversion.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv12_to_rgb_709hdtv_u8_p2_to_c3`](crate::image::raw::color::nv12_to_rgb_709hdtv_u8_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_rgb_709hdtv_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_rgb_709hdtv_u8_p2_to_c3)
    }

    /// 2 channel planar NV12 to 3 channel packed RGB 709 CSC color conversion.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv12_to_rgb_709csc_u8_p2_to_c3`](crate::image::raw::color::nv12_to_rgb_709csc_u8_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_rgb_709csc_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_rgb_709csc_u8_p2_to_c3)
    }

    /// 2 channel planar NV12 to 3 channel packed BGR color conversion.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv12_to_bgr_u8_p2_to_c3`](crate::image::raw::color::nv12_to_bgr_u8_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_bgr_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_bgr_u8_p2_to_c3)
    }

    /// 2 channel planar NV12 to 3 channel packed RGB 709 HDTV full color conversion.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv12_to_bgr_709hdtv_u8_p2_to_c3`](crate::image::raw::color::nv12_to_bgr_709hdtv_u8_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_bgr_709hdtv_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_bgr_709hdtv_u8_p2_to_c3)
    }

    /// 2 channel planar NV12 to 3 channel packed RGB 709 CSC color conversion.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv12_to_bgr_709csc_u8_p2_to_c3`](crate::image::raw::color::nv12_to_bgr_709csc_u8_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_bgr_709csc_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_bgr_709csc_u8_p2_to_c3)
    }

    /// 2 channel planar NV21 to 4 channel packed RGBA color conversion with constant alpha (0xFF).
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv21_to_rgb_u8_p2_to_c4`](crate::image::raw::color::nv21_to_rgb_u8_p2_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv21_to_rgb_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed(stream_context, color::nv21_to_rgb_u8_p2_to_c4)
    }

    /// 2 channel planar NV21 to 4 channel packed BGRA color conversion with constant alpha (0xFF).
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    ///
    /// Uses [`nv21_to_bgr_u8_p2_to_c4`](crate::image::raw::color::nv21_to_bgr_u8_p2_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv21_to_bgr_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed(stream_context, color::nv21_to_bgr_u8_p2_to_c4)
    }

    /// Two channel planar NV12 conversion to three channel packed RGB, using a color twist to compute the exact color space arithmetic.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
    ///
    /// Uses [`nv12_to_rgb_u8_color_twist_p2_to_c3`](crate::image::raw::color::nv12_to_rgb_u8_color_twist_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_rgb_color_twist_c3(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_twist_to_packed(
            stream_context,
            twist,
            color::nv12_to_rgb_u8_color_twist_p2_to_c3,
        )
    }
}

impl SemiplanarImage<u16> {
    /// Two channel planar NV12 conversion to three channel packed RGB, using a color twist to compute the exact color space arithmetic.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
    ///
    /// Uses [`nv12_to_rgb_u16_color_twist_p2_to_c3`](crate::image::raw::color::nv12_to_rgb_u16_color_twist_p2_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The source luma and interleaved-chroma planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn nv12_to_rgb_color_twist_c3(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Image<u16, C3>>
    where
        u16: SupportedImage<C3>,
    {
        self.color_twist_to_packed(
            stream_context,
            twist,
            color::nv12_to_rgb_u16_color_twist_p2_to_c3,
        )
    }
}
