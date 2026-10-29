use self::helpers::PlanarP4ColorConversionExt;
use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        memory::Image,
        view::{AC4, C3},
    },
    pipeline::image::planar::PlanarImage,
};

mod helpers;

impl PlanarImage<u8, 4> {
    /// Converts BGR color values to HLS color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_hls_u8_p4`](crate::image::raw::color::bgr_to_hls_u8_p4).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// After native conversion, copies the source fourth plane into the returned fourth plane.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    ///
    ///
    /// Copying the fourth plane can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn bgr_to_hls(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_hls_u8_p4)
    }

    /// Converts HLS color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`hls_to_bgr_u8_p4`](crate::image::raw::color::hls_to_bgr_u8_p4).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// After native conversion, copies the source fourth plane into the returned fourth plane.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    ///
    ///
    /// Copying the fourth plane can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn hls_to_bgr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::hls_to_bgr_u8_p4)
    }

    /// Converts JPEG YCCK planes to CMYK using 601 RGB coefficients and CMY inversion.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycck_to_cmyk_jpeg_601_u8_p4`](crate::image::raw::color::ycck_to_cmyk_jpeg_601_u8_p4).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// After native conversion, copies the source fourth plane into the returned fourth plane.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    ///
    ///
    /// Copying the fourth plane can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycck_to_cmyk_jpeg_601(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycck_to_cmyk_jpeg_601_u8_p4)
    }

    /// Converts JPEG CMYK or YCCK planes to packed RGB pixels without color-profile conversion.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3`](crate::image::raw::color::cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3).
    /// Allocates three owned full-resolution output planes.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn cmyk_or_ycck_to_rgb_jpeg_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_p3(stream_context, color::cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3)
    }

    /// Converts JPEG CMYK or YCCK planes to packed BGR pixels without color-profile conversion.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3`](crate::image::raw::color::cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3).
    /// Allocates three owned full-resolution output planes.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn cmyk_or_ycck_to_bgr_jpeg_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_p3(stream_context, color::cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3)
    }

    /// Converts JPEG CMYK or YCCK planes to packed RGB pixels without color-profile conversion.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3`](crate::image::raw::color::cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn cmyk_or_ycck_to_rgb_jpeg_c3(
        self,
        stream_context: &StreamContext,
    ) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_c3(stream_context, color::cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3)
    }

    /// Converts JPEG CMYK or YCCK planes to packed BGR pixels without color-profile conversion.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3`](crate::image::raw::color::cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn cmyk_or_ycck_to_bgr_jpeg_c3(
        self,
        stream_context: &StreamContext,
    ) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_c3(stream_context, color::cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3)
    }

    /// Converts BGR color values to HLS color values. The destination alpha channel is not processed.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_hls_u8_p4_to_ac4`](crate::image::raw::color::bgr_to_hls_u8_p4_to_ac4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Copies the source fourth plane into the returned packed alpha channel after conversion.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    ///
    ///
    /// Inserting the source fourth plane into alpha can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn bgr_to_hls_ac4(self, stream_context: &StreamContext) -> Result<Image<u8, AC4>>
    where
        u8: SupportedImage<AC4>,
    {
        self.color_convert_to_ac4(stream_context, color::bgr_to_hls_u8_p4_to_ac4)
    }

    /// Converts HLS color values to BGR color values. The destination alpha channel is not processed.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`hls_to_bgr_u8_p4_to_ac4`](crate::image::raw::color::hls_to_bgr_u8_p4_to_ac4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Copies the source fourth plane into the returned packed alpha channel after conversion.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    ///
    ///
    /// Inserting the source fourth plane into alpha can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn hls_to_bgr_ac4(self, stream_context: &StreamContext) -> Result<Image<u8, AC4>>
    where
        u8: SupportedImage<AC4>,
    {
        self.color_convert_to_ac4(stream_context, color::hls_to_bgr_u8_p4_to_ac4)
    }
}
