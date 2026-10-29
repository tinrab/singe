use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        memory::Image,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::planar::planar_color_methods::PlanarImage,
};

type PlanarToC2ColorConvert = for<'r, 'g, 'b, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'r, u8, C1>,
    &ImageView<'g, u8, C1>,
    &ImageView<'b, u8, C1>,
    &mut ImageOutputView<'destination, u8, C2>,
) -> Result<()>;

impl PlanarImage<u8, 3> {
    /// Converts RGB color values to YCbCr color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`rgb_to_ycbcr_u8_p3`](crate::image::raw::color::rgb_to_ycbcr_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
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
    pub fn rgb_to_ycbcr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::rgb_to_ycbcr_u8_p3)
    }

    /// Converts YCbCr color values to RGB color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr_to_rgb_u8_p3`](crate::image::raw::color::ycbcr_to_rgb_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
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
    pub fn ycbcr_to_rgb(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycbcr_to_rgb_u8_p3)
    }

    /// Converts RGB color values to `YCbCr444` color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`rgb_to_ycbcr444_jpeg_u8_p3`](crate::image::raw::color::rgb_to_ycbcr444_jpeg_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
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
    pub fn rgb_to_ycbcr444_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::rgb_to_ycbcr444_jpeg_u8_p3)
    }

    /// Converts BGR color values to `YCbCr444` color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_ycbcr444_jpeg_u8_p3`](crate::image::raw::color::bgr_to_ycbcr444_jpeg_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
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
    pub fn bgr_to_ycbcr444_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_ycbcr444_jpeg_u8_p3)
    }

    /// Converts `YCbCr444` color values to RGB color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr444_to_rgb_jpeg_u8_p3`](crate::image::raw::color::ycbcr444_to_rgb_jpeg_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
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
    pub fn ycbcr444_to_rgb_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycbcr444_to_rgb_jpeg_u8_p3)
    }

    /// Converts `YCbCr444` color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr444_to_bgr_jpeg_u8_p3`](crate::image::raw::color::ycbcr444_to_bgr_jpeg_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
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
    pub fn ycbcr444_to_bgr_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycbcr444_to_bgr_jpeg_u8_p3)
    }

    /// Converts the separate red, green, and blue source planes to packed `YCbCr422` pixels.
    ///
    /// Uses [`rgb_to_ycbcr422_p3_to_c2`](crate::image::raw::color::rgb_to_ycbcr422_p3_to_c2).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// All source and destination views must have equal image sizes, and the three source planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn rgb_to_ycbcr422(self, stream_context: &StreamContext) -> Result<Image<u8, C2>> {
        self.color_convert_to_c2(stream_context, color::rgb_to_ycbcr422_p3_to_c2)
    }

    /// Converts the separate red, green, and blue source planes to packed `YCrCb422` pixels.
    ///
    /// Uses [`rgb_to_ycrcb422_p3_to_c2`](crate::image::raw::color::rgb_to_ycrcb422_p3_to_c2).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// All source and destination views must have equal image sizes, and the three source planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn rgb_to_ycrcb422(self, stream_context: &StreamContext) -> Result<Image<u8, C2>> {
        self.color_convert_to_c2(stream_context, color::rgb_to_ycrcb422_p3_to_c2)
    }

    fn color_convert_to_c2(
        self,
        stream_context: &StreamContext,
        operation: PlanarToC2ColorConvert,
    ) -> Result<Image<u8, C2>>
    where
        u8: SupportedImage<C2>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Image::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source_r = self.planes[0].view()?;
            let source_g = self.planes[1].view()?;
            let source_b = self.planes[2].view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(
                        stream_context,
                        &source_r,
                        &source_g,
                        &source_b,
                        &mut destination_view,
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
