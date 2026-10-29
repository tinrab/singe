use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, color,
        view::{C1, C3, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, planar::PlanarImage},
    },
};

mod ac4_color;
mod c2_color;
mod subsampled_color;

type PackedToPlanarColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'source, u8, C3>,
    &mut PlanarImageOutputView<'destination, u8, 3>,
) -> Result<()>;

type PlanarToPackedColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 3>,
    &mut ImageOutputView<'destination, u8, C3>,
) -> Result<()>;

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C1>,
{
    /// Converts RGB color values to YUV color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`rgb_to_yuv_u8_c3_to_p3`](crate::image::raw::color::rgb_to_yuv_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_yuv_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::rgb_to_yuv_u8_c3_to_p3)
    }

    /// Converts RGB color values to YCbCr color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`rgb_to_ycbcr_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::rgb_to_ycbcr_u8_c3_to_p3)
    }

    /// Converts BGR color values to YUV color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_yuv_u8_c3_to_p3`](crate::image::raw::color::bgr_to_yuv_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_yuv_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_yuv_u8_c3_to_p3)
    }

    /// Converts BGR color values to YCbCr color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_ycbcr_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_ycbcr_u8_c3_to_p3)
    }

    /// Converts BGR color values to HLS color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_hls_u8_c3_to_p3`](crate::image::raw::color::bgr_to_hls_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_hls_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_hls_u8_c3_to_p3)
    }

    /// Converts HLS color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`hls_to_bgr_u8_c3_to_p3`](crate::image::raw::color::hls_to_bgr_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn hls_to_bgr_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::hls_to_bgr_u8_c3_to_p3)
    }

    /// Converts RGB color values to `YCbCr444` color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`rgb_to_ycbcr444_jpeg_u8_c3_to_p3`](crate::image::raw::color::rgb_to_ycbcr444_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr444_jpeg_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::rgb_to_ycbcr444_jpeg_u8_c3_to_p3)
    }

    /// Converts BGR color values to `YCbCr444` color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_ycbcr444_jpeg_u8_c3_to_p3`](crate::image::raw::color::bgr_to_ycbcr444_jpeg_u8_c3_to_p3).
    ///
    /// Allocates three owned full-resolution planes with the same dimensions as the input.
    /// Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr444_jpeg_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_ycbcr444_jpeg_u8_c3_to_p3)
    }

    fn color_convert_to_planar(
        self,
        operation: PackedToPlanarColorConvert,
    ) -> Result<PlanarImage<u8, 3>> {
        self.ensure_eager()?;

        let mut destination =
            PlanarImage::<u8, 3>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(self.stream_context, &source, &mut destination_view) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C3>,
{
    /// Converts YUV color values to RGB color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`yuv_to_rgb_u8_p3_to_c3`](crate::image::raw::color::yuv_to_rgb_u8_p3_to_c3).
    ///
    /// Reads the supplied `source` planes and allocates a new owned packed image at their
    /// full resolution. The current pipeline supplies the stream context and workspace;
    /// its existing image pixels are not conversion input. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn yuv_planar_to_rgb(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::yuv_to_rgb_u8_p3_to_c3)
    }

    /// Converts YUV color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`yuv_to_bgr_u8_p3_to_c3`](crate::image::raw::color::yuv_to_bgr_u8_p3_to_c3).
    ///
    /// Reads the supplied `source` planes and allocates a new owned packed image at their
    /// full resolution. The current pipeline supplies the stream context and workspace;
    /// its existing image pixels are not conversion input. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn yuv_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::yuv_to_bgr_u8_p3_to_c3)
    }

    /// Converts YCbCr color values to RGB color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr_to_rgb_u8_p3_to_c3`](crate::image::raw::color::ycbcr_to_rgb_u8_p3_to_c3).
    ///
    /// Reads the supplied `source` planes and allocates a new owned packed image at their
    /// full resolution. The current pipeline supplies the stream context and workspace;
    /// its existing image pixels are not conversion input. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn ycbcr_planar_to_rgb(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr_to_rgb_u8_p3_to_c3)
    }

    /// Converts YCbCr color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr_to_bgr_u8_p3_to_c3`](crate::image::raw::color::ycbcr_to_bgr_u8_p3_to_c3).
    ///
    /// Reads the supplied `source` planes and allocates a new owned packed image at their
    /// full resolution. The current pipeline supplies the stream context and workspace;
    /// its existing image pixels are not conversion input. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn ycbcr_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr_to_bgr_u8_p3_to_c3)
    }

    /// Converts `YCbCr444` color values to RGB color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr444_to_rgb_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr444_to_rgb_jpeg_u8_p3_to_c3).
    ///
    /// Reads the supplied `source` planes and allocates a new owned packed image at their
    /// full resolution. The current pipeline supplies the stream context and workspace;
    /// its existing image pixels are not conversion input. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn ycbcr444_jpeg_planar_to_rgb(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr444_to_rgb_jpeg_u8_p3_to_c3)
    }

    /// Converts `YCbCr444` color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`ycbcr444_to_bgr_jpeg_u8_p3_to_c3`](crate::image::raw::color::ycbcr444_to_bgr_jpeg_u8_p3_to_c3).
    ///
    /// Reads the supplied `source` planes and allocates a new owned packed image at their
    /// full resolution. The current pipeline supplies the stream context and workspace;
    /// its existing image pixels are not conversion input. Plane order follows the named color components.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn ycbcr444_jpeg_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr444_to_bgr_jpeg_u8_p3_to_c3)
    }

    fn color_convert_from_planar(
        self,
        source: &PlanarImageView<'_, u8, 3>,
        operation: PlanarToPackedColorConvert,
    ) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination = Image::<u8, C3>::create_uninit(
            self.stream_context.cuda_context(),
            source.planes()[0].size(),
        )?;

        {
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            for plane in source.planes() {
                self.stream_context.prepare_region(plane)?;
            }
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(self.stream_context, source, &mut destination_view) }
            })?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed conversion established every logical output channel.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
