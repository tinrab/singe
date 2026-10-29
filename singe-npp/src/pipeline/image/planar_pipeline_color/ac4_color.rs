use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, color, exchange,
        view::{AC4, C1, C4, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::{ImageBacking, planar::PlanarImage, planar_pipeline_color::ImagePipeline},
};

type PackedToPlanar4ColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'source, u8, AC4>,
    &mut PlanarImageOutputView<'destination, u8, 4>,
) -> Result<()>;

type Planar4ToPackedColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut ImageOutputView<'destination, u8, AC4>,
) -> Result<()>;

impl ImagePipeline<'_, u8, AC4>
where
    u8: SupportedImage<C1>,
{
    /// Converts RGB color values to YUV color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Allocates four full-resolution output planes. After native color conversion, extracts
    /// the input alpha channel into the fourth plane, preserving source alpha.
    /// Uses [`rgb_to_yuv_u8_ac4_to_p4`](crate::image::raw::color::rgb_to_yuv_u8_ac4_to_p4).
    /// Prepares all source regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, view construction, alpha-channel extraction or insertion, capture
    /// checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn rgb_to_yuv_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::rgb_to_yuv_u8_ac4_to_p4)
    }

    /// Converts BGR color values to YUV color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Allocates four full-resolution output planes. After native color conversion, extracts
    /// the input alpha channel into the fourth plane, preserving source alpha.
    /// Uses [`bgr_to_yuv_u8_ac4_to_p4`](crate::image::raw::color::bgr_to_yuv_u8_ac4_to_p4).
    /// Prepares all source regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, view construction, alpha-channel extraction or insertion, capture
    /// checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn bgr_to_yuv_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::bgr_to_yuv_u8_ac4_to_p4)
    }

    /// Converts BGR color values to HLS color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Allocates four full-resolution output planes. After native color conversion, extracts
    /// the input alpha channel into the fourth plane, preserving source alpha.
    /// Uses [`bgr_to_hls_u8_ac4_to_p4`](crate::image::raw::color::bgr_to_hls_u8_ac4_to_p4).
    /// Prepares all source regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, view construction, alpha-channel extraction or insertion, capture
    /// checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn bgr_to_hls_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::bgr_to_hls_u8_ac4_to_p4)
    }

    /// Converts HLS color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Allocates four full-resolution output planes. After native color conversion, extracts
    /// the input alpha channel into the fourth plane, preserving source alpha.
    /// Uses [`hls_to_bgr_u8_ac4_to_p4`](crate::image::raw::color::hls_to_bgr_u8_ac4_to_p4).
    /// Prepares all source regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, view construction, alpha-channel extraction or insertion, capture
    /// checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn hls_to_bgr_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::hls_to_bgr_u8_ac4_to_p4)
    }

    fn color_convert_to_planar(
        self,
        operation: PackedToPlanar4ColorConvert,
    ) -> Result<PlanarImage<u8, 4>> {
        self.ensure_eager()?;

        let mut destination =
            PlanarImage::<u8, 4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(self.stream_context, &source, &mut destination_view)?;
                    let rgba = source.with_layout::<C4>()?;
                    exchange::extract_channel_c4(
                        self.stream_context,
                        &rgba,
                        &mut destination_view.planes_mut()[3],
                        3,
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}

impl ImagePipeline<'_, u8, AC4>
where
    u8: SupportedImage<AC4>,
{
    /// Converts BGR color values to HLS color values. The destination alpha channel is not processed.
    ///
    /// Plane order follows the named color components.
    ///
    /// Reads the supplied `source` planes and allocates a new packed AC4 image at their
    /// full-resolution dimensions. The current pipeline supplies the stream context and
    /// workspace; its existing image pixels are not conversion input. Copies the fourth
    /// source plane into the output alpha channel after native color conversion.
    /// Uses [`bgr_to_hls_u8_p4_to_ac4`](crate::image::raw::color::bgr_to_hls_u8_p4_to_ac4).
    /// Prepares all source regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, view construction, alpha-channel extraction or insertion, capture
    /// checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn bgr_hls_planar_to_ac4(self, source: &PlanarImageView<'_, u8, 4>) -> Result<Self> {
        self.color_convert_from_planar(source, color::bgr_to_hls_u8_p4_to_ac4)
    }

    /// Converts HLS color values to BGR color values. The destination alpha channel is not processed.
    ///
    /// Plane order follows the named color components.
    ///
    /// Reads the supplied `source` planes and allocates a new packed AC4 image at their
    /// full-resolution dimensions. The current pipeline supplies the stream context and
    /// workspace; its existing image pixels are not conversion input. Copies the fourth
    /// source plane into the output alpha channel after native color conversion.
    /// Uses [`hls_to_bgr_u8_p4_to_ac4`](crate::image::raw::color::hls_to_bgr_u8_p4_to_ac4).
    /// Prepares all source regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, view construction, alpha-channel extraction or insertion, capture
    /// checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn hls_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 4>) -> Result<Self> {
        self.color_convert_from_planar(source, color::hls_to_bgr_u8_p4_to_ac4)
    }

    fn color_convert_from_planar(
        self,
        source: &PlanarImageView<'_, u8, 4>,
        operation: Planar4ToPackedColorConvert,
    ) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination = Image::<u8, AC4>::create_uninit(
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
                unsafe {
                    operation(self.stream_context, source, &mut destination_view)?;
                    let mut rgba = destination_view.output().with_layout::<C4>()?;
                    exchange::insert_channel_c4(
                        self.stream_context,
                        &source.planes()[3],
                        &mut rgba,
                        3,
                    )
                }
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
