use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C3, ImageOutputView, ImageView},
    },
    pipeline::ImagePipeline,
};

impl ImagePipeline<'_, u8, C3> {
    /// Converts RGB color values to YUV color values.
    ///
    /// Uses [`rgb_to_yuv_u8_c3`](crate::image::raw::color::rgb_to_yuv_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_yuv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::rgb_to_yuv_u8_c3(stream_context, source, destination) })
    }

    /// Converts YUV color values to RGB color values.
    ///
    /// Uses [`yuv_to_rgb_u8_c3`](crate::image::raw::color::yuv_to_rgb_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn yuv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::yuv_to_rgb_u8_c3(stream_context, source, destination) })
    }

    /// Converts RGB color values to YCbCr color values.
    ///
    /// Uses [`rgb_to_ycbcr_u8_c3`](crate::image::raw::color::rgb_to_ycbcr_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_ycbcr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::rgb_to_ycbcr_u8_c3(stream_context, source, destination)
        })
    }

    /// Converts YCbCr color values to RGB color values.
    ///
    /// Uses [`ycbcr_to_rgb_u8_c3`](crate::image::raw::color::ycbcr_to_rgb_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn ycbcr_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycbcr_to_rgb_u8_c3(stream_context, source, destination)
        })
    }

    /// Converts RGB color values to YCC color values.
    ///
    /// Uses [`rgb_to_ycc_u8_c3`](crate::image::raw::color::rgb_to_ycc_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_ycc_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::rgb_to_ycc_u8_c3(stream_context, source, destination) })
    }

    /// Converts YCC color values to RGB color values.
    ///
    /// Uses [`ycc_to_rgb_u8_c3`](crate::image::raw::color::ycc_to_rgb_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn ycc_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::ycc_to_rgb_u8_c3(stream_context, source, destination) })
    }

    /// Converts BGR color values to YUV color values.
    ///
    /// Uses [`bgr_to_yuv_u8_c3`](crate::image::raw::color::bgr_to_yuv_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn bgr_to_yuv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::bgr_to_yuv_u8_c3(stream_context, source, destination) })
    }

    /// Converts YUV color values to BGR color values.
    ///
    /// Uses [`yuv_to_bgr_u8_c3`](crate::image::raw::color::yuv_to_bgr_u8_c3).
    ///
    /// Writes the converted color channels into `destination`.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn yuv_to_bgr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::yuv_to_bgr_u8_c3(stream_context, source, destination) })
    }
}
