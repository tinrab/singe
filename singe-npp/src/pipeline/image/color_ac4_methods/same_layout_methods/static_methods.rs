use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, ImageOutputView, ImageView},
    },
    pipeline::ImagePipeline,
};

mod yuv_static_methods;

impl ImagePipeline<'_, u8, AC4> {
    /// Converts RGB color values to XYZ color values. The destination alpha channel is not processed.
    ///
    /// Uses [`rgb_to_xyz_u8_ac4`](crate::image::raw::color::rgb_to_xyz_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_xyz_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::rgb_to_xyz_u8_ac4(stream_context, source, destination) })
    }

    /// Converts XYZ color values to RGB color values. The destination alpha channel is not processed.
    ///
    /// Uses [`xyz_to_rgb_u8_ac4`](crate::image::raw::color::xyz_to_rgb_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn xyz_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::xyz_to_rgb_u8_ac4(stream_context, source, destination) })
    }

    /// Converts RGB color values to LUV color values. The destination alpha channel is not processed.
    ///
    /// Uses [`rgb_to_luv_u8_ac4`](crate::image::raw::color::rgb_to_luv_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_luv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::rgb_to_luv_u8_ac4(stream_context, source, destination) })
    }

    /// Converts LUV color values to RGB color values. The destination alpha channel is not processed.
    ///
    /// Uses [`luv_to_rgb_u8_ac4`](crate::image::raw::color::luv_to_rgb_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn luv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::luv_to_rgb_u8_ac4(stream_context, source, destination) })
    }

    /// Converts RGB color values to HSV color values. The destination alpha channel is not processed.
    ///
    /// Uses [`rgb_to_hsv_u8_ac4`](crate::image::raw::color::rgb_to_hsv_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_hsv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::rgb_to_hsv_u8_ac4(stream_context, source, destination) })
    }

    /// Converts HSV color values to RGB color values. The destination alpha channel is not processed.
    ///
    /// Uses [`hsv_to_rgb_u8_ac4`](crate::image::raw::color::hsv_to_rgb_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn hsv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::hsv_to_rgb_u8_ac4(stream_context, source, destination) })
    }

    /// Converts RGB color values to HLS color values. The destination alpha channel is not processed.
    ///
    /// Uses [`rgb_to_hls_u8_ac4`](crate::image::raw::color::rgb_to_hls_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_hls_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::rgb_to_hls_u8_ac4(stream_context, source, destination) })
    }

    /// Converts HLS color values to RGB color values. The destination alpha channel is not processed.
    ///
    /// Uses [`hls_to_rgb_u8_ac4`](crate::image::raw::color::hls_to_rgb_u8_ac4).
    ///
    /// Writes the converted color channels into `destination`.
    /// Destination alpha is not processed and retains its previous initialization state; successful conversion does not initialize alpha.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn hls_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageOutputView<'_, u8, AC4>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { color::hls_to_rgb_u8_ac4(stream_context, source, destination) })
    }
}
