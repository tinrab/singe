use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, C4, ImageOutputView, ImageView},
    },
    pipeline::ImagePipeline,
};

impl ImagePipeline<'_, u8, C2> {
    /// Converts `YCbCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.
    ///
    /// Writes converted values into `destination` using [`ycbcr422_to_bgr_u8_c2_to_c4`](crate::image::raw::color::ycbcr422_to_bgr_u8_c2_to_c4).
    /// Sets every output alpha sample to `alpha`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn ycbcr422_to_bgr_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C4>,
        alpha: u8,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycbcr422_to_bgr_u8_c2_to_c4(stream_context, source, destination, alpha)
        })
    }

    /// Converts `CbYCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.
    ///
    /// Writes converted values into `destination` using [`cbycr422_to_bgr_u8_c2_to_c4`](crate::image::raw::color::cbycr422_to_bgr_u8_c2_to_c4).
    /// Sets every output alpha sample to `alpha`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn cbycr422_to_bgr_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C4>,
        alpha: u8,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::cbycr422_to_bgr_u8_c2_to_c4(stream_context, source, destination, alpha)
        })
    }

    /// Converts `CbYCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.
    ///
    /// Writes converted values into `destination` using [`cbycr422_to_bgr_709hdtv_u8_c2_to_c4`](crate::image::raw::color::cbycr422_to_bgr_709hdtv_u8_c2_to_c4).
    /// Sets every output alpha sample to `alpha`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn cbycr422_to_bgr_709hdtv_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C4>,
        alpha: u8,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::cbycr422_to_bgr_709hdtv_u8_c2_to_c4(stream_context, source, destination, alpha)
        })
    }
}
