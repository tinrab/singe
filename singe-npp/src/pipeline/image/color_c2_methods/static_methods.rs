use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, C3, ImageOutputView, ImageView},
    },
    pipeline::ImagePipeline,
};

mod to_c4_methods;

impl ImagePipeline<'_, u8, C2> {
    /// Converts YUV422 color values to RGB color values.
    ///
    /// Writes converted values into `destination` using [`yuv422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::yuv422_to_rgb_u8_c2_to_c3).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn yuv422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::yuv422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
        })
    }

    /// Converts `YCbCr422` color values to RGB color values.
    ///
    /// Writes converted values into `destination` using [`ycbcr422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::ycbcr422_to_rgb_u8_c2_to_c3).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn ycbcr422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycbcr422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
        })
    }

    /// Converts `YCrCb422` color values to RGB color values.
    ///
    /// Writes converted values into `destination` using [`ycrcb422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::ycrcb422_to_rgb_u8_c2_to_c3).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn ycrcb422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycrcb422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
        })
    }

    /// Converts `CbYCr422` color values to RGB color values.
    ///
    /// Writes converted values into `destination` using [`cbycr422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::cbycr422_to_rgb_u8_c2_to_c3).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn cbycr422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::cbycr422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
        })
    }

    /// Converts `YCbCr422` color values to BGR color values.
    ///
    /// Writes converted values into `destination` using [`ycbcr422_to_bgr_u8_c2_to_c3`](crate::image::raw::color::ycbcr422_to_bgr_u8_c2_to_c3).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn ycbcr422_to_bgr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycbcr422_to_bgr_u8_c2_to_c3(stream_context, source, destination)
        })
    }

    /// Converts `CbYCr422` color values to BGR color values.
    ///
    /// Writes converted values into `destination` using [`cbycr422_to_bgr_709hdtv_u8_c2_to_c3`](crate::image::raw::color::cbycr422_to_bgr_709hdtv_u8_c2_to_c3).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn cbycr422_to_bgr_709hdtv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C3>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::cbycr422_to_bgr_709hdtv_u8_c2_to_c3(stream_context, source, destination)
        })
    }

    /// Converts `YCbCr422` color values to `YCrCb422` color values.
    ///
    /// Writes converted values into `destination` using [`ycbcr422_to_ycrcb422_u8_c2`](crate::image::raw::color::ycbcr422_to_ycrcb422_u8_c2).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn ycbcr422_to_ycrcb422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycbcr422_to_ycrcb422_u8_c2(stream_context, source, destination)
        })
    }

    /// Converts `YCbCr422` color values to `CbYCr422` color values.
    ///
    /// Writes converted values into `destination` using [`ycbcr422_to_cbycr422_u8_c2`](crate::image::raw::color::ycbcr422_to_cbycr422_u8_c2).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn ycbcr422_to_cbycr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::ycbcr422_to_cbycr422_u8_c2(stream_context, source, destination)
        })
    }

    /// Converts `CbYCr422` color values to `YCbCr422` color values.
    ///
    /// Writes converted values into `destination` using [`cbycr422_to_ycbcr422_u8_c2`](crate::image::raw::color::cbycr422_to_ycbcr422_u8_c2).
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn cbycr422_to_ycbcr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::cbycr422_to_ycbcr422_u8_c2(stream_context, source, destination)
        })
    }
}
