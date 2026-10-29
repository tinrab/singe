use crate::{
    error::Result,
    image::{
        SupportedImage, color,
        view::{C2, C4},
    },
    pipeline::image::ImagePipeline,
};

impl<'a> ImagePipeline<'a, u8, C2>
where
    u8: SupportedImage<C4>,
{
    /// Converts `YCbCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`ycbcr422_to_bgr_u8_c2_to_c4`](crate::image::raw::color::ycbcr422_to_bgr_u8_c2_to_c4).
    /// Sets every output alpha sample to `alpha`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, image or output-view construction, capture checks, preparing regions,
    /// waiting for producers, and stream synchronization can also fail.
    pub fn ycbcr422_to_bgr_c4(self, alpha: u8) -> Result<ImagePipeline<'a, u8, C4>> {
        self.color_convert_to_c4(alpha, color::ycbcr422_to_bgr_u8_c2_to_c4)
    }

    /// Converts `CbYCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`cbycr422_to_bgr_u8_c2_to_c4`](crate::image::raw::color::cbycr422_to_bgr_u8_c2_to_c4).
    /// Sets every output alpha sample to `alpha`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, image or output-view construction, capture checks, preparing regions,
    /// waiting for producers, and stream synchronization can also fail.
    pub fn cbycr422_to_bgr_c4(self, alpha: u8) -> Result<ImagePipeline<'a, u8, C4>> {
        self.color_convert_to_c4(alpha, color::cbycr422_to_bgr_u8_c2_to_c4)
    }

    /// Converts `CbYCr422` color values to BGR color values. Writes `alpha` to every destination alpha component.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`cbycr422_to_bgr_709hdtv_u8_c2_to_c4`](crate::image::raw::color::cbycr422_to_bgr_709hdtv_u8_c2_to_c4).
    /// Sets every output alpha sample to `alpha`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Output allocation, image or output-view construction, capture checks, preparing regions,
    /// waiting for producers, and stream synchronization can also fail.
    pub fn cbycr422_to_bgr_709hdtv_c4(self, alpha: u8) -> Result<ImagePipeline<'a, u8, C4>> {
        self.color_convert_to_c4(alpha, color::cbycr422_to_bgr_709hdtv_u8_c2_to_c4)
    }
}
