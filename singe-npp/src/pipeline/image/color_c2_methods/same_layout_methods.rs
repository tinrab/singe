use crate::{
    error::Result,
    image::{SupportedImage, color, view::C2},
    pipeline::image::ImagePipeline,
};

impl ImagePipeline<'_, u8, C2>
where
    u8: SupportedImage<C2>,
{
    /// Converts `YCbCr422` color values to `YCrCb422` color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`ycbcr422_to_ycrcb422_u8_c2`](crate::image::raw::color::ycbcr422_to_ycrcb422_u8_c2).
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
    pub fn ycbcr422_to_ycrcb422(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycbcr422_to_ycrcb422_u8_c2)
    }

    /// Converts `YCbCr422` color values to `CbYCr422` color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`ycbcr422_to_cbycr422_u8_c2`](crate::image::raw::color::ycbcr422_to_cbycr422_u8_c2).
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
    pub fn ycbcr422_to_cbycr422(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycbcr422_to_cbycr422_u8_c2)
    }

    /// Converts `CbYCr422` color values to `YCbCr422` color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`cbycr422_to_ycbcr422_u8_c2`](crate::image::raw::color::cbycr422_to_ycbcr422_u8_c2).
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
    pub fn cbycr422_to_ycbcr422(self) -> Result<Self> {
        self.color_convert_same_layout(color::cbycr422_to_ycbcr422_u8_c2)
    }
}
