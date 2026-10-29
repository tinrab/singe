use crate::{
    context::StreamContext,
    error::Result,
    image::{SupportedImage, color, memory::Image, view::C4},
    pipeline::image::{
        planar::SubsampledPlanarImage,
        planar_subsampled_packed_color::SubsampledPlanarToPackedColorConvertConstantAlpha,
    },
};

impl SubsampledPlanarImage<u8> {
    /// 3 channel planar `YCrCb420` to 4 channel packed RGB color conversion with constant alpha.
    ///
    /// Writes `alpha` to every destination alpha component.
    ///
    /// Uses [`ycrcb420_to_rgb_u8_p3_to_c4`](crate::image::raw::color::ycrcb420_to_rgb_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycrcb420_to_rgb_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycrcb420_to_rgb_u8_p3_to_c4,
            alpha,
        )
    }

    /// 3 channel planar `YCbCr411` to 4 channel packed RGB color conversion with constant alpha.
    ///
    /// Writes `alpha` to every destination alpha component.
    ///
    /// Uses [`ycbcr411_to_rgb_u8_p3_to_c4`](crate::image::raw::color::ycbcr411_to_rgb_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr411_to_rgb_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr411_to_rgb_u8_p3_to_c4,
            alpha,
        )
    }

    /// 3 channel planar `YCbCr411` to 4 channel packed BGR color conversion with constant alpha.
    ///
    /// Writes `alpha` to every destination alpha component.
    ///
    /// Uses [`ycbcr411_to_bgr_u8_p3_to_c4`](crate::image::raw::color::ycbcr411_to_bgr_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr411_to_bgr_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr411_to_bgr_u8_p3_to_c4,
            alpha,
        )
    }

    /// 3 channel planar `YCbCr420` to 4 channel packed BGR color conversion with constant alpha.
    ///
    /// Writes `alpha` to every destination alpha component.
    ///
    /// Uses [`ycbcr420_to_bgr_u8_p3_to_c4`](crate::image::raw::color::ycbcr420_to_bgr_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr420_to_bgr_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr420_to_bgr_u8_p3_to_c4,
            alpha,
        )
    }

    /// 3 channel planar `YCbCr420` to 4 channel packed `BGR_709HDTV` color conversion with constant alpha.
    ///
    /// Writes `alpha` to every destination alpha component.
    ///
    /// Uses [`ycbcr420_to_bgr_709hdtv_u8_p3_to_c4`](crate::image::raw::color::ycbcr420_to_bgr_709hdtv_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn ycbcr420_to_bgr_709hdtv_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr420_to_bgr_709hdtv_u8_p3_to_c4,
            alpha,
        )
    }

    pub(crate) fn color_convert_to_packed_with_alpha(
        self,
        stream_context: &StreamContext,
        operation: SubsampledPlanarToPackedColorConvertConstantAlpha,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = Image::create_uninit(stream_context.cuda_context(), self.y.size())?;

        {
            let y = self.y.view()?;
            let cb = self.cb.view()?;
            let cr = self.cr.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &y, &cb, &cr, &mut destination_view, alpha) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
