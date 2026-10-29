use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C2, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::planar::{PlanarImage, SemiplanarImage},
};

mod ycbcr_color_methods;

/// Native color-conversion callback with the plane and element layouts declared in its signature.
///
/// # Safety
///
/// Callers must uphold the selected raw operation's geometry, initialization, memory-access,
/// and allocation-lifetime requirements. The callback determines completion behavior; see
/// the [raw image execution contract](crate::image::raw).
pub type PlanarToPlanarColorConvert = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 3>,
    &mut PlanarImageOutputView<'destination, u8, 3>,
) -> Result<()>;

type PlanarColorTwist<T> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, T, 3>,
    &mut PlanarImageOutputView<'destination, T, 3>,
    color::ColorTwistMatrix,
) -> Result<()>;

impl PlanarImage<u8, 3> {
    /// Converts RGB color values to YUV color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`rgb_to_yuv_u8_p3`](crate::image::raw::color::rgb_to_yuv_u8_p3).
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
    pub fn rgb_to_yuv(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::rgb_to_yuv_u8_p3)
    }

    /// Converts YUV color values to RGB color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`yuv_to_rgb_u8_p3`](crate::image::raw::color::yuv_to_rgb_u8_p3).
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
    pub fn yuv_to_rgb(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::yuv_to_rgb_u8_p3)
    }

    /// Converts BGR color values to YUV color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_yuv_u8_p3`](crate::image::raw::color::bgr_to_yuv_u8_p3).
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
    pub fn bgr_to_yuv(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_yuv_u8_p3)
    }

    /// Converts YUV color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`yuv_to_bgr_u8_p3`](crate::image::raw::color::yuv_to_bgr_u8_p3).
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
    pub fn yuv_to_bgr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::yuv_to_bgr_u8_p3)
    }

    /// Converts BGR color values to HLS color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`bgr_to_hls_u8_p3`](crate::image::raw::color::bgr_to_hls_u8_p3).
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
    pub fn bgr_to_hls(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_hls_u8_p3)
    }

    /// Converts HLS color values to BGR color values.
    ///
    /// Plane order follows the named color components.
    ///
    /// Uses [`hls_to_bgr_u8_p3`](crate::image::raw::color::hls_to_bgr_u8_p3).
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
    pub fn hls_to_bgr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::hls_to_bgr_u8_p3)
    }

    /// 3 channel planar color not in place forward gamma correction.
    ///
    /// Uses [`gamma_forward_u8_p3`](crate::image::raw::color::gamma_forward_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn gamma_forward(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::gamma_forward_u8_p3)
    }

    /// 3 channel planar color not in place inverse gamma correction.
    ///
    /// Uses [`gamma_inverse_u8_p3`](crate::image::raw::color::gamma_inverse_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn gamma_inverse(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::gamma_inverse_u8_p3)
    }

    /// 3 channel planar color twist.
    ///
    /// Uses [`color_twist_u8_p3`](crate::image::raw::color::color_twist_u8_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_u8_p3)
    }

    /// Three channel planar RGB conversion to two channel planar NV12, using a Color Twist to compute the exact color space arithmetic.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
    /// Destination plane strides are passed independently.
    ///
    /// Uses [`rgb_to_nv12_u8_color_twist_p3`](crate::image::raw::color::rgb_to_nv12_u8_color_twist_p3).
    /// Allocates a full-resolution Y plane and an interleaved UV plane with half the width
    /// and half the height. Both full-resolution dimensions must be even.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The three full-resolution source color planes must have equal dimensions and byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn rgb_to_nv12(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<SemiplanarImage<u8>>
    where
        u8: SupportedImage<C2>,
    {
        self.color_twist_to_nv12(stream_context, twist, color::rgb_to_nv12_u8_color_twist_p3)
    }

    pub(crate) fn color_convert(
        self,
        stream_context: &StreamContext,
        operation: PlanarToPlanarColorConvert,
    ) -> Result<Self> {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Self::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<u8>,
    ) -> Result<Self> {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Self::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view, twist) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
