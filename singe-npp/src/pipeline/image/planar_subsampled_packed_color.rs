use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::context::Context as CudaContext;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        SupportedImage, color, exchange,
        memory::Image,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::planar::{SubsampledPlanarImage, subsampled_size},
    types::Size,
};

mod jpeg_packed_color;
mod ycbcr_packed_color;

/// Native color-conversion callback with the plane and element layouts declared in its signature.
///
/// # Safety
///
/// Callers must uphold the selected raw operation's geometry, initialization, memory-access,
/// and allocation-lifetime requirements. The callback determines completion behavior; see
/// the [raw image execution contract](crate::image::raw).
pub type SubsampledPlanarToPackedColorConvert<L> =
    for<'y, 'cb, 'cr, 'destination> unsafe fn(
        &StreamContext,
        &ImageView<'y, u8, C1>,
        &ImageView<'cb, u8, C1>,
        &ImageView<'cr, u8, C1>,
        &mut ImageOutputView<'destination, u8, L>,
    ) -> Result<()>;

/// Native color-conversion callback with the plane and element layouts declared in its signature.
///
/// # Safety
///
/// Callers must uphold the selected raw operation's geometry, initialization, memory-access,
/// and allocation-lifetime requirements. The callback determines completion behavior; see
/// the [raw image execution contract](crate::image::raw).
pub type SubsampledPlanarToPackedColorConvertConstantAlpha =
    for<'y, 'cb, 'cr, 'destination> unsafe fn(
        &StreamContext,
        &ImageView<'y, u8, C1>,
        &ImageView<'cb, u8, C1>,
        &ImageView<'cr, u8, C1>,
        &mut ImageOutputView<'destination, u8, C4>,
        u8,
    ) -> Result<()>;

impl SubsampledPlanarImage<u8> {
    /// Allocates an uninitialized full-resolution first plane and two separate chroma planes.
    /// Chroma dimensions are the full-resolution dimensions divided by the supplied horizontal
    /// and vertical factors. Does not initialize any samples.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions or subsampling factors are nonpositive, a dimension is not
    /// divisible by its factor, allocation-size arithmetic fails, or allocating a plane fails.
    pub fn create_uninit(
        allocation_context: &Arc<CudaContext>,
        size: Size,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
    ) -> Result<SubsampledPlanarImage<MaybeUninit<u8>>>
    where
        u8: SupportedImage<C1>,
    {
        let chroma_size = subsampled_size(size, horizontal_subsampling, vertical_subsampling)?;
        Ok(SubsampledPlanarImage {
            y: Image::create_uninit(allocation_context, size)?,
            cb: Image::create_uninit(allocation_context, chroma_size)?,
            cr: Image::create_uninit(allocation_context, chroma_size)?,
        })
    }

    /// Returns the dimensions of the full-resolution first plane.
    #[must_use]
    pub fn size(&self) -> Size {
        self.y.size()
    }

    /// Returns the dimensions of the first separately stored chroma plane.
    #[must_use]
    pub fn chroma_size(&self) -> Size {
        self.cb.size()
    }

    /// 3 channel planar YUV420 to 3 channel packed RGB color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv420_to_rgb_u8_p3_to_c3`](crate::image::raw::color::yuv420_to_rgb_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
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
    pub fn yuv420_to_rgb_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_rgb_u8_p3_to_c3, None)
    }

    /// 3 channel planar YUV422 to 3 channel packed RGB color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv422_to_rgb_u8_p3_to_c3`](crate::image::raw::color::yuv422_to_rgb_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn yuv422_to_rgb_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::yuv422_to_rgb_u8_p3_to_c3, None)
    }

    /// 3 channel planar YUV420 to 3 channel packed BGR color conversion.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv420_to_bgr_u8_p3_to_c3`](crate::image::raw::color::yuv420_to_bgr_u8_p3_to_c3).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
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
    pub fn yuv420_to_bgr_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        u8: SupportedImage<C3>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_bgr_u8_p3_to_c3, None)
    }

    /// 3 channel planar YUV422 to 4 channel packed RGB color conversion with alpha.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv422_to_rgb_u8_p3_to_ac4`](crate::image::raw::color::yuv422_to_rgb_u8_p3_to_ac4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`. The helper fills alpha after native color conversion.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    ///
    ///
    /// The explicit alpha-channel fill can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn yuv422_to_rgb_ac4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, AC4>>
    where
        u8: SupportedImage<AC4>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::yuv422_to_rgb_u8_p3_to_ac4,
            Some(alpha),
        )
    }

    /// 3 channel planar YUV420 to 4 channel packed RGB color conversion with alpha.
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv420_to_rgb_u8_p3_to_ac4`](crate::image::raw::color::yuv420_to_rgb_u8_p3_to_ac4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
    /// The returned image has every alpha sample set to `alpha`. The helper fills alpha after native color conversion.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// Source and destination full-resolution dimensions must match.
    /// Chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.
    /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
    /// Chroma plane strides are passed independently.
    ///
    ///
    /// The explicit alpha-channel fill can also fail.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn yuv420_to_rgb_ac4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, AC4>>
    where
        u8: SupportedImage<AC4>,
    {
        self.color_convert_to_packed(
            stream_context,
            color::yuv420_to_rgb_u8_p3_to_ac4,
            Some(alpha),
        )
    }

    /// 3 channel planar YUV420 to 4 channel packed RGB color conversion with constant alpha (0xFF).
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv420_to_rgb_u8_p3_to_c4`](crate::image::raw::color::yuv420_to_rgb_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
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
    pub fn yuv420_to_rgb_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_rgb_u8_p3_to_c4, None)
    }

    /// 3 channel planar YUV420 to 4 channel packed BGR color conversion with constant alpha (0xFF).
    ///
    /// For AC4 destination layouts, the alpha channel is not processed.
    ///
    /// Uses [`yuv420_to_bgr_u8_p3_to_c4`](crate::image::raw::color::yuv420_to_bgr_u8_p3_to_c4).
    /// Allocates an owned packed output image at the full-resolution first-plane dimensions.
    /// Prepares every source region, waiting for producers on other streams.
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
    pub fn yuv420_to_bgr_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        u8: SupportedImage<C4>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_bgr_u8_p3_to_c4, None)
    }

    pub(crate) fn color_convert_to_packed<L>(
        self,
        stream_context: &StreamContext,
        operation: SubsampledPlanarToPackedColorConvert<L>,
        alpha: Option<u8>,
    ) -> Result<Image<u8, L>>
    where
        L: ChannelLayout,
        u8: SupportedImage<L>,
    {
        self.prepare_eager(stream_context)?;
        if L::IGNORES_ALPHA && alpha.is_none() {
            return Err(Error::out_of_range("destination alpha"));
        }

        let mut destination = Image::create_uninit(stream_context.cuda_context(), self.y.size())?;

        {
            let y = self.y.view()?;
            let cb = self.cb.view()?;
            let cr = self.cr.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(stream_context, &y, &cb, &cr, &mut destination_view)?;
                    if let Some(alpha) = alpha {
                        let mut rgba = destination_view.output().with_layout::<C4>()?;
                        exchange::set_channel_c4(stream_context, &mut rgba, alpha, 3)?;
                    }
                    Ok(())
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
