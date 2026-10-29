use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, color,
        view::{C2, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{
            ColorTwistMatrix, ImagePipeline,
            color::{
                DifferentLayoutColorConvert, DifferentLayoutColorConvertConstantAlpha,
                SameLayoutColorConvert,
            },
        },
    },
};

mod same_layout_methods;
mod static_methods;
mod to_c4_methods;

impl<'a> ImagePipeline<'a, u8, C2>
where
    u8: SupportedImage<C2> + SupportedImage<C3> + SupportedImage<C4>,
{
    impl_static_different_layout_twist!(
        "Two channel packed YUV4:2:2 conversion to three channel packed RGB, using a Color Twist to compute the exact color space arithmetic.\n\nIgnored channels and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.\nEach source and destination image pair must have matching sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv422_to_rgb_color_twist_into,
        u8,
        C2,
        C3,
        color::yuv422_to_rgb_u8_color_twist_c2_to_c3
    );

    /// Converts YUV422 color values to RGB color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`yuv422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::yuv422_to_rgb_u8_c2_to_c3).
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
    pub fn yuv422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::yuv422_to_rgb_u8_c2_to_c3)
    }

    /// Converts `YCbCr422` color values to RGB color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`ycbcr422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::ycbcr422_to_rgb_u8_c2_to_c3).
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
    pub fn ycbcr422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::ycbcr422_to_rgb_u8_c2_to_c3)
    }

    /// Converts `YCrCb422` color values to RGB color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`ycrcb422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::ycrcb422_to_rgb_u8_c2_to_c3).
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
    pub fn ycrcb422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::ycrcb422_to_rgb_u8_c2_to_c3)
    }

    /// Converts `CbYCr422` color values to RGB color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`cbycr422_to_rgb_u8_c2_to_c3`](crate::image::raw::color::cbycr422_to_rgb_u8_c2_to_c3).
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
    pub fn cbycr422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::cbycr422_to_rgb_u8_c2_to_c3)
    }

    /// Converts `YCbCr422` color values to BGR color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`ycbcr422_to_bgr_u8_c2_to_c3`](crate::image::raw::color::ycbcr422_to_bgr_u8_c2_to_c3).
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
    pub fn ycbcr422_to_bgr(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::ycbcr422_to_bgr_u8_c2_to_c3)
    }

    /// Converts `CbYCr422` color values to BGR color values.
    ///
    /// Allocates a new owned output image with the same pixel dimensions.
    /// Uses [`cbycr422_to_bgr_709hdtv_u8_c2_to_c3`](crate::image::raw::color::cbycr422_to_bgr_709hdtv_u8_c2_to_c3).
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
    pub fn cbycr422_to_bgr_709hdtv(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::cbycr422_to_bgr_709hdtv_u8_c2_to_c3)
    }

    pub(crate) fn color_convert_same_layout(
        self,
        operation: SameLayoutColorConvert<u8, C2>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<u8, C2>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects full-ROI native color routines with matching element/layout ABIs.
                unsafe { operation(self.stream_context, &source, &mut destination_view) }
            })?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed native conversion wrote every active channel; ignored alpha was preserved from input.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    fn color_convert_to_c3(
        self,
        operation: DifferentLayoutColorConvert<u8, C2, C3>,
    ) -> Result<ImagePipeline<'a, u8, C3>> {
        self.ensure_eager()?;
        let mut destination =
            Image::<u8, C3>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects full-ROI native color routines with matching element/layout ABIs.
                unsafe { operation(self.stream_context, &source, &mut destination_view) }
            })?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed native conversion wrote every active channel; ignored alpha was preserved from input.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn color_convert_to_c4(
        self,
        alpha: u8,
        operation: DifferentLayoutColorConvertConstantAlpha<u8, C2>,
    ) -> Result<ImagePipeline<'a, u8, C4>> {
        self.ensure_eager()?;
        let mut destination =
            Image::<u8, C4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects full-ROI native color routines with matching element/layout ABIs.
                unsafe { operation(self.stream_context, &source, &mut destination_view, alpha) }
            })?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed native conversion wrote every active channel; ignored alpha was preserved from input.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl ImagePipeline<'_, u16, C2> {
    impl_static_different_layout_twist!(
        "Two channel packed YUV4:2:2 conversion to three channel packed RGB, using a Color Twist to compute the exact color space arithmetic.\n\nIgnored channels and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.\nEach source and destination image pair must have matching sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv422_to_rgb_color_twist_into,
        u16,
        C2,
        C3,
        color::yuv422_to_rgb_u16_color_twist_c2_to_c3
    );
}
