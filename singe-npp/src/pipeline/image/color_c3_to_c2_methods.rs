pub use crate::pipeline::image::color_dispatch::*;
use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, color,
        view::{C2, C3, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
};

impl<'a> ImagePipeline<'a, u8, C3>
where
    u8: SupportedImage<C2>,
{
    /// Converts RGB color values to YUV422 color values.
    ///
    /// Uses [`rgb_to_yuv422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_yuv422_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn rgb_to_yuv422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::rgb_to_yuv422_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Converts RGB color values to `YCbCr422` color values.
    ///
    /// Uses [`rgb_to_ycbcr422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_ycbcr422_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn rgb_to_ycbcr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::rgb_to_ycbcr422_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Converts RGB color values to `YCrCb422` color values.
    ///
    /// Uses [`rgb_to_ycrcb422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_ycrcb422_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn rgb_to_ycrcb422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::rgb_to_ycrcb422_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Converts RGB color values to `CbYCr422` color values.
    ///
    /// Uses [`rgb_to_cbycr422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_cbycr422_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn rgb_to_cbycr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::rgb_to_cbycr422_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Applies forward gamma correction to RGB values and converts them to `CbYCr422`.
    ///
    /// Uses [`rgb_to_cbycr422_gamma_u8_c3_to_c2`](crate::image::raw::color::rgb_to_cbycr422_gamma_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn rgb_to_cbycr422_gamma_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::rgb_to_cbycr422_gamma_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Converts BGR color values to `YCbCr422` color values.
    ///
    /// Uses [`bgr_to_ycbcr422_u8_c3_to_c2`](crate::image::raw::color::bgr_to_ycbcr422_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn bgr_to_ycbcr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::bgr_to_ycbcr422_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Converts BGR color values to `CbYCr422` color values.
    ///
    /// Uses [`bgr_to_cbycr422_709hdtv_u8_c3_to_c2`](crate::image::raw::color::bgr_to_cbycr422_709hdtv_u8_c3_to_c2).
    ///
    /// Writes packed 4:2:2 output into `destination` at the source image dimensions.
    /// Prepares both regions, waiting for producers on other streams, and completes the conversion before returning successfully.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Capture checks, region preparation, producer waits, and stream synchronization can also fail.
    pub fn bgr_to_cbycr422_709hdtv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageOutputView<'_, u8, C2>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::bgr_to_cbycr422_709hdtv_u8_c3_to_c2(stream_context, source, destination)
        })
    }

    /// Converts RGB color values to YUV422 color values.
    ///
    /// Uses [`rgb_to_yuv422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_yuv422_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_yuv422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_yuv422_u8_c3_to_c2)
    }

    /// Converts RGB color values to `YCbCr422` color values.
    ///
    /// Uses [`rgb_to_ycbcr422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_ycbcr422_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycbcr422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_ycbcr422_u8_c3_to_c2)
    }

    /// Converts RGB color values to `YCrCb422` color values.
    ///
    /// Uses [`rgb_to_ycrcb422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_ycrcb422_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_ycrcb422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_ycrcb422_u8_c3_to_c2)
    }

    /// Converts RGB color values to `CbYCr422` color values.
    ///
    /// Uses [`rgb_to_cbycr422_u8_c3_to_c2`](crate::image::raw::color::rgb_to_cbycr422_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_cbycr422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_cbycr422_u8_c3_to_c2)
    }

    /// Applies forward gamma correction to RGB values and converts them to `CbYCr422`.
    ///
    /// Uses [`rgb_to_cbycr422_gamma_u8_c3_to_c2`](crate::image::raw::color::rgb_to_cbycr422_gamma_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn rgb_to_cbycr422_gamma(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_cbycr422_gamma_u8_c3_to_c2)
    }

    /// Converts BGR color values to `YCbCr422` color values.
    ///
    /// Uses [`bgr_to_ycbcr422_u8_c3_to_c2`](crate::image::raw::color::bgr_to_ycbcr422_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_ycbcr422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::bgr_to_ycbcr422_u8_c3_to_c2)
    }

    /// Converts BGR color values to `CbYCr422` color values.
    ///
    /// Uses [`bgr_to_cbycr422_709hdtv_u8_c3_to_c2`](crate::image::raw::color::bgr_to_cbycr422_709hdtv_u8_c3_to_c2).
    ///
    /// Allocates a new owned packed two-component image with the same pixel dimensions.
    /// Prepares the input regions, waiting for producers on other streams, and returns after successful stream completion.
    /// Stream capture is rejected. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
    /// Source and destination image sizes must match.
    /// Returns an error if NPP rejects the image geometry or conversion parameters.
    /// Creating the owned output, constructing views, checking capture state, preparing input regions,
    /// waiting for producers, and synchronizing the stream can also fail.
    pub fn bgr_to_cbycr422_709hdtv(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::bgr_to_cbycr422_709hdtv_u8_c3_to_c2)
    }

    fn color_convert_to_c2(
        self,
        operation: DifferentLayoutColorConvert<u8, C3, C2>,
    ) -> Result<ImagePipeline<'a, u8, C2>> {
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

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed native conversion wrote every active channel; ignored alpha was preserved from input.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
