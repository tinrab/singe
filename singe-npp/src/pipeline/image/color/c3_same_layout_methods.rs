use crate::{
    error::Result,
    image::{Image, SupportedImage, color, view::C3},
    pipeline::{
        ImagePipeline,
        image::{ImageBacking, color::SameLayoutColorConvert},
    },
};

mod static_methods;

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C3>,
{
    /// Converts RGB color values to XYZ color values.
    ///
    /// Uses [`rgb_to_xyz_u8_c3`](crate::image::raw::color::rgb_to_xyz_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_xyz(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_xyz_u8_c3)
    }

    /// Converts XYZ color values to RGB color values.
    ///
    /// Uses [`xyz_to_rgb_u8_c3`](crate::image::raw::color::xyz_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn xyz_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::xyz_to_rgb_u8_c3)
    }

    /// Converts RGB color values to LUV color values.
    ///
    /// Uses [`rgb_to_luv_u8_c3`](crate::image::raw::color::rgb_to_luv_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_luv(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_luv_u8_c3)
    }

    /// Converts LUV color values to RGB color values.
    ///
    /// Uses [`luv_to_rgb_u8_c3`](crate::image::raw::color::luv_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn luv_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::luv_to_rgb_u8_c3)
    }

    /// Converts RGB color values to HSV color values.
    ///
    /// Uses [`rgb_to_hsv_u8_c3`](crate::image::raw::color::rgb_to_hsv_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_hsv(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_hsv_u8_c3)
    }

    /// Converts HSV color values to RGB color values.
    ///
    /// Uses [`hsv_to_rgb_u8_c3`](crate::image::raw::color::hsv_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn hsv_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::hsv_to_rgb_u8_c3)
    }

    /// Converts RGB color values to HLS color values.
    ///
    /// Uses [`rgb_to_hls_u8_c3`](crate::image::raw::color::rgb_to_hls_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_hls(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_hls_u8_c3)
    }

    /// Converts HLS color values to RGB color values.
    ///
    /// Uses [`hls_to_rgb_u8_c3`](crate::image::raw::color::hls_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn hls_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::hls_to_rgb_u8_c3)
    }

    /// Converts BGR color values to Lab color values.
    ///
    /// Uses [`bgr_to_lab_u8_c3`](crate::image::raw::color::bgr_to_lab_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn bgr_to_lab(self) -> Result<Self> {
        self.color_convert_same_layout(color::bgr_to_lab_u8_c3)
    }

    /// Converts Lab color values to BGR color values.
    ///
    /// Uses [`lab_to_bgr_u8_c3`](crate::image::raw::color::lab_to_bgr_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn lab_to_bgr(self) -> Result<Self> {
        self.color_convert_same_layout(color::lab_to_bgr_u8_c3)
    }

    /// Converts RGB color values to YUV color values.
    ///
    /// Uses [`rgb_to_yuv_u8_c3`](crate::image::raw::color::rgb_to_yuv_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_yuv(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_yuv_u8_c3)
    }

    /// Converts YUV color values to RGB color values.
    ///
    /// Uses [`yuv_to_rgb_u8_c3`](crate::image::raw::color::yuv_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn yuv_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::yuv_to_rgb_u8_c3)
    }

    /// Converts RGB color values to YCbCr color values.
    ///
    /// Uses [`rgb_to_ycbcr_u8_c3`](crate::image::raw::color::rgb_to_ycbcr_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_ycbcr(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_ycbcr_u8_c3)
    }

    /// Converts YCbCr color values to RGB color values.
    ///
    /// Uses [`ycbcr_to_rgb_u8_c3`](crate::image::raw::color::ycbcr_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn ycbcr_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycbcr_to_rgb_u8_c3)
    }

    /// Converts RGB color values to YCC color values.
    ///
    /// Uses [`rgb_to_ycc_u8_c3`](crate::image::raw::color::rgb_to_ycc_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn rgb_to_ycc(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_ycc_u8_c3)
    }

    /// Converts YCC color values to RGB color values.
    ///
    /// Uses [`ycc_to_rgb_u8_c3`](crate::image::raw::color::ycc_to_rgb_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn ycc_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycc_to_rgb_u8_c3)
    }

    /// Converts BGR color values to YUV color values.
    ///
    /// Uses [`bgr_to_yuv_u8_c3`](crate::image::raw::color::bgr_to_yuv_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn bgr_to_yuv(self) -> Result<Self> {
        self.color_convert_same_layout(color::bgr_to_yuv_u8_c3)
    }

    /// Converts YUV color values to BGR color values.
    ///
    /// Uses [`yuv_to_bgr_u8_c3`](crate::image::raw::color::yuv_to_bgr_u8_c3).
    ///
    /// Allocates a new owned image with the same dimensions and returns it after successful conversion.
    /// Prepares source and destination regions, waiting for producers on other streams, rejects capture, and completes the conversion before returning successfully.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Image allocation, image or output-view construction, CUDA context binding, or region-context validation can fail.
    /// Capture checks, preparing regions, waiting for producers, NPP conversion, and stream synchronization can also fail.
    pub fn yuv_to_bgr(self) -> Result<Self> {
        self.color_convert_same_layout(color::yuv_to_bgr_u8_c3)
    }

    fn color_convert_same_layout(self, operation: SameLayoutColorConvert<u8, C3>) -> Result<Self> {
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed native conversion wrote every active channel; ignored alpha was preserved from input.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
