use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        memory::Image,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::PlanarImage;

type PlanarToC2ColorConvert = for<'r, 'g, 'b, 'destination> fn(
    &StreamContext,
    &ImageView<'r, u8, C1>,
    &ImageView<'g, u8, C1>,
    &ImageView<'b, u8, C1>,
    &mut ImageViewMut<'destination, u8, C2>,
) -> Result<()>;

impl PlanarImage<u8, 3> {
    pub fn rgb_to_ycbcr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::rgb_to_ycbcr_u8_p3)
    }

    pub fn ycbcr_to_rgb(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycbcr_to_rgb_u8_p3)
    }

    pub fn rgb_to_ycbcr444_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::rgb_to_ycbcr444_jpeg_u8_p3)
    }

    pub fn bgr_to_ycbcr444_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_ycbcr444_jpeg_u8_p3)
    }

    pub fn ycbcr444_to_rgb_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycbcr444_to_rgb_jpeg_u8_p3)
    }

    pub fn ycbcr444_to_bgr_jpeg(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycbcr444_to_bgr_jpeg_u8_p3)
    }

    pub fn rgb_to_ycbcr422(self, stream_context: &StreamContext) -> Result<Image<u8, C2>> {
        self.color_convert_to_c2(stream_context, color::rgb_to_ycbcr422_p3_to_c2)
    }

    pub fn rgb_to_ycrcb422(self, stream_context: &StreamContext) -> Result<Image<u8, C2>> {
        self.color_convert_to_c2(stream_context, color::rgb_to_ycrcb422_p3_to_c2)
    }

    fn color_convert_to_c2(
        self,
        stream_context: &StreamContext,
        operation: PlanarToC2ColorConvert,
    ) -> Result<Image<u8, C2>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        let mut destination = Workspace::create_image(self.planes[0].size())?;

        {
            let source_r = self.planes[0].view()?;
            let source_g = self.planes[1].view()?;
            let source_b = self.planes[2].view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                stream_context,
                &source_r,
                &source_g,
                &source_b,
                &mut destination_view,
            )?;
        }

        Ok(destination)
    }
}
