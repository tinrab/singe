use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, C3, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

pub(super) use super::color_dispatch::*;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C2>,
{
    pub fn rgb_to_yuv422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::rgb_to_yuv422_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn rgb_to_ycbcr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::rgb_to_ycbcr422_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn rgb_to_ycrcb422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::rgb_to_ycrcb422_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn rgb_to_cbycr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::rgb_to_cbycr422_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn rgb_to_cbycr422_gamma_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::rgb_to_cbycr422_gamma_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn bgr_to_ycbcr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::bgr_to_ycbcr422_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn bgr_to_cbycr422_709hdtv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::bgr_to_cbycr422_709hdtv_u8_c3_to_c2(stream_context, source, destination)
    }

    pub fn rgb_to_yuv422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_yuv422_u8_c3_to_c2)
    }

    pub fn rgb_to_ycbcr422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_ycbcr422_u8_c3_to_c2)
    }

    pub fn rgb_to_ycrcb422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_ycrcb422_u8_c3_to_c2)
    }

    pub fn rgb_to_cbycr422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_cbycr422_u8_c3_to_c2)
    }

    pub fn rgb_to_cbycr422_gamma(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::rgb_to_cbycr422_gamma_u8_c3_to_c2)
    }

    pub fn bgr_to_ycbcr422(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::bgr_to_ycbcr422_u8_c3_to_c2)
    }

    pub fn bgr_to_cbycr422_709hdtv(self) -> Result<ImagePipeline<'a, u8, C2>> {
        self.color_convert_to_c2(color::bgr_to_cbycr422_709hdtv_u8_c3_to_c2)
    }

    fn color_convert_to_c2(
        self,
        operation: DifferentLayoutColorConvert<u8, C3, C2>,
    ) -> Result<ImagePipeline<'a, u8, C2>> {
        let mut destination = self.workspace.image::<u8, C2>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
