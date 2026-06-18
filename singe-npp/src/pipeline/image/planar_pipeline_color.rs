use crate::{
    error::Result,
    image::{
        color,
        view::{C1, C3, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::planar::PlanarImage;
use super::{ImageBacking, ImagePipeline};

#[path = "planar_pipeline_ac4_color.rs"]
mod ac4_color;
#[path = "planar_pipeline_c2_color.rs"]
mod c2_color;
#[path = "planar_pipeline_subsampled_color.rs"]
mod subsampled_color;

type PackedToPlanarColorConvert = for<'source, 'destination> fn(
    &crate::context::StreamContext,
    &ImageView<'source, u8, C3>,
    &mut PlanarImageViewMut<'destination, u8, 3>,
) -> Result<()>;

type PlanarToPackedColorConvert = for<'source, 'destination> fn(
    &crate::context::StreamContext,
    &PlanarImageView<'source, u8, 3>,
    &mut ImageViewMut<'destination, u8, C3>,
) -> Result<()>;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn rgb_to_yuv_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::rgb_to_yuv_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::rgb_to_ycbcr_u8_c3_to_p3)
    }

    pub fn bgr_to_yuv_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_yuv_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_ycbcr_u8_c3_to_p3)
    }

    pub fn bgr_to_hls_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_hls_u8_c3_to_p3)
    }

    pub fn hls_to_bgr_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::hls_to_bgr_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr444_jpeg_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::rgb_to_ycbcr444_jpeg_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr444_jpeg_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::bgr_to_ycbcr444_jpeg_u8_c3_to_p3)
    }

    fn color_convert_to_planar(
        self,
        operation: PackedToPlanarColorConvert,
    ) -> Result<PlanarImage<u8, 3>> {
        let mut destination = PlanarImage::<u8, 3>::create(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }
}

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    pub fn yuv_planar_to_rgb(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::yuv_to_rgb_u8_p3_to_c3)
    }

    pub fn yuv_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::yuv_to_bgr_u8_p3_to_c3)
    }

    pub fn ycbcr_planar_to_rgb(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr_to_rgb_u8_p3_to_c3)
    }

    pub fn ycbcr_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr_to_bgr_u8_p3_to_c3)
    }

    pub fn ycbcr444_jpeg_planar_to_rgb(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr444_to_rgb_jpeg_u8_p3_to_c3)
    }

    pub fn ycbcr444_jpeg_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 3>) -> Result<Self> {
        self.color_convert_from_planar(source, color::ycbcr444_to_bgr_jpeg_u8_p3_to_c3)
    }

    fn color_convert_from_planar(
        self,
        source: &PlanarImageView<'_, u8, 3>,
        operation: PlanarToPackedColorConvert,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<u8, C3>(source.planes()[0].size())?;

        {
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
