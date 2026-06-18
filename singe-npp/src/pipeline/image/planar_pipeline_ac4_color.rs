use crate::{
    error::Result,
    image::{
        color,
        view::{AC4, C1, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::planar::PlanarImage;
use super::{ImageBacking, ImagePipeline};

type PackedToPlanar4ColorConvert = for<'source, 'destination> fn(
    &crate::context::StreamContext,
    &ImageView<'source, u8, AC4>,
    &mut PlanarImageViewMut<'destination, u8, 4>,
) -> Result<()>;

type Planar4ToPackedColorConvert = for<'source, 'destination> fn(
    &crate::context::StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut ImageViewMut<'destination, u8, AC4>,
) -> Result<()>;

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn rgb_to_yuv_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::rgb_to_yuv_u8_ac4_to_p4)
    }

    pub fn bgr_to_yuv_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::bgr_to_yuv_u8_ac4_to_p4)
    }

    pub fn bgr_to_hls_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::bgr_to_hls_u8_ac4_to_p4)
    }

    pub fn hls_to_bgr_planar(self) -> Result<PlanarImage<u8, 4>> {
        self.color_convert_to_planar(color::hls_to_bgr_u8_ac4_to_p4)
    }

    fn color_convert_to_planar(
        self,
        operation: PackedToPlanar4ColorConvert,
    ) -> Result<PlanarImage<u8, 4>> {
        let mut destination = PlanarImage::<u8, 4>::create(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }
}

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Workspace: ImageAllocator<u8, AC4>,
{
    pub fn bgr_hls_planar_to_ac4(self, source: &PlanarImageView<'_, u8, 4>) -> Result<Self> {
        self.color_convert_from_planar(source, color::bgr_to_hls_u8_p4_to_ac4)
    }

    pub fn hls_planar_to_bgr(self, source: &PlanarImageView<'_, u8, 4>) -> Result<Self> {
        self.color_convert_from_planar(source, color::hls_to_bgr_u8_p4_to_ac4)
    }

    fn color_convert_from_planar(
        self,
        source: &PlanarImageView<'_, u8, 4>,
        operation: Planar4ToPackedColorConvert,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<u8, AC4>(source.planes()[0].size())?;

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
