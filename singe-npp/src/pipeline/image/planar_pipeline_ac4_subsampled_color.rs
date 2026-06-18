use crate::{
    error::Result,
    image::{
        color,
        view::{AC4, C1},
    },
    pipeline::{ImageAllocator, ImagePipeline, SubsampledPlanarImage, Workspace},
};

use super::PackedToSubsampledPlanarColorConvert;

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn bgr_to_yuv420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_yuv420_u8_ac4_to_p3)
    }

    pub fn rgb_to_ycrcb420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_ycrcb420_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycbcr420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycrcb420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycrcb420_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycbcr420_709csc_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_709csc_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycbcr420_709hdtv_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_709hdtv_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycrcb420_709csc_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycrcb420_709csc_u8_ac4_to_p3)
    }

    pub fn rgb_to_ycbcr411_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::rgb_to_ycbcr411_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycbcr411_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::bgr_to_ycbcr411_u8_ac4_to_p3)
    }

    pub fn bgr_to_ycbcr422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::bgr_to_ycbcr422_u8_ac4_to_p3)
    }

    fn color_convert_to_subsampled_planar(
        self,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
        operation: PackedToSubsampledPlanarColorConvert<AC4>,
    ) -> Result<SubsampledPlanarImage<u8>> {
        let mut destination = SubsampledPlanarImage::create(
            self.size(),
            horizontal_subsampling,
            vertical_subsampling,
        )?;

        {
            let source = self.view()?;
            let mut y = destination.y.view_mut()?;
            let mut cb = destination.cb.view_mut()?;
            let mut cr = destination.cr.view_mut()?;
            operation(self.stream_context, &source, &mut y, &mut cb, &mut cr)?;
        }

        Ok(destination)
    }
}
