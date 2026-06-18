use crate::{
    error::Result,
    image::{
        color,
        view::{C1, C2, C3, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::planar::SubsampledPlanarImage;
use super::ImagePipeline;

pub(super) type PackedToSubsampledPlanarColorConvert<L> =
    for<'source, 'y, 'cb, 'cr> fn(
        &crate::context::StreamContext,
        &ImageView<'source, u8, L>,
        &mut ImageViewMut<'y, u8, C1>,
        &mut ImageViewMut<'cb, u8, C1>,
        &mut ImageViewMut<'cr, u8, C1>,
    ) -> Result<()>;

pub(super) type Nv12PackedForward<T, L> = for<'source, 'y, 'uv> fn(
    &crate::context::StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'y, T, C1>,
    &mut ImageViewMut<'uv, T, C2>,
    color::ColorTwistMatrix,
) -> Result<()>;

#[path = "planar_pipeline_nv12_color.rs"]
mod nv12_color;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn rgb_to_ycbcr420_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_ycbcr420_jpeg_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr422_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::rgb_to_ycbcr422_jpeg_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr411_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::rgb_to_ycbcr411_jpeg_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr420_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_jpeg_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr422_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::bgr_to_ycbcr422_jpeg_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr411_jpeg_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::bgr_to_ycbcr411_jpeg_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_ycbcr420_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_u8_c3_to_p3)
    }

    pub fn bgr_to_ycrcb420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycrcb420_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr420_709csc_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycbcr420_709csc_u8_c3_to_p3)
    }

    pub fn bgr_to_ycrcb420_709csc_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::bgr_to_ycrcb420_709csc_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr411_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::rgb_to_ycbcr411_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr411_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(4, 1, color::bgr_to_ycbcr411_u8_c3_to_p3)
    }

    pub fn rgb_to_ycbcr422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::rgb_to_ycbcr422_u8_c3_to_p3)
    }

    pub fn bgr_to_ycbcr422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::bgr_to_ycbcr422_u8_c3_to_p3)
    }

    pub fn rgb_to_yuv420_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 2, color::rgb_to_yuv420_u8_c3_to_p3)
    }

    pub fn rgb_to_yuv422_planar(self) -> Result<SubsampledPlanarImage<u8>> {
        self.color_convert_to_subsampled_planar(2, 1, color::rgb_to_yuv422_u8_c3_to_p3)
    }

    fn color_convert_to_subsampled_planar(
        self,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
        operation: PackedToSubsampledPlanarColorConvert<C3>,
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

#[path = "planar_pipeline_ac4_subsampled_color.rs"]
mod ac4_subsampled_color;
#[path = "planar_pipeline_u16_nv12_color.rs"]
mod u16_nv12_color;
