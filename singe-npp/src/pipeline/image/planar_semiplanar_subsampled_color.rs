use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::planar::{SemiplanarImage, SubsampledPlanarImage};

#[path = "planar_semiplanar_c2_color.rs"]
mod c2_color;
#[path = "planar_semiplanar_nv12_color.rs"]
mod nv12_color;
#[path = "planar_semiplanar_ycbcr411_color.rs"]
mod ycbcr411_color;

type SemiplanarToSubsampledPlanar = for<'source_y, 'source_uv, 'y, 'cb, 'cr> fn(
    &StreamContext,
    &ImageView<'source_y, u8, C1>,
    &ImageView<'source_uv, u8, C2>,
    &mut ImageViewMut<'y, u8, C1>,
    &mut ImageViewMut<'cb, u8, C1>,
    &mut ImageViewMut<'cr, u8, C1>,
) -> Result<()>;

impl SemiplanarImage<u8> {
    pub fn ycbcr420_to_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(stream_context, 2, 2, color::ycbcr420_u8_p2_to_p3)
    }

    pub fn ycbcr420_to_ycbcr422_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            1,
            color::ycbcr420_to_ycbcr422_u8_p2_to_p3,
        )
    }

    pub fn ycbcr420_to_ycbcr411_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            4,
            1,
            color::ycbcr420_to_ycbcr411_u8_p2_to_p3,
        )
    }

    pub fn ycbcr420_to_ycrcb420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            2,
            color::ycbcr420_to_ycrcb420_u8_p2_to_p3,
        )
    }

    fn color_convert_to_subsampled_planar(
        self,
        stream_context: &StreamContext,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
        operation: SemiplanarToSubsampledPlanar,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        let mut destination = SubsampledPlanarImage::create(
            self.y.size(),
            horizontal_subsampling,
            vertical_subsampling,
        )?;

        {
            let source_y = self.y.view()?;
            let source_uv = self.uv.view()?;
            let mut y = destination.y.view_mut()?;
            let mut cb = destination.cb.view_mut()?;
            let mut cr = destination.cr.view_mut()?;
            operation(
                stream_context,
                &source_y,
                &source_uv,
                &mut y,
                &mut cb,
                &mut cr,
            )?;
        }

        Ok(destination)
    }
}
