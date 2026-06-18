use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::planar::{SemiplanarImage, SubsampledPlanarImage};

type Nv12ToPlanar = for<'source_y, 'source_uv, 'y, 'u, 'v> fn(
    &StreamContext,
    &ImageView<'source_y, u8, C1>,
    &ImageView<'source_uv, u8, C2>,
    &mut ImageViewMut<'y, u8, C1>,
    &mut ImageViewMut<'u, u8, C1>,
    &mut ImageViewMut<'v, u8, C1>,
) -> Result<()>;

impl SemiplanarImage<u8> {
    pub fn nv12_to_yuv420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_yuv420_planar(stream_context, color::nv12_to_yuv420_u8_p2_to_p3)
    }

    fn color_convert_to_yuv420_planar(
        self,
        stream_context: &StreamContext,
        operation: Nv12ToPlanar,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        let mut destination = SubsampledPlanarImage::create(self.y.size(), 2, 2)?;

        {
            let source_y = self.y.view()?;
            let source_uv = self.uv.view()?;
            let mut y = destination.y.view_mut()?;
            let mut u = destination.cb.view_mut()?;
            let mut v = destination.cr.view_mut()?;
            operation(
                stream_context,
                &source_y,
                &source_uv,
                &mut y,
                &mut u,
                &mut v,
            )?;
        }

        Ok(destination)
    }
}
