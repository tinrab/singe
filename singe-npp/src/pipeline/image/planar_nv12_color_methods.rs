use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{PlanarImage, SemiplanarImage};

type Nv12PlanarForward<T> = for<'r, 'g, 'b, 'y, 'uv> fn(
    &StreamContext,
    &ImageView<'r, T, C1>,
    &ImageView<'g, T, C1>,
    &ImageView<'b, T, C1>,
    &mut ImageViewMut<'y, T, C1>,
    &mut ImageViewMut<'uv, T, C2>,
    color::ColorTwistMatrix,
) -> Result<()>;

impl<T> PlanarImage<T, 3>
where
    T: Copy,
{
    pub(super) fn color_twist_to_nv12(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: Nv12PlanarForward<T>,
    ) -> Result<SemiplanarImage<T>>
    where
        Workspace: ImageAllocator<T, C1> + ImageAllocator<T, C2>,
    {
        let mut destination = SemiplanarImage::create_nv12(self.planes[0].size())?;

        {
            let source_r = self.planes[0].view()?;
            let source_g = self.planes[1].view()?;
            let source_b = self.planes[2].view()?;
            let mut destination_y = destination.y.view_mut()?;
            let mut destination_uv = destination.uv.view_mut()?;
            operation(
                stream_context,
                &source_r,
                &source_g,
                &source_b,
                &mut destination_y,
                &mut destination_uv,
                twist,
            )?;
        }

        Ok(destination)
    }
}
