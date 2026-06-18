use crate::{
    error::Result,
    image::{
        color,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImagePipeline, PlanarImage};

type C2ToPlanarColorConvert = for<'source, 'r, 'g, 'b> fn(
    &crate::context::StreamContext,
    &ImageView<'source, u8, C2>,
    &mut ImageViewMut<'r, u8, C1>,
    &mut ImageViewMut<'g, u8, C1>,
    &mut ImageViewMut<'b, u8, C1>,
) -> Result<()>;

impl<'a> ImagePipeline<'a, u8, C2>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn ycbcr422_to_rgb_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::ycbcr422_to_rgb_c2_to_p3)
    }

    pub fn ycrcb422_to_rgb_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::ycrcb422_to_rgb_c2_to_p3)
    }

    fn color_convert_to_planar(
        self,
        operation: C2ToPlanarColorConvert,
    ) -> Result<PlanarImage<u8, 3>> {
        let mut destination = PlanarImage::<u8, 3>::create(self.size())?;

        {
            let source = self.view()?;
            let [destination_r, destination_g, destination_b] = &mut destination.planes;
            let mut destination_r_view = destination_r.view_mut()?;
            let mut destination_g_view = destination_g.view_mut()?;
            let mut destination_b_view = destination_b.view_mut()?;
            operation(
                self.stream_context,
                &source,
                &mut destination_r_view,
                &mut destination_g_view,
                &mut destination_b_view,
            )?;
        }

        Ok(destination)
    }
}
