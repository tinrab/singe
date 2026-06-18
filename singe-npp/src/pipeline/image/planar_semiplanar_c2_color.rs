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

use super::super::planar::SemiplanarImage;

type SemiplanarToC2 = for<'source_y, 'source_uv, 'destination> fn(
    &StreamContext,
    &ImageView<'source_y, u8, C1>,
    &ImageView<'source_uv, u8, C2>,
    &mut ImageViewMut<'destination, u8, C2>,
) -> Result<()>;

impl SemiplanarImage<u8> {
    pub fn ycbcr420_to_ycbcr422_c2(self, stream_context: &StreamContext) -> Result<Image<u8, C2>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        self.color_convert_to_c2(stream_context, color::ycbcr420_to_ycbcr422_u8_p2_to_c2)
    }

    pub fn ycbcr420_to_cbycr422_c2(self, stream_context: &StreamContext) -> Result<Image<u8, C2>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        self.color_convert_to_c2(stream_context, color::ycbcr420_to_cbycr422_u8_p2_to_c2)
    }

    pub fn ycbcr411_to_ycbcr422_c2(self, stream_context: &StreamContext) -> Result<Image<u8, C2>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        self.color_convert_to_c2(stream_context, color::ycbcr411_to_ycbcr422_u8_p2_to_c2)
    }

    fn color_convert_to_c2(
        self,
        stream_context: &StreamContext,
        operation: SemiplanarToC2,
    ) -> Result<Image<u8, C2>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        let mut destination = Workspace::create_image(self.y.size())?;

        {
            let source_y = self.y.view()?;
            let source_uv = self.uv.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source_y, &source_uv, &mut destination_view)?;
        }

        Ok(destination)
    }
}
