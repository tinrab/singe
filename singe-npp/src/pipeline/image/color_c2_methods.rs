use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, C3, C4, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::color::{
    DifferentLayoutColorConvert, DifferentLayoutColorConvertConstantAlpha, SameLayoutColorConvert,
};
use super::{ColorTwistMatrix, ImageBacking, ImagePipeline};

#[path = "color_c2_same_layout_methods.rs"]
mod same_layout_methods;
#[path = "color_c2_static_methods.rs"]
mod static_methods;
#[path = "color_c2_to_c4_methods.rs"]
mod to_c4_methods;

impl<'a> ImagePipeline<'a, u8, C2>
where
    Workspace: ImageAllocator<u8, C2> + ImageAllocator<u8, C3> + ImageAllocator<u8, C4>,
{
    impl_static_different_layout_twist!(
        yuv422_to_rgb_color_twist_into,
        u8,
        C2,
        C3,
        color::yuv422_to_rgb_u8_color_twist_c2_to_c3
    );

    pub fn yuv422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::yuv422_to_rgb_u8_c2_to_c3)
    }

    pub fn ycbcr422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::ycbcr422_to_rgb_u8_c2_to_c3)
    }

    pub fn ycrcb422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::ycrcb422_to_rgb_u8_c2_to_c3)
    }

    pub fn cbycr422_to_rgb(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::cbycr422_to_rgb_u8_c2_to_c3)
    }

    pub fn ycbcr422_to_bgr(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::ycbcr422_to_bgr_u8_c2_to_c3)
    }

    pub fn cbycr422_to_bgr_709hdtv(self) -> Result<ImagePipeline<'a, u8, C3>> {
        self.color_convert_to_c3(color::cbycr422_to_bgr_709hdtv_u8_c2_to_c3)
    }

    pub(super) fn color_convert_same_layout(
        self,
        operation: SameLayoutColorConvert<u8, C2>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<u8, C2>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    fn color_convert_to_c3(
        self,
        operation: DifferentLayoutColorConvert<u8, C2, C3>,
    ) -> Result<ImagePipeline<'a, u8, C3>> {
        let mut destination = self.workspace.image::<u8, C3>(self.size())?;

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

    pub(super) fn color_convert_to_c4(
        self,
        alpha: u8,
        operation: DifferentLayoutColorConvertConstantAlpha<u8, C2>,
    ) -> Result<ImagePipeline<'a, u8, C4>> {
        let mut destination = self.workspace.image::<u8, C4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view, alpha)?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a> ImagePipeline<'a, u16, C2> {
    impl_static_different_layout_twist!(
        yuv422_to_rgb_color_twist_into,
        u16,
        C2,
        C3,
        color::yuv422_to_rgb_u16_color_twist_c2_to_c3
    );
}
