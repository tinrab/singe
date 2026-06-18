use crate::{
    context::StreamContext,
    error::Result,
    image::{color, memory::Image, view::C4},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    planar::SubsampledPlanarImage,
    planar_subsampled_packed_color::SubsampledPlanarToPackedColorConvertConstantAlpha,
};

impl SubsampledPlanarImage<u8> {
    pub fn ycrcb420_to_rgb_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycrcb420_to_rgb_u8_p3_to_c4,
            alpha,
        )
    }

    pub fn ycbcr411_to_rgb_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr411_to_rgb_u8_p3_to_c4,
            alpha,
        )
    }

    pub fn ycbcr411_to_bgr_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr411_to_bgr_u8_p3_to_c4,
            alpha,
        )
    }

    pub fn ycbcr420_to_bgr_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr420_to_bgr_u8_p3_to_c4,
            alpha,
        )
    }

    pub fn ycbcr420_to_bgr_709hdtv_c4(
        self,
        stream_context: &StreamContext,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed_with_alpha(
            stream_context,
            color::ycbcr420_to_bgr_709hdtv_u8_p3_to_c4,
            alpha,
        )
    }

    pub(super) fn color_convert_to_packed_with_alpha(
        self,
        stream_context: &StreamContext,
        operation: SubsampledPlanarToPackedColorConvertConstantAlpha,
        alpha: u8,
    ) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        let mut destination = Workspace::create_image(self.y.size())?;

        {
            let y = self.y.view()?;
            let cb = self.cb.view()?;
            let cr = self.cr.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &y, &cb, &cr, &mut destination_view, alpha)?;
        }

        Ok(destination)
    }
}
