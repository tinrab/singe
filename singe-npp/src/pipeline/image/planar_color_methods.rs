use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{PlanarImage, SemiplanarImage};

#[path = "planar_ycbcr_color_methods.rs"]
mod ycbcr_color_methods;

pub(super) type PlanarToPlanarColorConvert = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 3>,
    &mut PlanarImageViewMut<'destination, u8, 3>,
) -> Result<()>;

type PlanarColorTwist<T> = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, T, 3>,
    &mut PlanarImageViewMut<'destination, T, 3>,
    color::ColorTwistMatrix,
) -> Result<()>;

impl PlanarImage<u8, 3> {
    pub fn rgb_to_yuv(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::rgb_to_yuv_u8_p3)
    }

    pub fn yuv_to_rgb(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::yuv_to_rgb_u8_p3)
    }

    pub fn bgr_to_yuv(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_yuv_u8_p3)
    }

    pub fn yuv_to_bgr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::yuv_to_bgr_u8_p3)
    }

    pub fn bgr_to_hls(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_hls_u8_p3)
    }

    pub fn hls_to_bgr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::hls_to_bgr_u8_p3)
    }

    pub fn gamma_forward(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::gamma_forward_u8_p3)
    }

    pub fn gamma_inverse(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::gamma_inverse_u8_p3)
    }

    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_u8_p3)
    }

    pub fn rgb_to_nv12(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<SemiplanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        self.color_twist_to_nv12(stream_context, twist, color::rgb_to_nv12_u8_color_twist_p3)
    }

    pub(super) fn color_convert(
        self,
        stream_context: &StreamContext,
        operation: PlanarToPlanarColorConvert,
    ) -> Result<Self> {
        let mut destination = Self::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<u8>,
    ) -> Result<Self> {
        let mut destination = Self::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view, twist)?;
        }

        Ok(destination)
    }
}
