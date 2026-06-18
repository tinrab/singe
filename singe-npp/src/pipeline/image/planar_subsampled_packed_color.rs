use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        memory::Image,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::planar::{SubsampledPlanarImage, subsampled_size};

#[path = "planar_subsampled_jpeg_packed_color.rs"]
mod jpeg_packed_color;
#[path = "planar_subsampled_ycbcr_packed_color.rs"]
mod ycbcr_packed_color;

pub(super) type SubsampledPlanarToPackedColorConvert<L> =
    for<'y, 'cb, 'cr, 'destination> fn(
        &StreamContext,
        &ImageView<'y, u8, C1>,
        &ImageView<'cb, u8, C1>,
        &ImageView<'cr, u8, C1>,
        &mut ImageViewMut<'destination, u8, L>,
    ) -> Result<()>;

pub(super) type SubsampledPlanarToPackedColorConvertConstantAlpha =
    for<'y, 'cb, 'cr, 'destination> fn(
        &StreamContext,
        &ImageView<'y, u8, C1>,
        &ImageView<'cb, u8, C1>,
        &ImageView<'cr, u8, C1>,
        &mut ImageViewMut<'destination, u8, C4>,
        u8,
    ) -> Result<()>;

impl SubsampledPlanarImage<u8> {
    pub fn create(
        size: Size,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
    ) -> Result<Self>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        let chroma_size = subsampled_size(size, horizontal_subsampling, vertical_subsampling)?;
        Ok(Self {
            y: Workspace::create_image(size)?,
            cb: Workspace::create_image(chroma_size)?,
            cr: Workspace::create_image(chroma_size)?,
        })
    }

    pub fn size(&self) -> Size {
        self.y.size()
    }

    pub fn chroma_size(&self) -> Size {
        self.cb.size()
    }

    pub fn yuv420_to_rgb_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_rgb_u8_p3_to_c3)
    }

    pub fn yuv422_to_rgb_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::yuv422_to_rgb_u8_p3_to_c3)
    }

    pub fn yuv420_to_bgr_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_bgr_u8_p3_to_c3)
    }

    pub fn yuv422_to_rgb_ac4(self, stream_context: &StreamContext) -> Result<Image<u8, AC4>>
    where
        Workspace: ImageAllocator<u8, AC4>,
    {
        self.color_convert_to_packed(stream_context, color::yuv422_to_rgb_u8_p3_to_ac4)
    }

    pub fn yuv420_to_rgb_ac4(self, stream_context: &StreamContext) -> Result<Image<u8, AC4>>
    where
        Workspace: ImageAllocator<u8, AC4>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_rgb_u8_p3_to_ac4)
    }

    pub fn yuv420_to_rgb_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_rgb_u8_p3_to_c4)
    }

    pub fn yuv420_to_bgr_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed(stream_context, color::yuv420_to_bgr_u8_p3_to_c4)
    }

    pub(super) fn color_convert_to_packed<L>(
        self,
        stream_context: &StreamContext,
        operation: SubsampledPlanarToPackedColorConvert<L>,
    ) -> Result<Image<u8, L>>
    where
        L: ChannelLayout,
        Workspace: ImageAllocator<u8, L>,
    {
        let mut destination = Workspace::create_image(self.y.size())?;

        {
            let y = self.y.view()?;
            let cb = self.cb.view()?;
            let cr = self.cr.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &y, &cb, &cr, &mut destination_view)?;
        }

        Ok(destination)
    }
}
