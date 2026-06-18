use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C3, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{ColorSpace, Point},
};

use super::color::{Uyvp10uToRgbImage, Uyvp10uToRgbOperation};
use super::{ColorTwistMatrix, ImageBacking, ImagePipeline};

#[path = "color_uyvp_ac4_methods.rs"]
mod ac4_methods;
#[path = "color_uyvp_same_type_methods.rs"]
mod same_type_methods;
#[path = "color_uyvp_u16_methods.rs"]
mod u16_methods;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    pub fn uyvp_10u_to_rgb_into<T, L>(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        color_space: ColorSpace,
        alpha: Option<<Self as Uyvp10uToRgbImage<T, L>>::Alpha>,
    ) -> Result<()>
    where
        T: Copy,
        L: ChannelLayout,
        Self: Uyvp10uToRgbImage<T, L>,
    {
        <Self as Uyvp10uToRgbImage<T, L>>::convert(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
            alpha,
        )
    }

    pub fn uyvp_10u_to_rgb<T, L>(
        self,
        source_offset: Point,
        color_space: ColorSpace,
        alpha: Option<<Self as Uyvp10uToRgbImage<T, L>>::Alpha>,
    ) -> Result<ImagePipeline<'a, T, L>>
    where
        T: Copy,
        L: ChannelLayout,
        Workspace: ImageAllocator<T, L>,
        Self: Uyvp10uToRgbImage<T, L>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as Uyvp10uToRgbImage<T, L>>::convert(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                color_space,
                alpha,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub fn uyvp_10u_to_rgba_into<T>(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, AC4>,
        color_space: ColorSpace,
        alpha: <Self as Uyvp10uToRgbImage<T, AC4>>::Alpha,
    ) -> Result<()>
    where
        T: Copy,
        Self: Uyvp10uToRgbImage<T, AC4>,
    {
        Self::uyvp_10u_to_rgb_into(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
            Some(alpha),
        )
    }

    pub fn uyvp_10u_to_rgba<T>(
        self,
        source_offset: Point,
        color_space: ColorSpace,
        alpha: <Self as Uyvp10uToRgbImage<T, AC4>>::Alpha,
    ) -> Result<ImagePipeline<'a, T, AC4>>
    where
        T: Copy,
        Workspace: ImageAllocator<T, AC4>,
        Self: Uyvp10uToRgbImage<T, AC4>,
    {
        self.uyvp_10u_to_rgb::<T, AC4>(source_offset, color_space, Some(alpha))
    }
}
