use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::AlphaOperation,
};

use super::{ImageBacking, ImagePipeline, operation_traits::*};

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: ColorKeyImage<L>,
    <Self as ColorKeyImage<L>>::ColorKey: Copy,
{
    pub fn comp_color_key_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, u8, L>,
        right: &ImageView<'_, u8, L>,
        color_key: <Self as ColorKeyImage<L>>::ColorKey,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as ColorKeyImage<L>>::comp_color_key_image(
            stream_context,
            left,
            right,
            color_key,
            destination,
        )
    }
}

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Self: AlphaCompColorKeyImage,
{
    pub fn alpha_comp_color_key_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        alpha: u8,
        other: &ImageView<'_, u8, AC4>,
        other_alpha: u8,
        color_key: [u8; 4],
        destination: &mut ImageViewMut<'_, u8, AC4>,
        operation: AlphaOperation,
    ) -> Result<()> {
        <Self as AlphaCompColorKeyImage>::alpha_comp_color_key_image(
            stream_context,
            source,
            alpha,
            other,
            other_alpha,
            color_key,
            destination,
            operation,
        )
    }
}

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Workspace: ImageAllocator<u8, AC4>,
    Self: AlphaCompColorKeyImage,
{
    pub fn alpha_comp_color_key(
        self,
        alpha: u8,
        other: &ImageView<'_, u8, AC4>,
        other_alpha: u8,
        color_key: [u8; 4],
        operation: AlphaOperation,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<u8, AC4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AlphaCompColorKeyImage>::alpha_comp_color_key_image(
                self.stream_context,
                &source,
                alpha,
                other,
                other_alpha,
                color_key,
                &mut destination_view,
                operation,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, L>,
    Self: ColorKeyImage<L>,
    <Self as ColorKeyImage<L>>::ColorKey: Copy,
{
    pub fn comp_color_key(
        self,
        other: &ImageView<'_, u8, L>,
        color_key: <Self as ColorKeyImage<L>>::ColorKey,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<u8, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ColorKeyImage<L>>::comp_color_key_image(
                self.stream_context,
                &source,
                other,
                color_key,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
