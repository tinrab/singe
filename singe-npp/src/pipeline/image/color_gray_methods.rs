use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline,
    color::{ColorToGrayImage, RgbToGrayImage},
};

#[path = "color_cfa_methods.rs"]
mod cfa_methods;
#[path = "color_gradient_gray_methods.rs"]
mod gradient_gray_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: RgbToGrayImage<T, L>,
{
    pub fn rgb_to_gray_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, C1>,
    ) -> Result<()> {
        <Self as RgbToGrayImage<T, L>>::rgb_to_gray_image(stream_context, source, destination)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, C1>,
    Self: RgbToGrayImage<T, L>,
{
    pub fn rgb_to_gray(self) -> Result<ImagePipeline<'a, T, C1>> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as RgbToGrayImage<T, L>>::rgb_to_gray_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn color_to_gray_into<const COEFFICIENTS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, C1>,
        coefficients: &[f32; COEFFICIENTS],
    ) -> Result<()>
    where
        Self: ColorToGrayImage<T, L, COEFFICIENTS>,
    {
        <Self as ColorToGrayImage<T, L, COEFFICIENTS>>::color_to_gray_image(
            stream_context,
            source,
            destination,
            coefficients,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, C1>,
{
    pub fn color_to_gray<const COEFFICIENTS: usize>(
        self,
        coefficients: [f32; COEFFICIENTS],
    ) -> Result<ImagePipeline<'a, T, C1>>
    where
        Self: ColorToGrayImage<T, L, COEFFICIENTS>,
    {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ColorToGrayImage<T, L, COEFFICIENTS>>::color_to_gray_image(
                self.stream_context,
                &source,
                &mut destination_view,
                &coefficients,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
