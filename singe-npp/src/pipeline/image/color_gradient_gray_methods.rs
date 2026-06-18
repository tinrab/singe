use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C3, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::ImageNormalization,
};

use super::super::color::GradientColorToGrayImage;
use super::{ImageBacking, ImagePipeline};

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: GradientColorToGrayImage<T>,
{
    pub fn gradient_color_to_gray_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageViewMut<'_, T, C1>,
        normalization: ImageNormalization,
    ) -> Result<()> {
        <Self as GradientColorToGrayImage<T>>::gradient_color_to_gray_image(
            stream_context,
            source,
            destination,
            normalization,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: GradientColorToGrayImage<T>,
{
    pub fn gradient_color_to_gray(
        self,
        normalization: ImageNormalization,
    ) -> Result<ImagePipeline<'a, T, C1>> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as GradientColorToGrayImage<T>>::gradient_color_to_gray_image(
                self.stream_context,
                &source,
                &mut destination_view,
                normalization,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
