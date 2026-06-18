use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, C3, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BayerGridPosition, Rectangle},
};

use super::{
    super::color::{CfaToRgbImage, CfaToRgbaImage},
    ImageBacking, ImagePipeline,
};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn cfa_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageViewMut<'_, T, C3>,
        grid: BayerGridPosition,
    ) -> Result<()>
    where
        Self: CfaToRgbImage<T>,
    {
        <Self as CfaToRgbImage<T>>::cfa_to_rgb_image(
            stream_context,
            source,
            source_roi,
            destination,
            grid,
        )
    }

    pub fn cfa_to_rgba_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageViewMut<'_, T, AC4>,
        grid: BayerGridPosition,
        alpha: T,
    ) -> Result<()>
    where
        Self: CfaToRgbaImage<T>,
    {
        <Self as CfaToRgbaImage<T>>::cfa_to_rgba_image(
            stream_context,
            source,
            source_roi,
            destination,
            grid,
            alpha,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C3>,
    Self: CfaToRgbImage<T>,
{
    pub fn cfa_to_rgb(
        self,
        source_roi: Rectangle,
        grid: BayerGridPosition,
    ) -> Result<ImagePipeline<'a, T, C3>> {
        let mut destination = self.workspace.image::<T, C3>(source_roi.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CfaToRgbImage<T>>::cfa_to_rgb_image(
                self.stream_context,
                &source,
                source_roi,
                &mut destination_view,
                grid,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, AC4>,
    Self: CfaToRgbaImage<T>,
{
    pub fn cfa_to_rgba(
        self,
        source_roi: Rectangle,
        grid: BayerGridPosition,
        alpha: T,
    ) -> Result<ImagePipeline<'a, T, AC4>> {
        let mut destination = self.workspace.image::<T, AC4>(source_roi.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CfaToRgbaImage<T>>::cfa_to_rgba_image(
                self.stream_context,
                &source,
                source_roi,
                &mut destination_view,
                grid,
                alpha,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
