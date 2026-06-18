use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_bilateral_gaussian_methods.rs"]
mod bilateral_gaussian_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AdvancedGaussFilterImage<T, L>,
{
    pub fn filter_gauss_advanced_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
    ) -> Result<()> {
        <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_image(
            stream_context,
            source,
            destination,
            kernel,
        )
    }

    pub fn filter_gauss_advanced_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()> {
        <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: AdvancedGaussFilterImage<T, L>,
{
    pub fn filter_gauss_advanced(self, kernel: &[f32]) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_image(
                self.stream_context,
                &source,
                &mut destination_view,
                kernel,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub fn filter_gauss_advanced_border(
        self,
        source_offset: Point,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
