use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, filtering::*};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DoubleSeparableFilterImage<T, L>,
{
    pub fn filter_column64f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()> {
        <Self as DoubleSeparableFilterImage<T, L>>::filter_column64f_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
        )
    }

    pub fn filter_row64f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()> {
        <Self as DoubleSeparableFilterImage<T, L>>::filter_row64f_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: DoubleSeparableFilterImage<T, L>,
{
    pub fn filter_column64f(self, kernel: &[f64], anchor: i32) -> Result<Self> {
        self.double_separable_filter(
            kernel,
            anchor,
            <Self as DoubleSeparableFilterImage<T, L>>::filter_column64f_image,
        )
    }

    pub fn filter_row64f(self, kernel: &[f64], anchor: i32) -> Result<Self> {
        self.double_separable_filter(
            kernel,
            anchor,
            <Self as DoubleSeparableFilterImage<T, L>>::filter_row64f_image,
        )
    }

    fn double_separable_filter(
        self,
        kernel: &[f64],
        anchor: i32,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            &[f64],
            i32,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
                kernel,
                anchor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
