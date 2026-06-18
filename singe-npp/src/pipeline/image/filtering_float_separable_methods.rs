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
    Self: FloatSeparableFilterImage<T, L>,
{
    pub fn filter_column32f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
    ) -> Result<()> {
        <Self as FloatSeparableFilterImage<T, L>>::filter_column32f_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
        )
    }

    pub fn filter_row32f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
    ) -> Result<()> {
        <Self as FloatSeparableFilterImage<T, L>>::filter_row32f_image(
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
    Self: FloatSeparableFilterImage<T, L>,
{
    pub fn filter_column32f(self, kernel: &[f32], anchor: i32) -> Result<Self> {
        self.float_separable_filter(
            kernel,
            anchor,
            <Self as FloatSeparableFilterImage<T, L>>::filter_column32f_image,
        )
    }

    pub fn filter_row32f(self, kernel: &[f32], anchor: i32) -> Result<Self> {
        self.float_separable_filter(
            kernel,
            anchor,
            <Self as FloatSeparableFilterImage<T, L>>::filter_row32f_image,
        )
    }

    fn float_separable_filter(
        self,
        kernel: &[f32],
        anchor: i32,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            &[f32],
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
