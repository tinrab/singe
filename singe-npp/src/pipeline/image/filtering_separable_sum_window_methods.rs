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
    Self: SumWindowFilterImage<T, L>,
{
    pub fn sum_window_column_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()> {
        <Self as SumWindowFilterImage<T, L>>::sum_window_column_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }

    pub fn sum_window_row_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()> {
        <Self as SumWindowFilterImage<T, L>>::sum_window_row_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<f32, L>,
    Self: SumWindowFilterImage<T, L>,
{
    pub fn sum_window_column(
        self,
        mask_size: i32,
        anchor: i32,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_filter(
            mask_size,
            anchor,
            <Self as SumWindowFilterImage<T, L>>::sum_window_column_image,
        )
    }

    pub fn sum_window_row(self, mask_size: i32, anchor: i32) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_filter(
            mask_size,
            anchor,
            <Self as SumWindowFilterImage<T, L>>::sum_window_row_image,
        )
    }

    fn sum_window_filter(
        self,
        mask_size: i32,
        anchor: i32,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, f32, L>,
            i32,
            i32,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        let mut destination = self.workspace.image::<f32, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
                anchor,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
