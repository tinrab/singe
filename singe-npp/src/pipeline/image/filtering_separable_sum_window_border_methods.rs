use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::super::{ImageBacking, ImagePipeline, filtering::*};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SumWindowBorderFilterImage<T, L>,
{
    pub fn sum_window_column_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as SumWindowBorderFilterImage<T, L>>::sum_window_column_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }

    pub fn sum_window_row_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as SumWindowBorderFilterImage<T, L>>::sum_window_row_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<f32, L>,
    Self: SumWindowBorderFilterImage<T, L>,
{
    pub fn sum_window_column_border(
        self,
        source_offset: Point,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as SumWindowBorderFilterImage<T, L>>::sum_window_column_border_image,
        )
    }

    pub fn sum_window_row_border(
        self,
        source_offset: Point,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as SumWindowBorderFilterImage<T, L>>::sum_window_row_border_image,
        )
    }

    fn sum_window_border_filter(
        self,
        source_offset: Point,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, f32, L>,
            i32,
            i32,
            BorderType,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        let mut destination = self.workspace.image::<f32, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                anchor,
                border_type,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
