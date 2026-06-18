use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::{FloatSeparableBorderFilterImage, ImageBacking, ImagePipeline};

#[path = "filtering_separable_float_border_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: FloatSeparableBorderFilterImage<T, L>,
{
    pub fn filter_column_border32f(
        self,
        source_offset: Point,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<Self> {
        self.float_separable_border_filter(
            source_offset,
            kernel,
            anchor,
            border_type,
            <Self as FloatSeparableBorderFilterImage<T, L>>::filter_column_border32f_image,
        )
    }

    pub fn filter_row_border32f(
        self,
        source_offset: Point,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<Self> {
        self.float_separable_border_filter(
            source_offset,
            kernel,
            anchor,
            border_type,
            <Self as FloatSeparableBorderFilterImage<T, L>>::filter_row_border32f_image,
        )
    }

    fn float_separable_border_filter(
        self,
        source_offset: Point,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, T, L>,
            &[f32],
            i32,
            BorderType,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
                anchor,
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
