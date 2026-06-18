use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_separable_border_methods.rs"]
mod border_methods;

#[path = "filtering_separable_sum_window_methods.rs"]
mod sum_window_methods;

#[path = "filtering_separable_sum_window_border_methods.rs"]
mod sum_window_border_methods;

#[path = "filtering_separable_double_methods.rs"]
mod double_methods;

#[path = "filtering_float_separable_methods.rs"]
mod float_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IntegerSeparableFilterImage<T, L>,
{
    pub fn filter_column_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()> {
        <Self as IntegerSeparableFilterImage<T, L>>::filter_column_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
            divisor,
        )
    }

    pub fn filter_row_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()> {
        <Self as IntegerSeparableFilterImage<T, L>>::filter_row_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
            divisor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: IntegerSeparableFilterImage<T, L>,
{
    pub fn filter_column(self, kernel: &[i32], anchor: i32, divisor: i32) -> Result<Self> {
        self.integer_separable_filter(
            kernel,
            anchor,
            divisor,
            <Self as IntegerSeparableFilterImage<T, L>>::filter_column_image,
        )
    }

    pub fn filter_row(self, kernel: &[i32], anchor: i32, divisor: i32) -> Result<Self> {
        self.integer_separable_filter(
            kernel,
            anchor,
            divisor,
            <Self as IntegerSeparableFilterImage<T, L>>::filter_row_image,
        )
    }

    fn integer_separable_filter(
        self,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            &[i32],
            i32,
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
                divisor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
