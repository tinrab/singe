use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BoxFilterImage<T, L>,
{
    pub fn filter_box_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as BoxFilterImage<T, L>>::filter_box_image(
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
    Self: DoubleKernelFilterImage<T, L>,
{
    pub fn filter_kernel64f_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f64],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as DoubleKernelFilterImage<T, L>>::filter_kernel64f_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: DoubleKernelFilterImage<T, L>,
{
    pub fn filter_kernel64f(
        self,
        kernel: &[f64],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as DoubleKernelFilterImage<T, L>>::filter_kernel64f_image(
                self.stream_context,
                &source,
                &mut destination_view,
                kernel,
                kernel_size,
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
