use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::{FloatKernelFilterImage, ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: FloatKernelFilterImage<T, L>,
{
    pub fn filter_kernel32f(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<Self> {
        self.float_kernel_filter(
            kernel,
            kernel_size,
            anchor,
            <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_image,
        )
    }

    pub fn filter_kernel32f_border(
        self,
        source_offset: Point,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
                kernel_size,
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

    fn float_kernel_filter(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            &[f32],
            Size,
            Point,
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
