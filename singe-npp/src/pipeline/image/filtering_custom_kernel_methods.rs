use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_float_kernel_methods.rs"]
mod float_kernel_methods;
#[path = "filtering_custom_kernel_into_methods.rs"]
mod into_methods;
#[path = "filtering_sharpen_unsharp_border_methods.rs"]
mod sharpen_unsharp_border_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: IntegerKernelFilterImage<T, L>,
{
    pub fn filter_kernel(
        self,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
    ) -> Result<Self> {
        self.integer_kernel_filter(
            kernel,
            kernel_size,
            anchor,
            divisor,
            <Self as IntegerKernelFilterImage<T, L>>::filter_kernel_image,
        )
    }

    pub fn filter_kernel_border(
        self,
        source_offset: Point,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as IntegerKernelFilterImage<T, L>>::filter_kernel_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
                kernel_size,
                anchor,
                divisor,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    fn integer_kernel_filter(
        self,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            &[i32],
            Size,
            Point,
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
                kernel_size,
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
