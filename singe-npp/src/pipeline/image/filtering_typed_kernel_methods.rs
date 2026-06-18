use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_typed_kernel_border_methods.rs"]
mod border_methods;

impl<'a, S, SL> ImagePipeline<'a, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    pub fn filter_kernel32f_to_into<D, DL>(
        stream_context: &StreamContext,
        source: &ImageView<'_, S, SL>,
        destination: &mut ImageViewMut<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>
    where
        D: Copy,
        DL: ChannelLayout,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_to_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
        )
    }
}

impl<'a, S, SL> ImagePipeline<'a, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    pub fn filter_kernel32f_to<D, DL>(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<ImagePipeline<'a, D, DL>>
    where
        D: Copy,
        DL: ChannelLayout,
        Workspace: ImageAllocator<D, DL>,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        self.typed_float_kernel_filter(
            kernel,
            kernel_size,
            anchor,
            <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_to_image,
        )
    }

    fn typed_float_kernel_filter<D, DL>(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        filter: fn(
            &StreamContext,
            &ImageView<'_, S, SL>,
            &mut ImageViewMut<'_, D, DL>,
            &[f32],
            Size,
            Point,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, D, DL>>
    where
        D: Copy,
        DL: ChannelLayout,
        Workspace: ImageAllocator<D, DL>,
    {
        let mut destination = self.workspace.image::<D, DL>(self.size())?;

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

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
