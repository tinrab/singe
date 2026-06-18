use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::{ImageBacking, ImagePipeline, TypedFloatKernelFilterImage};

impl<'a, S, SL> ImagePipeline<'a, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    pub fn filter_kernel32f_border_to_into<D, DL>(
        stream_context: &StreamContext,
        source: &ImageView<'_, S, SL>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        DL: ChannelLayout,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            kernel_size,
            anchor,
            border_type,
        )
    }
}

impl<'a, S, SL> ImagePipeline<'a, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    pub fn filter_kernel32f_border_to<D, DL>(
        self,
        source_offset: Point,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, D, DL>>
    where
        D: Copy,
        DL: ChannelLayout,
        Workspace: ImageAllocator<D, DL>,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        let mut destination = self.workspace.image::<D, DL>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_border_to_image(
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

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
