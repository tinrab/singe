use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, DifferentialKernel, MaskSize, Point},
};

use super::{HarrisCornersBorderFilterImage, ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HarrisCornersBorderFilterImage<T, L>,
{
    pub fn filter_harris_corners_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HarrisCornersBorderFilterImage<T, L>>::filter_harris_corners_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            average_window_size,
            k,
            scale,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<f32, C1>,
    Self: HarrisCornersBorderFilterImage<T, L>,
{
    pub fn filter_harris_corners_border(
        self,
        source_offset: Point,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, f32, C1>> {
        let mut destination = self.workspace.image::<f32, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as HarrisCornersBorderFilterImage<T, L>>::filter_harris_corners_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                filter_type,
                mask_size,
                average_window_size,
                k,
                scale,
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
