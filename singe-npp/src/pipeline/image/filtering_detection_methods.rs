use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, DifferentialKernel, ImageNormalization, MaskSize, Point},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_harris_corners_methods.rs"]
mod harris_corners_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CannyBorderFilterImage<T, L>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn filter_canny_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as CannyBorderFilterImage<T, L>>::filter_canny_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, C1>,
    Self: CannyBorderFilterImage<T, L>,
{
    pub fn filter_canny_border(
        self,
        source_offset: Point,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, u8, C1>> {
        let mut destination = self.workspace.image::<u8, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CannyBorderFilterImage<T, L>>::filter_canny_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                filter_type,
                mask_size,
                low_threshold,
                high_threshold,
                norm,
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
