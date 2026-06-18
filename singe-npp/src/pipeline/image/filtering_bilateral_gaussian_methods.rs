use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::{BilateralGaussBorderFilterImage, ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BilateralGaussBorderFilterImage<T, L>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn filter_bilateral_gauss_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as BilateralGaussBorderFilterImage<T, L>>::filter_bilateral_gauss_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            radius,
            step_between_source_pixels,
            value_square_sigma,
            position_square_sigma,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: BilateralGaussBorderFilterImage<T, L>,
{
    pub fn filter_bilateral_gauss_border(
        self,
        source_offset: Point,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as BilateralGaussBorderFilterImage<T, L>>::filter_bilateral_gauss_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                radius,
                step_between_source_pixels,
                value_square_sigma,
                position_square_sigma,
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
