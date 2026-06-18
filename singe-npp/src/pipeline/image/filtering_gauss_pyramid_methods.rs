use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::GaussPyramidBorderFilterImage};

#[path = "filtering_gauss_pyramid_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: GaussPyramidBorderFilterImage<T, L>,
{
    pub fn filter_gauss_pyramid_layer_down_border(
        self,
        source_offset: Point,
        destination_size: Size,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<Self> {
        self.filter_gauss_pyramid_border_with(
            source_offset,
            destination_size,
            rate,
            kernel,
            border_type,
            <Self as GaussPyramidBorderFilterImage<
                T,
                L,
            >>::filter_gauss_pyramid_layer_down_border_image,
        )
    }

    pub fn filter_gauss_pyramid_layer_up_border(
        self,
        source_offset: Point,
        destination_size: Size,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<Self> {
        self.filter_gauss_pyramid_border_with(
            source_offset,
            destination_size,
            rate,
            kernel,
            border_type,
            <Self as GaussPyramidBorderFilterImage<
                T,
                L,
            >>::filter_gauss_pyramid_layer_up_border_image,
        )
    }

    fn filter_gauss_pyramid_border_with(
        self,
        source_offset: Point,
        destination_size: Size,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, T, L>,
            f32,
            &[f32],
            BorderType,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(destination_size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                rate,
                kernel,
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
