use crate::{
    context::StreamContext,
    error::Result,
    image::{
        memory::Image,
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Rectangle,
};

use super::{ImageBacking, ImagePipeline, rect_standard_deviation_size};

impl<'a> ImagePipeline<'a, i32, C1>
where
    Workspace: ImageAllocator<i32, C1>,
{
    pub fn rect_standard_deviation_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, C1>,
        squared: &ImageView<'_, i32, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<()> {
        statistics::rect_standard_deviation_scaled(
            stream_context,
            source,
            squared,
            destination,
            rectangle,
            scale_factor,
        )
    }

    pub fn rect_standard_deviation_scaled(
        mut self,
        squared: &ImageView<'_, i32, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<Self> {
        let destination = self.rect_standard_deviation_with_scale(
            squared,
            rectangle,
            scale_factor,
            statistics::rect_standard_deviation_scaled,
        )?;
        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    fn rect_standard_deviation_with_scale<S, D>(
        &mut self,
        squared: &ImageView<'_, S, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, C1>,
            &ImageView<'_, S, C1>,
            &mut ImageViewMut<'_, D, C1>,
            Rectangle,
            i32,
        ) -> Result<()>,
    ) -> Result<Image<D, C1>>
    where
        S: Copy,
        D: Copy,
        Workspace: ImageAllocator<D, C1>,
    {
        let size = rect_standard_deviation_size(self.size(), rectangle)?;
        let mut destination = self.workspace.image::<D, C1>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                squared,
                &mut destination_view,
                rectangle,
                scale_factor,
            )?;
        }

        Ok(destination)
    }
}
