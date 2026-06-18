use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Rectangle,
};

use super::{ImageBacking, ImagePipeline};

impl<'a> ImagePipeline<'a, i32, C1>
where
    Workspace: ImageAllocator<f32, C1>,
{
    pub fn rect_standard_deviation_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, C1>,
        squared: &ImageView<'_, f64, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        statistics::rect_standard_deviation_to(
            stream_context,
            source,
            squared,
            destination,
            rectangle,
        )
    }

    pub fn rect_standard_deviation_to_f32(
        mut self,
        squared: &ImageView<'_, f64, C1>,
        rectangle: Rectangle,
    ) -> Result<ImagePipeline<'a, f32, C1>> {
        let destination = self.rect_standard_deviation_with(
            squared,
            rectangle,
            statistics::rect_standard_deviation_to,
        )?;
        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
