use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline};

impl<'a> ImagePipeline<'a, u8, C1>
where
    Workspace: ImageAllocator<i32, C1>,
{
    pub fn integral_to_i32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
    ) -> Result<()> {
        statistics::integral_to_i32(stream_context, source, destination, value)
    }

    pub fn integral_to_i32(mut self, value: i32) -> Result<ImagePipeline<'a, i32, C1>> {
        let destination = self.integral(value, statistics::integral_to_i32)?;

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a> ImagePipeline<'a, u8, C1>
where
    Workspace: ImageAllocator<f32, C1>,
{
    pub fn integral_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        value: f32,
    ) -> Result<()> {
        statistics::integral_to_f32(stream_context, source, destination, value)
    }

    pub fn integral_to_f32(mut self, value: f32) -> Result<ImagePipeline<'a, f32, C1>> {
        let destination = self.integral(value, statistics::integral_to_f32)?;

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
