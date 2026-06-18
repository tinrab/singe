use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        linear,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

type ComplexMagnitudeImageOperation =
    fn(&StreamContext, &ImageView<'_, Complex32, C1>, &mut ImageViewMut<'_, f32, C1>) -> Result<()>;

pub trait MagnitudeImage {
    fn magnitude_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()>;

    fn magnitude_squared_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()>;
}

impl<'a> MagnitudeImage for ImagePipeline<'a, Complex32, C1> {
    fn magnitude_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()> {
        linear::magnitude(stream_context, source, destination)
    }

    fn magnitude_squared_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()> {
        linear::magnitude_squared(stream_context, source, destination)
    }
}

impl<'a> ImagePipeline<'a, Complex32, C1>
where
    Self: MagnitudeImage,
{
    pub fn magnitude_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()> {
        <Self as MagnitudeImage>::magnitude_image(stream_context, source, destination)
    }

    pub fn magnitude_squared_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()> {
        <Self as MagnitudeImage>::magnitude_squared_image(stream_context, source, destination)
    }
}

impl<'a> ImagePipeline<'a, Complex32, C1>
where
    Workspace: ImageAllocator<f32, C1>,
    Self: MagnitudeImage,
{
    pub fn magnitude(self) -> Result<ImagePipeline<'a, f32, C1>> {
        self.linear_to_f32(<Self as MagnitudeImage>::magnitude_image)
    }

    pub fn magnitude_squared(self) -> Result<ImagePipeline<'a, f32, C1>> {
        self.linear_to_f32(<Self as MagnitudeImage>::magnitude_squared_image)
    }

    fn linear_to_f32(
        self,
        operation: ComplexMagnitudeImageOperation,
    ) -> Result<ImagePipeline<'a, f32, C1>> {
        let mut destination = self.workspace.image::<f32, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
