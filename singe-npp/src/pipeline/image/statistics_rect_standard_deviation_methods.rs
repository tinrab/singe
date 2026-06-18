use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        memory::Image,
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::{Rectangle, Size},
};

use super::{ImageBacking, ImagePipeline};

#[path = "statistics_rect_standard_deviation_i32_methods.rs"]
mod i32_methods;
#[path = "statistics_rect_standard_deviation_scaled_methods.rs"]
mod scaled_methods;
#[path = "statistics_rect_standard_deviation_to_methods.rs"]
mod to_methods;

impl<'a> ImagePipeline<'a, f32, C1>
where
    Workspace: ImageAllocator<f32, C1>,
{
    pub fn rect_standard_deviation_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, C1>,
        squared: &ImageView<'_, f64, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        statistics::rect_standard_deviation(stream_context, source, squared, destination, rectangle)
    }

    pub fn rect_standard_deviation(
        mut self,
        squared: &ImageView<'_, f64, C1>,
        rectangle: Rectangle,
    ) -> Result<Self> {
        let destination = self.rect_standard_deviation_with(
            squared,
            rectangle,
            statistics::rect_standard_deviation,
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
    pub(super) fn rect_standard_deviation_with<S, D>(
        &mut self,
        squared: &ImageView<'_, S, C1>,
        rectangle: Rectangle,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, C1>,
            &ImageView<'_, S, C1>,
            &mut ImageViewMut<'_, D, C1>,
            Rectangle,
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
            )?;
        }

        Ok(destination)
    }
}

pub(super) fn rect_standard_deviation_size(source: Size, rectangle: Rectangle) -> Result<Size> {
    if rectangle.x < 0 || rectangle.y < 0 || rectangle.width <= 0 || rectangle.height <= 0 {
        return Err(Error::OutOfRange {
            name: "rectangle".into(),
        });
    }

    Ok(Size {
        width: source
            .width
            .checked_sub(rectangle.x)
            .and_then(|value| value.checked_sub(rectangle.width))
            .ok_or_else(|| Error::OutOfRange {
                name: "rect stddev destination width".into(),
            })?,
        height: source
            .height
            .checked_sub(rectangle.y)
            .and_then(|value| value.checked_sub(rectangle.height))
            .ok_or_else(|| Error::OutOfRange {
                name: "rect stddev destination height".into(),
            })?,
    })
}
