use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        memory::Image,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{ImageBacking, ImagePipeline};

#[path = "statistics_integral_concrete_methods.rs"]
mod concrete_methods;
#[path = "statistics_integral_dispatch.rs"]
mod integral_dispatch;
#[path = "statistics_integral_squared_methods.rs"]
mod squared_methods;

use integral_dispatch::{IntegralImage, SquaredIntegralImage};

impl<'a> ImagePipeline<'a, u8, C1> {
    pub fn integral_to_into<T>(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        value: T,
    ) -> Result<()>
    where
        T: Copy,
        Self: IntegralImage<T>,
    {
        <Self as IntegralImage<T>>::integral_image(stream_context, source, destination, value)
    }

    pub fn integral_to<T>(self, value: T) -> Result<ImagePipeline<'a, T, C1>>
    where
        T: Copy,
        Workspace: ImageAllocator<T, C1>,
        Self: IntegralImage<T>,
    {
        let size = integral_size(self.size())?;
        let mut destination = self.workspace.image::<T, C1>(size)?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as IntegralImage<T>>::integral_image(
                self.stream_context,
                &source,
                &mut destination_view,
                value,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub(super) fn integral<T>(
        &mut self,
        value: T,
        operation: fn(
            &StreamContext,
            &ImageView<'_, u8, C1>,
            &mut ImageViewMut<'_, T, C1>,
            T,
        ) -> Result<()>,
    ) -> Result<Image<T, C1>>
    where
        T: Copy,
        Workspace: ImageAllocator<T, C1>,
    {
        let size = integral_size(self.size())?;
        let mut destination = self.workspace.image::<T, C1>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view, value)?;
        }

        Ok(destination)
    }
}

pub(super) fn integral_size(source: Size) -> Result<Size> {
    Ok(Size {
        width: source
            .width
            .checked_add(1)
            .ok_or_else(|| Error::OutOfRange {
                name: "integral destination width".into(),
            })?,
        height: source
            .height
            .checked_add(1)
            .ok_or_else(|| Error::OutOfRange {
                name: "integral destination height".into(),
            })?,
    })
}
