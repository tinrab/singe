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

use super::{ImageBacking, ImagePipeline, rect_standard_deviation_size};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn rect_standard_deviation_to_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        squared: &ImageView<'_, T::Squared, C1>,
        destination: &mut ImageViewMut<'_, T::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()>
    where
        T: statistics::RectStandardDeviation,
    {
        statistics::rect_standard_deviation(stream_context, source, squared, destination, rectangle)
    }

    pub fn rect_standard_deviation_to(
        self,
        squared: &ImageView<'_, T::Squared, C1>,
        rectangle: Rectangle,
    ) -> Result<ImagePipeline<'a, T::Destination, C1>>
    where
        T: statistics::RectStandardDeviation,
        Workspace: ImageAllocator<T::Destination, C1>,
    {
        let size = rect_standard_deviation_size(self.size(), rectangle)?;
        let mut destination = self.workspace.image::<T::Destination, C1>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            statistics::rect_standard_deviation(
                self.stream_context,
                &source,
                squared,
                &mut destination_view,
                rectangle,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
