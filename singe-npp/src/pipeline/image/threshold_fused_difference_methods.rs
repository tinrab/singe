use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline, threshold_dispatch::FusedAbsoluteDifferenceThresholdImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: FusedAbsoluteDifferenceThresholdImage<T, L>,
{
    pub fn fused_absolute_difference_threshold_greater_value_into(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FusedAbsoluteDifferenceThresholdImage<
            T,
            L,
        >>::fused_absolute_difference_threshold_greater_value_image(
            stream_context,
            source1,
            source2,
            destination,
            threshold,
            value,
        )
    }

    pub fn fused_absolute_difference_threshold_greater_value_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FusedAbsoluteDifferenceThresholdImage<
            T,
            L,
        >>::fused_absolute_difference_threshold_greater_value_image_in_place(
            stream_context,
            source_destination,
            source2,
            threshold,
            value,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: FusedAbsoluteDifferenceThresholdImage<T, L>,
{
    pub fn fused_absolute_difference_threshold_greater_value(
        mut self,
        other: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as FusedAbsoluteDifferenceThresholdImage<
                    T,
                    L,
                >>::fused_absolute_difference_threshold_greater_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    other,
                    threshold,
                    value,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as FusedAbsoluteDifferenceThresholdImage<
                    T,
                    L,
                >>::fused_absolute_difference_threshold_greater_value_image(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    threshold,
                    value,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
