use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, threshold_dispatch::FusedAbsoluteDifferenceThresholdImage};

impl<'a, T, L> FusedAbsoluteDifferenceThresholdImage<T, L> for ImagePipeline<'a, T, L>
where
    T: npp_threshold::FusedAbsDiffElement,
    L: npp_threshold::FusedAbsDiffChannels,
{
    fn fused_absolute_difference_threshold_greater_value_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        npp_threshold::fused_absolute_difference_threshold_greater_value(
            stream_context,
            source1,
            source2,
            destination,
            threshold,
            value,
        )
    }

    fn fused_absolute_difference_threshold_greater_value_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        npp_threshold::fused_absolute_difference_threshold_greater_value_in_place(
            stream_context,
            source_destination,
            source2,
            threshold,
            value,
        )
    }
}
