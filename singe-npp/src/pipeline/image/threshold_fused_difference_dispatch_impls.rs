use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, threshold_dispatch::FusedAbsoluteDifferenceThresholdImage},
};

unsafe impl<T, L> FusedAbsoluteDifferenceThresholdImage<T, L> for ImagePipeline<'_, T, L>
where
    T: npp_threshold::FusedAbsDiffElement,
    L: npp_threshold::FusedAbsDiffChannels,
{
    fn fused_absolute_difference_threshold_greater_value_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source1)?;
        stream_context.prepare_region(source2)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            npp_threshold::fused_absolute_difference_threshold_greater_value(
                stream_context,
                source1,
                source2,
                destination,
                threshold,
                value,
            )
        })
    }

    fn fused_absolute_difference_threshold_greater_value_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source_destination)?;
        stream_context.prepare_region(source2)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            npp_threshold::fused_absolute_difference_threshold_greater_value_in_place(
                stream_context,
                source_destination,
                source2,
                threshold,
                value,
            )
        })
    }
}
