use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{C1, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, threshold_dispatch::ValueThresholdImage},
    types::ComparisonOperation,
};

macro_rules! impl_value_threshold_image {
    ($ty:ty, $threshold:path, $threshold_in_place:path) => {
        unsafe impl<'a> ValueThresholdImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn threshold_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $threshold(
                        stream_context,
                        source,
                        destination,
                        threshold,
                        value,
                        operation,
                    )
                })
            }

            fn threshold_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $threshold_in_place(stream_context, image, threshold, value, operation)
                })
            }
        }
    };
}

macro_rules! impl_packed_value_threshold_image {
    ($ty:ty, $layout:ty, $channels:literal, $threshold:path, $threshold_in_place:path) => {
        unsafe impl<'a> PackedValueThresholdImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn threshold_channels_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
                operation: ComparisonOperation,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $threshold(
                        stream_context,
                        source,
                        destination,
                        thresholds,
                        values,
                        operation,
                    )
                })
            }

            fn threshold_channels_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
                operation: ComparisonOperation,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $threshold_in_place(stream_context, image, thresholds, values, operation)
                })
            }
        }
    };
}

mod less_greater;
mod packed_value;

impl_value_threshold_image!(
    u8,
    npp_threshold::threshold_value_u8_c1,
    npp_threshold::threshold_value_u8_c1_in_place
);

impl_value_threshold_image!(
    u16,
    npp_threshold::threshold_value_u16_c1,
    npp_threshold::threshold_value_u16_c1_in_place
);

impl_value_threshold_image!(
    i16,
    npp_threshold::threshold_value_i16_c1,
    npp_threshold::threshold_value_i16_c1_in_place
);

impl_value_threshold_image!(
    f32,
    npp_threshold::threshold_value_f32_c1,
    npp_threshold::threshold_value_f32_c1_in_place
);
