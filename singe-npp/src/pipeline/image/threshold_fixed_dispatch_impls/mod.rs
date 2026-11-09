use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{C1, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, threshold_dispatch::FixedThresholdImage},
};

macro_rules! impl_fixed_threshold_image {
    ($ty:ty, $greater:path, $greater_in_place:path, $less:path, $less_in_place:path) => {
        unsafe impl<'a> FixedThresholdImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn threshold_greater_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                threshold: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $greater(stream_context, source, destination, threshold)
                })
            }

            fn threshold_greater_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $greater_in_place(stream_context, image, threshold) })
            }

            fn threshold_less_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                threshold: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $less(stream_context, source, destination, threshold) })
            }

            fn threshold_less_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $less_in_place(stream_context, image, threshold) })
            }
        }
    };
}

mod packed;

impl_fixed_threshold_image!(
    u8,
    npp_threshold::threshold_greater_u8_c1,
    npp_threshold::threshold_greater_u8_c1_in_place,
    npp_threshold::threshold_less_u8_c1,
    npp_threshold::threshold_less_u8_c1_in_place
);

impl_fixed_threshold_image!(
    u16,
    npp_threshold::threshold_greater_u16_c1,
    npp_threshold::threshold_greater_u16_c1_in_place,
    npp_threshold::threshold_less_u16_c1,
    npp_threshold::threshold_less_u16_c1_in_place
);

impl_fixed_threshold_image!(
    i16,
    npp_threshold::threshold_greater_i16_c1,
    npp_threshold::threshold_greater_i16_c1_in_place,
    npp_threshold::threshold_less_i16_c1,
    npp_threshold::threshold_less_i16_c1_in_place
);

impl_fixed_threshold_image!(
    f32,
    npp_threshold::threshold_greater_f32_c1,
    npp_threshold::threshold_greater_f32_c1_in_place,
    npp_threshold::threshold_less_f32_c1,
    npp_threshold::threshold_less_f32_c1_in_place
);
