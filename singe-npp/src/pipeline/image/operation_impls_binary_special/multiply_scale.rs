use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::MultiplyScaleImage},
};

macro_rules! impl_multiply_scale_image {
    ($ty:ty, $layout:ty, $multiply:path, $multiply_in_place:path) => {
        unsafe impl<'a> MultiplyScaleImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn multiply_scale_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $multiply(stream_context, left, right, destination) })
            }

            fn multiply_scale_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $multiply_in_place(stream_context, source, source_destination)
                })
            }
        }
    };
}

impl_multiply_scale_image!(
    u8,
    C1,
    arithmetic::multiply_scale_u8_c1,
    arithmetic::multiply_scale_u8_c1_in_place
);

impl_multiply_scale_image!(
    u8,
    C3,
    arithmetic::multiply_scale_u8_c3,
    arithmetic::multiply_scale_u8_c3_in_place
);

impl_multiply_scale_image!(
    u8,
    C4,
    arithmetic::multiply_scale_u8_c4,
    arithmetic::multiply_scale_u8_c4_in_place
);

impl_multiply_scale_image!(
    u8,
    AC4,
    arithmetic::multiply_scale_u8_ac4,
    arithmetic::multiply_scale_u8_ac4_in_place
);

impl_multiply_scale_image!(
    u16,
    C1,
    arithmetic::multiply_scale_u16_c1,
    arithmetic::multiply_scale_u16_c1_in_place
);

impl_multiply_scale_image!(
    u16,
    C3,
    arithmetic::multiply_scale_u16_c3,
    arithmetic::multiply_scale_u16_c3_in_place
);

impl_multiply_scale_image!(
    u16,
    C4,
    arithmetic::multiply_scale_u16_c4,
    arithmetic::multiply_scale_u16_c4_in_place
);

impl_multiply_scale_image!(
    u16,
    AC4,
    arithmetic::multiply_scale_u16_ac4,
    arithmetic::multiply_scale_u16_ac4_in_place
);
