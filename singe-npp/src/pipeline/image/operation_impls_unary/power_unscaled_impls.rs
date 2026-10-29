use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        operation_impls_unary::ImagePipeline,
        operation_traits::{SquareImage, SquareRootImage},
    },
};

macro_rules! impl_unscaled_unary_power_image {
    (
        $ty:ty,
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        unsafe impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $square(stream_context, source, destination) })
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $square_in_place(stream_context, source_destination) })
            }
        }

        unsafe impl<'a> SquareRootImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_root_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $square_root(stream_context, source, destination) })
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_root_in_place(stream_context, source_destination)
                })
            }
        }
    };
}

impl_unscaled_unary_power_image!(
    f16,
    C1,
    arithmetic::square_f16_c1,
    arithmetic::square_f16_c1_in_place,
    arithmetic::square_root_f16_c1,
    arithmetic::square_root_f16_c1_in_place
);

impl_unscaled_unary_power_image!(
    f16,
    C3,
    arithmetic::square_f16_c3,
    arithmetic::square_f16_c3_in_place,
    arithmetic::square_root_f16_c3,
    arithmetic::square_root_f16_c3_in_place
);

impl_unscaled_unary_power_image!(
    f16,
    C4,
    arithmetic::square_f16_c4,
    arithmetic::square_f16_c4_in_place,
    arithmetic::square_root_f16_c4,
    arithmetic::square_root_f16_c4_in_place
);
