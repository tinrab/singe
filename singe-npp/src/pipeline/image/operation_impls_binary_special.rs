use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImagePipeline,
        operation_traits::{ScaledSubtractImage, UnscaledBinaryArithmeticImage},
    },
};

macro_rules! impl_unscaled_out_of_place_binary_arithmetic_image {
    (
        $ty:ty,
        $layout:ty,
        $add:path,
        $subtract:path,
        $multiply:path,
        $divide:path
    ) => {
        unsafe impl<'a> UnscaledBinaryArithmeticImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn add_unscaled_image(
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
                    .sync_scope(|| unsafe { $add(stream_context, left, right, destination) })
            }

            fn subtract_unscaled_image(
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
                    .sync_scope(|| unsafe { $subtract(stream_context, left, right, destination) })
            }

            fn multiply_unscaled_image(
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

            fn divide_unscaled_image(
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
                    .sync_scope(|| unsafe { $divide(stream_context, left, right, destination) })
            }
        }
    };
}

macro_rules! impl_scaled_subtract_image {
    ($ty:ty, $layout:ty, $subtract:path, $subtract_in_place:path) => {
        unsafe impl<'a> ScaledSubtractImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn subtract_scaled_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $subtract(stream_context, left, right, destination, scale_factor)
                })
            }

            fn subtract_scaled_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $subtract_in_place(stream_context, source, source_destination, scale_factor)
                })
            }
        }
    };
}

mod divide_round;
mod multiply_scale;

impl_unscaled_out_of_place_binary_arithmetic_image!(
    i32,
    C1,
    arithmetic::add_i32_c1_unscaled,
    arithmetic::subtract_i32_c1_unscaled,
    arithmetic::multiply_i32_c1_unscaled,
    arithmetic::divide_i32_c1_unscaled
);

impl_scaled_subtract_image!(
    i32,
    C4,
    arithmetic::subtract_i32_c4,
    arithmetic::subtract_i32_c4_in_place
);
