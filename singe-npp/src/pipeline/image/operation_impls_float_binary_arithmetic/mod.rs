use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::BinaryArithmeticImage},
};

macro_rules! impl_float_binary_arithmetic_image {
    (
        $layout:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path,
        $divide:path,
        $divide_in_place:path
    ) => {
        unsafe impl<'a> BinaryArithmeticImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn add_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $add(stream_context, left, right, destination) })
            }

            fn add_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $add_in_place(stream_context, source, source_destination)
                })
            }

            fn subtract_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $subtract(stream_context, left, right, destination) })
            }

            fn subtract_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $subtract_in_place(stream_context, source, source_destination)
                })
            }

            fn multiply_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $multiply(stream_context, left, right, destination) })
            }

            fn multiply_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $multiply_in_place(stream_context, source, source_destination)
                })
            }

            fn divide_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $divide(stream_context, left, right, destination) })
            }

            fn divide_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide_in_place(stream_context, source, source_destination)
                })
            }
        }
    };
}

mod ac4_impls;

impl_float_binary_arithmetic_image!(
    C1,
    arithmetic::add_f32_c1,
    arithmetic::add_f32_c1_in_place,
    arithmetic::subtract_f32_c1,
    arithmetic::subtract_f32_c1_in_place,
    arithmetic::multiply_f32_c1,
    arithmetic::multiply_f32_c1_in_place,
    arithmetic::divide_f32_c1,
    arithmetic::divide_f32_c1_in_place
);

impl_float_binary_arithmetic_image!(
    C3,
    arithmetic::add_f32_c3,
    arithmetic::add_f32_c3_in_place,
    arithmetic::subtract_f32_c3,
    arithmetic::subtract_f32_c3_in_place,
    arithmetic::multiply_f32_c3,
    arithmetic::multiply_f32_c3_in_place,
    arithmetic::divide_f32_c3,
    arithmetic::divide_f32_c3_in_place
);

impl_float_binary_arithmetic_image!(
    C4,
    arithmetic::add_f32_c4,
    arithmetic::add_f32_c4_in_place,
    arithmetic::subtract_f32_c4,
    arithmetic::subtract_f32_c4_in_place,
    arithmetic::multiply_f32_c4,
    arithmetic::multiply_f32_c4_in_place,
    arithmetic::divide_f32_c4,
    arithmetic::divide_f32_c4_in_place
);
