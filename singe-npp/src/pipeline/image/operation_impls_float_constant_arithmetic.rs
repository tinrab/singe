use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::ConstantArithmeticImage},
};

macro_rules! impl_float_constant_arithmetic_image {
    (
        $layout:ty,
        $constant_ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path,
        $divide:path,
        $divide_in_place:path
    ) => {
        unsafe impl<'a> ConstantArithmeticImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            type Constant = $constant_ty;

            fn add_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: Self::Constant,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $add(stream_context, source, constant, destination) })
            }

            fn add_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $add_in_place(stream_context, constant, source_destination)
                })
            }

            fn subtract_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: Self::Constant,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $subtract(stream_context, source, constant, destination)
                })
            }

            fn subtract_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $subtract_in_place(stream_context, constant, source_destination)
                })
            }

            fn multiply_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: Self::Constant,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $multiply(stream_context, source, constant, destination)
                })
            }

            fn multiply_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $multiply_in_place(stream_context, constant, source_destination)
                })
            }

            fn divide_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: Self::Constant,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide(stream_context, source, constant, destination)
                })
            }

            fn divide_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide_in_place(stream_context, constant, source_destination)
                })
            }
        }
    };
}

mod ac4_impls;
mod c4_impls;

impl_float_constant_arithmetic_image!(
    C1,
    f32,
    arithmetic::add_constant_f32_c1,
    arithmetic::add_constant_f32_c1_in_place,
    arithmetic::subtract_constant_f32_c1,
    arithmetic::subtract_constant_f32_c1_in_place,
    arithmetic::multiply_constant_f32_c1,
    arithmetic::multiply_constant_f32_c1_in_place,
    arithmetic::divide_constant_f32_c1,
    arithmetic::divide_constant_f32_c1_in_place
);

impl_float_constant_arithmetic_image!(
    C3,
    [f32; 3],
    arithmetic::add_constant_f32_c3,
    arithmetic::add_constant_f32_c3_in_place,
    arithmetic::subtract_constant_f32_c3,
    arithmetic::subtract_constant_f32_c3_in_place,
    arithmetic::multiply_constant_f32_c3,
    arithmetic::multiply_constant_f32_c3_in_place,
    arithmetic::divide_constant_f32_c3,
    arithmetic::divide_constant_f32_c3_in_place
);
