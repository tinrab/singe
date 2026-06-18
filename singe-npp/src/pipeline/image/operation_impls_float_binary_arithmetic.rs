use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

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
        impl<'a> BinaryArithmeticImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn add_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, left, right, destination)
            }

            fn add_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, source, source_destination)
            }

            fn subtract_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination)
            }

            fn subtract_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, source, source_destination)
            }

            fn multiply_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination)
            }

            fn multiply_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, source, source_destination)
            }

            fn divide_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, f32, $layout>,
                right: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, left, right, destination)
            }

            fn divide_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, source, source_destination)
            }
        }
    };
}

#[path = "operation_impls_float_binary_arithmetic_ac4.rs"]
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
