use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

macro_rules! impl_unscaled_binary_arithmetic_image {
    (
        $ty:ty,
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
        impl<'a> BinaryArithmeticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn add_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, left, right, destination)
            }

            fn add_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, source, source_destination)
            }

            fn subtract_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination)
            }

            fn subtract_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, source, source_destination)
            }

            fn multiply_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination)
            }

            fn multiply_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, source, source_destination)
            }

            fn divide_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, left, right, destination)
            }

            fn divide_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, source, source_destination)
            }
        }
    };
}

#[path = "operation_impls_float_binary_unscaled_ac4.rs"]
mod ac4_impls;
#[path = "operation_impls_float_binary_unscaled_c4.rs"]
mod c4_impls;
#[path = "operation_impls_f16_binary_unscaled.rs"]
mod f16_impls;

impl_unscaled_binary_arithmetic_image!(
    Complex32,
    C1,
    arithmetic::add_f32_complex_c1,
    arithmetic::add_f32_complex_c1_in_place,
    arithmetic::subtract_f32_complex_c1,
    arithmetic::subtract_f32_complex_c1_in_place,
    arithmetic::multiply_f32_complex_c1,
    arithmetic::multiply_f32_complex_c1_in_place,
    arithmetic::divide_f32_complex_c1,
    arithmetic::divide_f32_complex_c1_in_place
);
impl_unscaled_binary_arithmetic_image!(
    Complex32,
    C3,
    arithmetic::add_f32_complex_c3,
    arithmetic::add_f32_complex_c3_in_place,
    arithmetic::subtract_f32_complex_c3,
    arithmetic::subtract_f32_complex_c3_in_place,
    arithmetic::multiply_f32_complex_c3,
    arithmetic::multiply_f32_complex_c3_in_place,
    arithmetic::divide_f32_complex_c3,
    arithmetic::divide_f32_complex_c3_in_place
);
