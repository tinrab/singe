use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
    types::ComplexI16,
};

use super::{ImagePipeline, operation_traits::*};

macro_rules! impl_scaled_binary_arithmetic_image {
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
                scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, left, right, destination, scale_factor)
            }

            fn add_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, source, source_destination, scale_factor)
            }

            fn subtract_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination, scale_factor)
            }

            fn subtract_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, source, source_destination, scale_factor)
            }

            fn multiply_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination, scale_factor)
            }

            fn multiply_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, source, source_destination, scale_factor)
            }

            fn divide_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, left, right, destination, scale_factor)
            }

            fn divide_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, source, source_destination, scale_factor)
            }
        }
    };
}

#[path = "operation_impls_complex_i16_ac4_binary_arithmetic.rs"]
mod complex_i16_ac4_binary_arithmetic;
#[path = "operation_impls_complex_i32_binary_arithmetic.rs"]
mod complex_i32_binary_arithmetic;

impl_scaled_binary_arithmetic_image!(
    ComplexI16,
    C1,
    arithmetic::add_i16_complex_c1,
    arithmetic::add_i16_complex_c1_in_place,
    arithmetic::subtract_i16_complex_c1,
    arithmetic::subtract_i16_complex_c1_in_place,
    arithmetic::multiply_i16_complex_c1,
    arithmetic::multiply_i16_complex_c1_in_place,
    arithmetic::divide_i16_complex_c1,
    arithmetic::divide_i16_complex_c1_in_place
);
impl_scaled_binary_arithmetic_image!(
    ComplexI16,
    C3,
    arithmetic::add_i16_complex_c3,
    arithmetic::add_i16_complex_c3_in_place,
    arithmetic::subtract_i16_complex_c3,
    arithmetic::subtract_i16_complex_c3_in_place,
    arithmetic::multiply_i16_complex_c3,
    arithmetic::multiply_i16_complex_c3_in_place,
    arithmetic::divide_i16_complex_c3,
    arithmetic::divide_i16_complex_c3_in_place
);
