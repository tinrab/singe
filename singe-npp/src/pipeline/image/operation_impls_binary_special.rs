use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, operation_traits::*};

macro_rules! impl_unscaled_out_of_place_binary_arithmetic_image {
    (
        $ty:ty,
        $layout:ty,
        $add:path,
        $subtract:path,
        $multiply:path,
        $divide:path
    ) => {
        impl<'a> UnscaledBinaryArithmeticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn add_unscaled_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $add(stream_context, left, right, destination)
            }

            fn subtract_unscaled_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination)
            }

            fn multiply_unscaled_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination)
            }

            fn divide_unscaled_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $divide(stream_context, left, right, destination)
            }
        }
    };
}

macro_rules! impl_scaled_subtract_image {
    ($ty:ty, $layout:ty, $subtract:path, $subtract_in_place:path) => {
        impl<'a> ScaledSubtractImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn subtract_scaled_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination, scale_factor)
            }

            fn subtract_scaled_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, source, source_destination, scale_factor)
            }
        }
    };
}

#[path = "operation_impls_binary_special_divide_round.rs"]
mod divide_round;
#[path = "operation_impls_binary_special_multiply_scale.rs"]
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
