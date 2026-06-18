use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

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

impl_scaled_binary_arithmetic_image!(
    u8,
    AC4,
    arithmetic::add_u8_ac4,
    arithmetic::add_u8_ac4_in_place,
    arithmetic::subtract_u8_ac4,
    arithmetic::subtract_u8_ac4_in_place,
    arithmetic::multiply_u8_ac4,
    arithmetic::multiply_u8_ac4_in_place,
    arithmetic::divide_u8_ac4,
    arithmetic::divide_u8_ac4_in_place
);
impl_scaled_binary_arithmetic_image!(
    u16,
    AC4,
    arithmetic::add_u16_ac4,
    arithmetic::add_u16_ac4_in_place,
    arithmetic::subtract_u16_ac4,
    arithmetic::subtract_u16_ac4_in_place,
    arithmetic::multiply_u16_ac4,
    arithmetic::multiply_u16_ac4_in_place,
    arithmetic::divide_u16_ac4,
    arithmetic::divide_u16_ac4_in_place
);
