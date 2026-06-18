use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, ImageView, ImageViewMut},
    },
};

use super::super::{
    ImagePipeline,
    operation_traits::{LeftShiftConstantImage, RightShiftConstantImage},
};

macro_rules! impl_right_shift_constant_image {
    ($ty:ty, $layout:ty, $constant_ty:ty, $shift:path, $shift_in_place:path) => {
        impl<'a> RightShiftConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn right_shift_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $shift(stream_context, source, constant, destination)
            }

            fn right_shift_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $shift_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

macro_rules! impl_left_shift_constant_image {
    ($ty:ty, $layout:ty, $constant_ty:ty, $shift:path, $shift_in_place:path) => {
        impl<'a> LeftShiftConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn left_shift_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $shift(stream_context, source, constant, destination)
            }

            fn left_shift_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $shift_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

macro_rules! impl_shift_constant_image {
    (
        $ty:ty,
        $layout:ty,
        $constant_ty:ty,
        $right_shift:path,
        $right_shift_in_place:path,
        $left_shift:path,
        $left_shift_in_place:path
    ) => {
        impl_right_shift_constant_image!(
            $ty,
            $layout,
            $constant_ty,
            $right_shift,
            $right_shift_in_place
        );
        impl_left_shift_constant_image!(
            $ty,
            $layout,
            $constant_ty,
            $left_shift,
            $left_shift_in_place
        );
    };
}

impl_shift_constant_image!(
    u8,
    AC4,
    [u32; 3],
    arithmetic::right_shift_constant_u8_ac4,
    arithmetic::right_shift_constant_u8_ac4_in_place,
    arithmetic::left_shift_constant_u8_ac4,
    arithmetic::left_shift_constant_u8_ac4_in_place
);
impl_shift_constant_image!(
    u16,
    AC4,
    [u32; 3],
    arithmetic::right_shift_constant_u16_ac4,
    arithmetic::right_shift_constant_u16_ac4_in_place,
    arithmetic::left_shift_constant_u16_ac4,
    arithmetic::left_shift_constant_u16_ac4_in_place
);
impl_shift_constant_image!(
    i32,
    AC4,
    [u32; 3],
    arithmetic::right_shift_constant_i32_ac4,
    arithmetic::right_shift_constant_i32_ac4_in_place,
    arithmetic::left_shift_constant_i32_ac4,
    arithmetic::left_shift_constant_i32_ac4_in_place
);
