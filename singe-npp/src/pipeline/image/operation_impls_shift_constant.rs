use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{
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

#[path = "operation_impls_ac4_shift_constant.rs"]
mod ac4_shift_constant;
#[path = "operation_impls_i32_shift_constant.rs"]
mod i32_shift_constant;
#[path = "operation_impls_signed_right_shift_constant.rs"]
mod signed_right_shift_constant;
#[path = "operation_impls_u16_shift_constant.rs"]
mod u16_shift_constant;

impl_shift_constant_image!(
    u8,
    C1,
    u32,
    arithmetic::right_shift_constant_u8_c1,
    arithmetic::right_shift_constant_u8_c1_in_place,
    arithmetic::left_shift_constant_u8_c1,
    arithmetic::left_shift_constant_u8_c1_in_place
);
impl_shift_constant_image!(
    u8,
    C3,
    [u32; 3],
    arithmetic::right_shift_constant_u8_c3,
    arithmetic::right_shift_constant_u8_c3_in_place,
    arithmetic::left_shift_constant_u8_c3,
    arithmetic::left_shift_constant_u8_c3_in_place
);
impl_shift_constant_image!(
    u8,
    C4,
    [u32; 4],
    arithmetic::right_shift_constant_u8_c4,
    arithmetic::right_shift_constant_u8_c4_in_place,
    arithmetic::left_shift_constant_u8_c4,
    arithmetic::left_shift_constant_u8_c4_in_place
);
