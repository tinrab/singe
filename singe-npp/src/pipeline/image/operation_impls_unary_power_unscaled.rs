use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::operation_traits::*;
use super::ImagePipeline;

macro_rules! impl_unscaled_unary_power_image {
    (
        $ty:ty,
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square(stream_context, source, destination)
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square_in_place(stream_context, source_destination)
            }
        }

        impl<'a> SquareRootImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_root_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square_root(stream_context, source, destination)
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square_root_in_place(stream_context, source_destination)
            }
        }
    };
}

impl_unscaled_unary_power_image!(
    f16,
    C1,
    arithmetic::square_f16_c1,
    arithmetic::square_f16_c1_in_place,
    arithmetic::square_root_f16_c1,
    arithmetic::square_root_f16_c1_in_place
);
impl_unscaled_unary_power_image!(
    f16,
    C3,
    arithmetic::square_f16_c3,
    arithmetic::square_f16_c3_in_place,
    arithmetic::square_root_f16_c3,
    arithmetic::square_root_f16_c3_in_place
);
impl_unscaled_unary_power_image!(
    f16,
    C4,
    arithmetic::square_f16_c4,
    arithmetic::square_f16_c4_in_place,
    arithmetic::square_root_f16_c4,
    arithmetic::square_root_f16_c4_in_place
);
