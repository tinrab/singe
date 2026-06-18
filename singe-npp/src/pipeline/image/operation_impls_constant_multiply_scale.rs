use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::MultiplyConstantScaleImage};

macro_rules! impl_multiply_constant_scale_image {
    ($ty:ty, $layout:ty, $constant_ty:ty, $multiply:path, $multiply_in_place:path) => {
        impl<'a> MultiplyConstantScaleImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn multiply_constant_scale_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $multiply(stream_context, source, constant, destination)
            }

            fn multiply_constant_scale_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $multiply_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

impl_multiply_constant_scale_image!(
    u8,
    C1,
    u8,
    arithmetic::multiply_constant_scale_u8_c1,
    arithmetic::multiply_constant_scale_u8_c1_in_place
);
impl_multiply_constant_scale_image!(
    u8,
    C3,
    [u8; 3],
    arithmetic::multiply_constant_scale_u8_c3,
    arithmetic::multiply_constant_scale_u8_c3_in_place
);
impl_multiply_constant_scale_image!(
    u8,
    C4,
    [u8; 4],
    arithmetic::multiply_constant_scale_u8_c4,
    arithmetic::multiply_constant_scale_u8_c4_in_place
);
impl_multiply_constant_scale_image!(
    u8,
    AC4,
    [u8; 3],
    arithmetic::multiply_constant_scale_u8_ac4,
    arithmetic::multiply_constant_scale_u8_ac4_in_place
);
impl_multiply_constant_scale_image!(
    u16,
    C1,
    u16,
    arithmetic::multiply_constant_scale_u16_c1,
    arithmetic::multiply_constant_scale_u16_c1_in_place
);
impl_multiply_constant_scale_image!(
    u16,
    C3,
    [u16; 3],
    arithmetic::multiply_constant_scale_u16_c3,
    arithmetic::multiply_constant_scale_u16_c3_in_place
);
impl_multiply_constant_scale_image!(
    u16,
    C4,
    [u16; 4],
    arithmetic::multiply_constant_scale_u16_c4,
    arithmetic::multiply_constant_scale_u16_c4_in_place
);
impl_multiply_constant_scale_image!(
    u16,
    AC4,
    [u16; 3],
    arithmetic::multiply_constant_scale_u16_ac4,
    arithmetic::multiply_constant_scale_u16_ac4_in_place
);
