use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, operation_traits::*};

macro_rules! impl_absolute_image {
    ($ty:ty, $layout:ty, $absolute:path, $absolute_in_place:path) => {
        impl<'a> AbsoluteImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn absolute_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $absolute(stream_context, source, destination)
            }

            fn absolute_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $absolute_in_place(stream_context, source_destination)
            }
        }
    };
}

macro_rules! impl_absolute_difference_image {
    ($ty:ty, $layout:ty, $absolute_difference:path) => {
        impl<'a> AbsoluteDifferenceImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn absolute_difference_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $absolute_difference(stream_context, left, right, destination)
            }
        }
    };
}

macro_rules! impl_absolute_difference_constant_image {
    ($ty:ty, $absolute_difference_constant:path, $absolute_difference_device_constant:path) => {
        impl<'a> AbsoluteDifferenceConstantImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn absolute_difference_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                constant: $ty,
                destination: &mut ImageViewMut<'_, $ty, C1>,
            ) -> Result<()> {
                $absolute_difference_constant(stream_context, source, constant, destination)
            }

            fn absolute_difference_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                constant: &DeviceMemory<$ty>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
            ) -> Result<()> {
                $absolute_difference_device_constant(stream_context, source, constant, destination)
            }
        }
    };
}

#[path = "operation_impls_absolute_difference_constant.rs"]
mod absolute_difference_constant_impls;
#[path = "operation_impls_absolute_difference.rs"]
mod absolute_difference_impls;
#[path = "operation_impls_f16_unary.rs"]
mod f16_impls;
#[path = "operation_impls_unary_power.rs"]
mod power_impls;
#[path = "operation_impls_unary_power_unscaled.rs"]
mod power_unscaled_impls;

impl_absolute_image!(
    i16,
    C1,
    arithmetic::absolute_i16_c1,
    arithmetic::absolute_i16_c1_in_place
);
impl_absolute_image!(
    i16,
    C3,
    arithmetic::absolute_i16_c3,
    arithmetic::absolute_i16_c3_in_place
);
impl_absolute_image!(
    i16,
    C4,
    arithmetic::absolute_i16_c4,
    arithmetic::absolute_i16_c4_in_place
);
impl_absolute_image!(
    i16,
    AC4,
    arithmetic::absolute_i16_ac4,
    arithmetic::absolute_i16_ac4_in_place
);
impl_absolute_image!(
    f32,
    C1,
    arithmetic::absolute_f32_c1,
    arithmetic::absolute_f32_c1_in_place
);
impl_absolute_image!(
    f32,
    C3,
    arithmetic::absolute_f32_c3,
    arithmetic::absolute_f32_c3_in_place
);
impl_absolute_image!(
    f32,
    C4,
    arithmetic::absolute_f32_c4,
    arithmetic::absolute_f32_c4_in_place
);
impl_absolute_image!(
    f32,
    AC4,
    arithmetic::absolute_f32_ac4,
    arithmetic::absolute_f32_ac4_in_place
);
