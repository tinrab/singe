use crate::{
    context::StreamContext,
    error::Result,
    image::{
        compare,
        view::{AC4, C1, C3, C4, ImageView, MaskViewMut},
    },
    types::ComparisonOperation,
};

use super::ImagePipeline;

#[path = "compare_epsilon_dispatch.rs"]
mod epsilon_dispatch;

pub use epsilon_dispatch::{CompareEqualEpsilonConstantImage, CompareEqualEpsilonImage};

pub trait CompareImage<T, L> {
    fn compare_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut MaskViewMut<'_>,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

pub trait CompareConstantImage<T, L> {
    type Constant;

    fn compare_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut MaskViewMut<'_>,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

macro_rules! impl_compare_image {
    ($ty:ty, $layout:ty, $compare:path) => {
        impl<'a> CompareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn compare_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, $ty, $layout>,
                source2: &ImageView<'_, $ty, $layout>,
                destination: &mut MaskViewMut<'_>,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $compare(stream_context, source1, source2, destination, operation)
            }
        }
    };
}

macro_rules! impl_compare_constant_image {
    ($ty:ty, $layout:ty, $constant:ty, $compare:path) => {
        impl<'a> CompareConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant;

            fn compare_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut MaskViewMut<'_>,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $compare(stream_context, source, constant, destination, operation)
            }
        }
    };
}

impl_compare_image!(u8, C1, compare::compare_u8_c1);
impl_compare_image!(u8, C3, compare::compare_u8_c3);
impl_compare_image!(u8, C4, compare::compare_u8_c4);
impl_compare_image!(u8, AC4, compare::compare_u8_ac4);
impl_compare_image!(u16, C1, compare::compare_u16_c1);
impl_compare_image!(u16, C3, compare::compare_u16_c3);
impl_compare_image!(u16, C4, compare::compare_u16_c4);
impl_compare_image!(u16, AC4, compare::compare_u16_ac4);
impl_compare_image!(i16, C1, compare::compare_i16_c1);
impl_compare_image!(i16, C3, compare::compare_i16_c3);
impl_compare_image!(i16, C4, compare::compare_i16_c4);
impl_compare_image!(i16, AC4, compare::compare_i16_ac4);
impl_compare_image!(f32, C1, compare::compare_f32_c1);
impl_compare_image!(f32, C3, compare::compare_f32_c3);
impl_compare_image!(f32, C4, compare::compare_f32_c4);
impl_compare_image!(f32, AC4, compare::compare_f32_ac4);

impl_compare_constant_image!(u8, C1, u8, compare::compare_constant_u8_c1);
impl_compare_constant_image!(u8, C3, [u8; 3], compare::compare_constant_u8_c3);
impl_compare_constant_image!(u8, C4, [u8; 4], compare::compare_constant_u8_c4);
impl_compare_constant_image!(u8, AC4, [u8; 3], compare::compare_constant_u8_ac4);
impl_compare_constant_image!(u16, C1, u16, compare::compare_constant_u16_c1);
impl_compare_constant_image!(u16, C3, [u16; 3], compare::compare_constant_u16_c3);
impl_compare_constant_image!(u16, C4, [u16; 4], compare::compare_constant_u16_c4);
impl_compare_constant_image!(u16, AC4, [u16; 3], compare::compare_constant_u16_ac4);
impl_compare_constant_image!(i16, C1, i16, compare::compare_constant_i16_c1);
impl_compare_constant_image!(i16, C3, [i16; 3], compare::compare_constant_i16_c3);
impl_compare_constant_image!(i16, C4, [i16; 4], compare::compare_constant_i16_c4);
impl_compare_constant_image!(i16, AC4, [i16; 3], compare::compare_constant_i16_ac4);
impl_compare_constant_image!(f32, C1, f32, compare::compare_constant_f32_c1);
impl_compare_constant_image!(f32, C3, [f32; 3], compare::compare_constant_f32_c3);
impl_compare_constant_image!(f32, C4, [f32; 4], compare::compare_constant_f32_c4);
impl_compare_constant_image!(f32, AC4, [f32; 3], compare::compare_constant_f32_ac4);
