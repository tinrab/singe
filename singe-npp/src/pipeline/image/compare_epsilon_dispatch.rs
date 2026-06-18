use crate::{
    context::StreamContext,
    error::Result,
    image::{
        compare,
        view::{AC4, C1, C3, C4, ImageView, MaskViewMut},
    },
};

use super::ImagePipeline;

pub trait CompareEqualEpsilonImage<L> {
    fn compare_equal_epsilon_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, f32, L>,
        source2: &ImageView<'_, f32, L>,
        destination: &mut MaskViewMut<'_>,
        epsilon: f32,
    ) -> Result<()>;
}

pub trait CompareEqualEpsilonConstantImage<L> {
    type Constant;

    fn compare_equal_epsilon_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        constant: Self::Constant,
        destination: &mut MaskViewMut<'_>,
        epsilon: f32,
    ) -> Result<()>;
}

macro_rules! impl_compare_equal_epsilon_image {
    ($layout:ty, $compare:path) => {
        impl<'a> CompareEqualEpsilonImage<$layout> for ImagePipeline<'a, f32, $layout> {
            fn compare_equal_epsilon_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, f32, $layout>,
                source2: &ImageView<'_, f32, $layout>,
                destination: &mut MaskViewMut<'_>,
                epsilon: f32,
            ) -> Result<()> {
                $compare(stream_context, source1, source2, destination, epsilon)
            }
        }
    };
}

macro_rules! impl_compare_equal_epsilon_constant_image {
    ($layout:ty, $constant:ty, $compare:path) => {
        impl<'a> CompareEqualEpsilonConstantImage<$layout> for ImagePipeline<'a, f32, $layout> {
            type Constant = $constant;

            fn compare_equal_epsilon_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: Self::Constant,
                destination: &mut MaskViewMut<'_>,
                epsilon: f32,
            ) -> Result<()> {
                $compare(stream_context, source, constant, destination, epsilon)
            }
        }
    };
}

impl_compare_equal_epsilon_image!(C1, compare::compare_equal_epsilon_f32_c1);
impl_compare_equal_epsilon_image!(C3, compare::compare_equal_epsilon_f32_c3);
impl_compare_equal_epsilon_image!(C4, compare::compare_equal_epsilon_f32_c4);
impl_compare_equal_epsilon_image!(AC4, compare::compare_equal_epsilon_f32_ac4);

impl_compare_equal_epsilon_constant_image!(C1, f32, compare::compare_equal_epsilon_constant_f32_c1);
impl_compare_equal_epsilon_constant_image!(
    C3,
    [f32; 3],
    compare::compare_equal_epsilon_constant_f32_c3
);
impl_compare_equal_epsilon_constant_image!(
    C4,
    [f32; 4],
    compare::compare_equal_epsilon_constant_f32_c4
);
impl_compare_equal_epsilon_constant_image!(
    AC4,
    [f32; 3],
    compare::compare_equal_epsilon_constant_f32_ac4
);
