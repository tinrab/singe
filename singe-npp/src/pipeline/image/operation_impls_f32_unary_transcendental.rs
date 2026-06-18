use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::UnaryTranscendentalImage};

impl_float_unary_transcendental_image!(
    C1,
    arithmetic::natural_logarithm_f32_c1,
    arithmetic::natural_logarithm_f32_c1_in_place,
    arithmetic::exponential_f32_c1,
    arithmetic::exponential_f32_c1_in_place
);
impl_float_unary_transcendental_image!(
    C3,
    arithmetic::natural_logarithm_f32_c3,
    arithmetic::natural_logarithm_f32_c3_in_place,
    arithmetic::exponential_f32_c3,
    arithmetic::exponential_f32_c3_in_place
);
