use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::UnaryTranscendentalImage};

impl_scaled_unary_transcendental_image!(
    i16,
    C1,
    arithmetic::natural_logarithm_i16_c1,
    arithmetic::natural_logarithm_i16_c1_in_place,
    arithmetic::exponential_i16_c1,
    arithmetic::exponential_i16_c1_in_place
);
impl_scaled_unary_transcendental_image!(
    i16,
    C3,
    arithmetic::natural_logarithm_i16_c3,
    arithmetic::natural_logarithm_i16_c3_in_place,
    arithmetic::exponential_i16_c3,
    arithmetic::exponential_i16_c3_in_place
);
