use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageView, ImageViewMut},
    },
    types::{BorderType, ImageNormalization, MaskSize, Point},
};

use super::super::super::{ImagePipeline, filtering_traits::GradientVectorBorderImage};

impl_gradient_vector_border_image!(
    u8,
    C1,
    i16,
    filtering::gradient_vector_prewitt_border_u8_to_i16_f32_c1,
    filtering::gradient_vector_scharr_border_u8_to_i16_f32_c1,
    filtering::gradient_vector_sobel_border_u8_to_i16_f32_c1
);
impl_gradient_vector_border_image!(
    u8,
    C3,
    i16,
    filtering::gradient_vector_prewitt_border_u8_c3_to_i16_f32_c1,
    filtering::gradient_vector_scharr_border_u8_c3_to_i16_f32_c1,
    filtering::gradient_vector_sobel_border_u8_c3_to_i16_f32_c1
);
