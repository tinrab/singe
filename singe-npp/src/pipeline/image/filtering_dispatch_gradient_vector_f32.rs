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
    f32,
    C1,
    f32,
    filtering::gradient_vector_prewitt_border_f32_to_f32_c1,
    filtering::gradient_vector_scharr_border_f32_to_f32_c1,
    filtering::gradient_vector_sobel_border_f32_to_f32_c1
);
impl_gradient_vector_border_image!(
    f32,
    C3,
    f32,
    filtering::gradient_vector_prewitt_border_f32_c3_to_f32_c1,
    filtering::gradient_vector_scharr_border_f32_c3_to_f32_c1,
    filtering::gradient_vector_sobel_border_f32_c3_to_f32_c1
);
