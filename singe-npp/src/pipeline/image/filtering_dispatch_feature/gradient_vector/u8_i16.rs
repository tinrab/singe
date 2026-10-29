use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::GradientVectorBorderImage},
    types::{BorderType, ImageNormalization, MaskSize, Point},
};

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
