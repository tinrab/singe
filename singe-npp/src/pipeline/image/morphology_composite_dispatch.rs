use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
    workspace::ScratchBuffer,
};

use super::ImagePipeline;
use super::morphology_traits::{
    CompositeMorphologyBorderImage, CompositeMorphologyBorderScratchImage,
};

#[macro_use]
#[path = "morphology_composite_dispatch_macros.rs"]
mod macros;

impl_composite_morphology_border_image!(
    u8,
    C1,
    morphology::morph_close_border_u8_c1_with_scratch,
    morphology::morph_close_border_u8_c1,
    morphology::morph_open_border_u8_c1_with_scratch,
    morphology::morph_open_border_u8_c1,
    morphology::morph_top_hat_border_u8_c1_with_scratch,
    morphology::morph_top_hat_border_u8_c1,
    morphology::morph_black_hat_border_u8_c1_with_scratch,
    morphology::morph_black_hat_border_u8_c1,
    morphology::morph_gradient_border_u8_c1_with_scratch,
    morphology::morph_gradient_border_u8_c1
);
impl_morph_buffer_size!(u8, C1, morphology::morph_buffer_size_u8_c1);
impl_composite_morphology_border_image!(
    u8,
    C3,
    morphology::morph_close_border_u8_c3_with_scratch,
    morphology::morph_close_border_u8_c3,
    morphology::morph_open_border_u8_c3_with_scratch,
    morphology::morph_open_border_u8_c3,
    morphology::morph_top_hat_border_u8_c3_with_scratch,
    morphology::morph_top_hat_border_u8_c3,
    morphology::morph_black_hat_border_u8_c3_with_scratch,
    morphology::morph_black_hat_border_u8_c3,
    morphology::morph_gradient_border_u8_c3_with_scratch,
    morphology::morph_gradient_border_u8_c3
);
impl_morph_buffer_size!(u8, C3, morphology::morph_buffer_size_u8_c3);
impl_composite_morphology_border_image!(
    u8,
    C4,
    morphology::morph_close_border_u8_c4_with_scratch,
    morphology::morph_close_border_u8_c4,
    morphology::morph_open_border_u8_c4_with_scratch,
    morphology::morph_open_border_u8_c4,
    morphology::morph_top_hat_border_u8_c4_with_scratch,
    morphology::morph_top_hat_border_u8_c4,
    morphology::morph_black_hat_border_u8_c4_with_scratch,
    morphology::morph_black_hat_border_u8_c4,
    morphology::morph_gradient_border_u8_c4_with_scratch,
    morphology::morph_gradient_border_u8_c4
);
impl_morph_buffer_size!(u8, C4, morphology::morph_buffer_size_u8_c4);
impl_composite_morphology_border_image!(
    u16,
    C1,
    morphology::morph_close_border_u16_c1_with_scratch,
    morphology::morph_close_border_u16_c1,
    morphology::morph_open_border_u16_c1_with_scratch,
    morphology::morph_open_border_u16_c1,
    morphology::morph_top_hat_border_u16_c1_with_scratch,
    morphology::morph_top_hat_border_u16_c1,
    morphology::morph_black_hat_border_u16_c1_with_scratch,
    morphology::morph_black_hat_border_u16_c1,
    morphology::morph_gradient_border_u16_c1_with_scratch,
    morphology::morph_gradient_border_u16_c1
);
impl_morph_buffer_size!(u16, C1, morphology::morph_buffer_size_u16_c1);
impl_composite_morphology_border_image!(
    i16,
    C1,
    morphology::morph_close_border_i16_c1_with_scratch,
    morphology::morph_close_border_i16_c1,
    morphology::morph_open_border_i16_c1_with_scratch,
    morphology::morph_open_border_i16_c1,
    morphology::morph_top_hat_border_i16_c1_with_scratch,
    morphology::morph_top_hat_border_i16_c1,
    morphology::morph_black_hat_border_i16_c1_with_scratch,
    morphology::morph_black_hat_border_i16_c1,
    morphology::morph_gradient_border_i16_c1_with_scratch,
    morphology::morph_gradient_border_i16_c1
);
impl_morph_buffer_size!(i16, C1, morphology::morph_buffer_size_i16_c1);
impl_composite_morphology_border_image!(
    f32,
    C1,
    morphology::morph_close_border_f32_c1_with_scratch,
    morphology::morph_close_border_f32_c1,
    morphology::morph_open_border_f32_c1_with_scratch,
    morphology::morph_open_border_f32_c1,
    morphology::morph_top_hat_border_f32_c1_with_scratch,
    morphology::morph_top_hat_border_f32_c1,
    morphology::morph_black_hat_border_f32_c1_with_scratch,
    morphology::morph_black_hat_border_f32_c1,
    morphology::morph_gradient_border_f32_c1_with_scratch,
    morphology::morph_gradient_border_f32_c1
);
impl_morph_buffer_size!(f32, C1, morphology::morph_buffer_size_f32_c1);
impl_composite_morphology_border_image!(
    f32,
    C3,
    morphology::morph_close_border_f32_c3_with_scratch,
    morphology::morph_close_border_f32_c3,
    morphology::morph_open_border_f32_c3_with_scratch,
    morphology::morph_open_border_f32_c3,
    morphology::morph_top_hat_border_f32_c3_with_scratch,
    morphology::morph_top_hat_border_f32_c3,
    morphology::morph_black_hat_border_f32_c3_with_scratch,
    morphology::morph_black_hat_border_f32_c3,
    morphology::morph_gradient_border_f32_c3_with_scratch,
    morphology::morph_gradient_border_f32_c3
);
impl_morph_buffer_size!(f32, C3, morphology::morph_buffer_size_f32_c3);
impl_composite_morphology_border_image!(
    f32,
    C4,
    morphology::morph_close_border_f32_c4_with_scratch,
    morphology::morph_close_border_f32_c4,
    morphology::morph_open_border_f32_c4_with_scratch,
    morphology::morph_open_border_f32_c4,
    morphology::morph_top_hat_border_f32_c4_with_scratch,
    morphology::morph_top_hat_border_f32_c4,
    morphology::morph_black_hat_border_f32_c4_with_scratch,
    morphology::morph_black_hat_border_f32_c4,
    morphology::morph_gradient_border_f32_c4_with_scratch,
    morphology::morph_gradient_border_f32_c4
);
impl_morph_buffer_size!(f32, C4, morphology::morph_buffer_size_f32_c4);
