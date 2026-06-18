use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
};

use super::super::super::{ImagePipeline, morphology_traits::GrayMaskMorphologyBorderImage};

impl_gray_mask_morphology_border_image!(
    u8,
    i32,
    morphology::gray_dilate_border_u8_c1,
    morphology::gray_erode_border_u8_c1
);
impl_gray_mask_morphology_border_image!(
    f32,
    f32,
    morphology::gray_dilate_border_f32_c1,
    morphology::gray_erode_border_f32_c1
);
