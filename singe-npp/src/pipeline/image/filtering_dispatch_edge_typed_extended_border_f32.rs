use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{BorderType, MaskSize, Point},
};

use super::super::super::super::{
    ImagePipeline,
    filtering_traits::{ScharrBorderFilterImage, SobelExtendedBorderFilterImage},
};

impl_sobel_extended_border_filter_image!(
    f32,
    C1,
    filtering::filter_sobel_horizontal_mask_border_f32_c1,
    filtering::filter_sobel_vertical_mask_border_f32_c1,
    filtering::filter_sobel_horizontal_second_border_f32_c1,
    filtering::filter_sobel_vertical_second_border_f32_c1,
    filtering::filter_sobel_cross_border_f32_c1
);
impl_scharr_border_filter_image!(
    f32,
    C1,
    filtering::filter_scharr_horizontal_border_f32_c1,
    filtering::filter_scharr_vertical_border_f32_c1
);
