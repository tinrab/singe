use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageOutputView, ImageWindow},
    },
    pipeline::image::{
        ImagePipeline,
        filtering_traits::{ScharrFilterImage, SobelExtendedFilterImage},
    },
    types::MaskSize,
};

impl_sobel_extended_filter_image!(
    f32,
    C1,
    filtering::filter_sobel_horizontal_mask_f32_c1,
    filtering::filter_sobel_vertical_mask_f32_c1,
    filtering::filter_sobel_horizontal_second_f32_c1,
    filtering::filter_sobel_vertical_second_f32_c1,
    filtering::filter_sobel_cross_f32_c1
);

impl_scharr_filter_image!(
    f32,
    C1,
    filtering::filter_scharr_horizontal_f32_c1,
    filtering::filter_scharr_vertical_f32_c1
);
