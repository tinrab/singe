use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::super::{ImagePipeline, filtering_traits::*};

impl_edge_directional_filter_image!(
    f32,
    C1,
    filtering::filter_prewitt_horizontal_f32_c1,
    filtering::filter_prewitt_vertical_f32_c1,
    filtering::filter_roberts_down_f32_c1,
    filtering::filter_roberts_up_f32_c1,
    filtering::filter_sobel_horizontal_f32_c1,
    filtering::filter_sobel_vertical_f32_c1
);
impl_edge_directional_filter_image!(
    f32,
    C3,
    filtering::filter_prewitt_horizontal_f32_c3,
    filtering::filter_prewitt_vertical_f32_c3,
    filtering::filter_roberts_down_f32_c3,
    filtering::filter_roberts_up_f32_c3,
    filtering::filter_sobel_horizontal_f32_c3,
    filtering::filter_sobel_vertical_f32_c3
);
impl_edge_directional_filter_image!(
    f32,
    C4,
    filtering::filter_prewitt_horizontal_f32_c4,
    filtering::filter_prewitt_vertical_f32_c4,
    filtering::filter_roberts_down_f32_c4,
    filtering::filter_roberts_up_f32_c4,
    filtering::filter_sobel_horizontal_f32_c4,
    filtering::filter_sobel_vertical_f32_c4
);
impl_edge_directional_filter_image!(
    f32,
    AC4,
    filtering::filter_prewitt_horizontal_f32_ac4,
    filtering::filter_prewitt_vertical_f32_ac4,
    filtering::filter_roberts_down_f32_ac4,
    filtering::filter_roberts_up_f32_ac4,
    filtering::filter_sobel_horizontal_f32_ac4,
    filtering::filter_sobel_vertical_f32_ac4
);
