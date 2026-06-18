use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
};

use super::super::super::super::{ImagePipeline, filtering_traits::NeighborhoodBorderFilterImage};

impl_neighborhood_border_filter_image!(
    f32,
    C1,
    filtering::filter_max_border_f32_c1,
    filtering::filter_min_border_f32_c1
);
impl_neighborhood_border_filter_image!(
    f32,
    C3,
    filtering::filter_max_border_f32_c3,
    filtering::filter_min_border_f32_c3
);
impl_neighborhood_border_filter_image!(
    f32,
    C4,
    filtering::filter_max_border_f32_c4,
    filtering::filter_min_border_f32_c4
);
impl_neighborhood_border_filter_image!(
    f32,
    AC4,
    filtering::filter_max_border_f32_ac4,
    filtering::filter_min_border_f32_ac4
);
