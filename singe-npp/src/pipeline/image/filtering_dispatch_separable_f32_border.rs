use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point},
};

use super::{FloatSeparableBorderFilterImage, ImagePipeline};

impl_float_separable_border_filter_image!(
    f32,
    C1,
    filtering::filter_column_border_f32_c1,
    filtering::filter_row_border_f32_c1
);
impl_float_separable_border_filter_image!(
    f32,
    C3,
    filtering::filter_column_border_f32_c3,
    filtering::filter_row_border_f32_c3
);
impl_float_separable_border_filter_image!(
    f32,
    C4,
    filtering::filter_column_border_f32_c4,
    filtering::filter_row_border_f32_c4
);
impl_float_separable_border_filter_image!(
    f32,
    AC4,
    filtering::filter_column_border_f32_ac4,
    filtering::filter_row_border_f32_ac4
);
