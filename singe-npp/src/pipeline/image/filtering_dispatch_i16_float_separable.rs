use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, filtering_traits::FloatSeparableFilterImage};

impl_float_separable_filter_image!(
    i16,
    C1,
    filtering::filter_column32f_i16_c1,
    filtering::filter_row32f_i16_c1
);
impl_float_separable_filter_image!(
    i16,
    C3,
    filtering::filter_column32f_i16_c3,
    filtering::filter_row32f_i16_c3
);
impl_float_separable_filter_image!(
    i16,
    C4,
    filtering::filter_column32f_i16_c4,
    filtering::filter_row32f_i16_c4
);
impl_float_separable_filter_image!(
    i16,
    AC4,
    filtering::filter_column32f_i16_ac4,
    filtering::filter_row32f_i16_ac4
);
