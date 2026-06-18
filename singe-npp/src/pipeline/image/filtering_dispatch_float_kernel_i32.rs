use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
};

use super::super::{FloatKernelFilterImage, ImagePipeline};

impl_float_kernel_filter_image!(
    i32,
    C1,
    filtering::filter32f_i32_c1,
    filtering::filter_border32f_i32_c1
);
impl_float_kernel_filter_image!(
    i32,
    C3,
    filtering::filter32f_i32_c3,
    filtering::filter_border32f_i32_c3
);
impl_float_kernel_filter_image!(
    i32,
    C4,
    filtering::filter32f_i32_c4,
    filtering::filter_border32f_i32_c4
);
impl_float_kernel_filter_image!(
    i32,
    AC4,
    filtering::filter32f_i32_ac4,
    filtering::filter_border32f_i32_ac4
);
