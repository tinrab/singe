use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
};

use super::super::{DoubleKernelFilterImage, FloatKernelFilterImage, ImagePipeline};

impl_float_kernel_filter_image!(
    f16,
    C1,
    filtering::filter32f_f16_c1,
    filtering::filter_border32f_f16_c1
);
impl_float_kernel_filter_image!(
    f16,
    C3,
    filtering::filter32f_f16_c3,
    filtering::filter_border32f_f16_c3
);
impl_float_kernel_filter_image!(
    f16,
    C4,
    filtering::filter32f_f16_c4,
    filtering::filter_border32f_f16_c4
);
impl_float_kernel_filter_image!(
    f32,
    C1,
    filtering::filter_f32_c1,
    filtering::filter_border_f32_c1
);
impl_float_kernel_filter_image!(
    f32,
    C2,
    filtering::filter_f32_c2,
    filtering::filter_border_f32_c2
);
impl_float_kernel_filter_image!(
    f32,
    C3,
    filtering::filter_f32_c3,
    filtering::filter_border_f32_c3
);
impl_float_kernel_filter_image!(
    f32,
    C4,
    filtering::filter_f32_c4,
    filtering::filter_border_f32_c4
);
impl_float_kernel_filter_image!(
    f32,
    AC4,
    filtering::filter_f32_ac4,
    filtering::filter_border_f32_ac4
);
impl_double_kernel_filter_image!(f64, C1, filtering::filter_f64_c1);
