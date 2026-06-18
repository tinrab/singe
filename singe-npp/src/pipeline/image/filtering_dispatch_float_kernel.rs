use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
};

use super::{FloatKernelFilterImage, ImagePipeline};

#[path = "filtering_dispatch_float_kernel_float_impls.rs"]
mod float_impls;
#[path = "filtering_dispatch_float_kernel_i16.rs"]
mod i16_impls;
#[path = "filtering_dispatch_float_kernel_i32.rs"]
mod i32_impls;
#[path = "filtering_dispatch_float_kernel_u16.rs"]
mod u16_impls;

impl_float_kernel_filter_image!(
    u8,
    C1,
    filtering::filter32f_u8_c1,
    filtering::filter_border32f_u8_c1
);
impl_float_kernel_filter_image!(
    u8,
    C2,
    filtering::filter32f_u8_c2,
    filtering::filter_border32f_u8_c2
);
impl_float_kernel_filter_image!(
    u8,
    C3,
    filtering::filter32f_u8_c3,
    filtering::filter_border32f_u8_c3
);
impl_float_kernel_filter_image!(
    u8,
    C4,
    filtering::filter32f_u8_c4,
    filtering::filter_border32f_u8_c4
);
impl_float_kernel_filter_image!(
    u8,
    AC4,
    filtering::filter32f_u8_ac4,
    filtering::filter_border32f_u8_ac4
);
impl_float_kernel_filter_image!(
    i8,
    C1,
    filtering::filter32f_i8_c1,
    filtering::filter_border32f_i8_c1
);
impl_float_kernel_filter_image!(
    i8,
    C2,
    filtering::filter32f_i8_c2,
    filtering::filter_border32f_i8_c2
);
impl_float_kernel_filter_image!(
    i8,
    C3,
    filtering::filter32f_i8_c3,
    filtering::filter_border32f_i8_c3
);
impl_float_kernel_filter_image!(
    i8,
    C4,
    filtering::filter32f_i8_c4,
    filtering::filter_border32f_i8_c4
);
impl_float_kernel_filter_image!(
    i8,
    AC4,
    filtering::filter32f_i8_ac4,
    filtering::filter_border32f_i8_ac4
);
