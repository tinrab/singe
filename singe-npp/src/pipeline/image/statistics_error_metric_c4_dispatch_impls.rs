use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C4, ImageView},
    },
};

use super::super::super::super::ImagePipeline;
use super::super::*;

impl_error_metric_image!(
    u8,
    C4,
    statistics::maximum_error_u8_c4,
    statistics::average_error_u8_c4
);
impl_error_metric_image!(
    i8,
    C4,
    statistics::maximum_error_i8_c4,
    statistics::average_error_i8_c4
);
impl_error_metric_image!(
    u16,
    C4,
    statistics::maximum_error_u16_c4,
    statistics::average_error_u16_c4
);
impl_error_metric_image!(
    i16,
    C4,
    statistics::maximum_error_i16_c4,
    statistics::average_error_i16_c4
);
impl_error_metric_image!(
    u32,
    C4,
    statistics::maximum_error_u32_c4,
    statistics::average_error_u32_c4
);
impl_error_metric_image!(
    i32,
    C4,
    statistics::maximum_error_i32_c4,
    statistics::average_error_i32_c4
);
impl_error_metric_image!(
    f32,
    C4,
    statistics::maximum_error_f32_c4,
    statistics::average_error_f32_c4
);
impl_error_metric_image!(
    f64,
    C4,
    statistics::maximum_error_f64_c4,
    statistics::average_error_f64_c4
);
