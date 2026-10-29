use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, ImageView},
    },
    pipeline::image::{
        ImagePipeline, statistics_dispatch::quality_dispatch_impls::ErrorMetricImage,
    },
};

mod c4_impls;

impl_error_metric_image!(
    u8,
    C1,
    statistics::maximum_error_u8_c1,
    statistics::average_error_u8_c1
);

impl_error_metric_image!(
    i8,
    C1,
    statistics::maximum_error_i8_c1,
    statistics::average_error_i8_c1
);

impl_error_metric_image!(
    u16,
    C1,
    statistics::maximum_error_u16_c1,
    statistics::average_error_u16_c1
);

impl_error_metric_image!(
    i16,
    C1,
    statistics::maximum_error_i16_c1,
    statistics::average_error_i16_c1
);

impl_error_metric_image!(
    u32,
    C1,
    statistics::maximum_error_u32_c1,
    statistics::average_error_u32_c1
);

impl_error_metric_image!(
    i32,
    C1,
    statistics::maximum_error_i32_c1,
    statistics::average_error_i32_c1
);

impl_error_metric_image!(
    f32,
    C1,
    statistics::maximum_error_f32_c1,
    statistics::average_error_f32_c1
);

impl_error_metric_image!(
    f64,
    C1,
    statistics::maximum_error_f64_c1,
    statistics::average_error_f64_c1
);

impl_error_metric_image!(
    u8,
    C3,
    statistics::maximum_error_u8_c3,
    statistics::average_error_u8_c3
);

impl_error_metric_image!(
    i8,
    C3,
    statistics::maximum_error_i8_c3,
    statistics::average_error_i8_c3
);

impl_error_metric_image!(
    u16,
    C3,
    statistics::maximum_error_u16_c3,
    statistics::average_error_u16_c3
);

impl_error_metric_image!(
    i16,
    C3,
    statistics::maximum_error_i16_c3,
    statistics::average_error_i16_c3
);

impl_error_metric_image!(
    u32,
    C3,
    statistics::maximum_error_u32_c3,
    statistics::average_error_u32_c3
);

impl_error_metric_image!(
    i32,
    C3,
    statistics::maximum_error_i32_c3,
    statistics::average_error_i32_c3
);

impl_error_metric_image!(
    f32,
    C3,
    statistics::maximum_error_f32_c3,
    statistics::average_error_f32_c3
);

impl_error_metric_image!(
    f64,
    C3,
    statistics::maximum_error_f64_c3,
    statistics::average_error_f64_c3
);
