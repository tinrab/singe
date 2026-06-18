use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        statistics,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike, ZeroCrossingType},
    workspace::ScratchBuffer,
};

use super::SignalPipeline;

#[path = "statistics_traits.rs"]
mod statistics_traits;
pub use statistics_traits::*;
#[macro_use]
#[path = "statistics_macros.rs"]
mod statistics_macros;

impl_scalar_statistic_signal!(
    f32,
    statistics::sum_f32_buffer_size,
    statistics::sum_f32_to_device_with_scratch,
    statistics::sum_natural_logarithm_f32_buffer_size,
    statistics::sum_natural_logarithm_f32_to_device_with_scratch,
    statistics::mean_f32_buffer_size,
    statistics::mean_f32_to_device_with_scratch,
    statistics::standard_deviation_f32_buffer_size,
    statistics::standard_deviation_f32_to_device_with_scratch,
    statistics::norm_inf_f32_buffer_size,
    statistics::norm_inf_f32_to_device_with_scratch,
    statistics::norm_l1_f32_buffer_size,
    statistics::norm_l1_f32_to_device_with_scratch,
    statistics::norm_l2_f32_buffer_size,
    statistics::norm_l2_f32_to_device_with_scratch
);
impl_scalar_statistic_signal!(
    f64,
    statistics::sum_f64_buffer_size,
    statistics::sum_f64_to_device_with_scratch,
    statistics::sum_natural_logarithm_f64_buffer_size,
    statistics::sum_natural_logarithm_f64_to_device_with_scratch,
    statistics::mean_f64_buffer_size,
    statistics::mean_f64_to_device_with_scratch,
    statistics::standard_deviation_f64_buffer_size,
    statistics::standard_deviation_f64_to_device_with_scratch,
    statistics::norm_inf_f64_buffer_size,
    statistics::norm_inf_f64_to_device_with_scratch,
    statistics::norm_l1_f64_buffer_size,
    statistics::norm_l1_f64_to_device_with_scratch,
    statistics::norm_l2_f64_buffer_size,
    statistics::norm_l2_f64_to_device_with_scratch
);

impl_extremum_statistic_signal!(
    i16,
    statistics::max_i16_buffer_size,
    statistics::max_i16_to_device_with_scratch,
    statistics::min_i16_buffer_size,
    statistics::min_i16_to_device_with_scratch
);
impl_extremum_statistic_signal!(
    i32,
    statistics::max_i32_buffer_size,
    statistics::max_i32_to_device_with_scratch,
    statistics::min_i32_buffer_size,
    statistics::min_i32_to_device_with_scratch
);
impl_extremum_statistic_signal!(
    f32,
    statistics::max_f32_buffer_size,
    statistics::max_f32_to_device_with_scratch,
    statistics::min_f32_buffer_size,
    statistics::min_f32_to_device_with_scratch
);
impl_extremum_statistic_signal!(
    f64,
    statistics::max_f64_buffer_size,
    statistics::max_f64_to_device_with_scratch,
    statistics::min_f64_buffer_size,
    statistics::min_f64_to_device_with_scratch
);

impl_absolute_extremum_statistic_signal!(
    i16,
    statistics::max_absolute_i16_buffer_size,
    statistics::max_absolute_i16_to_device_with_scratch,
    statistics::min_absolute_i16_buffer_size,
    statistics::min_absolute_i16_to_device_with_scratch
);
impl_absolute_extremum_statistic_signal!(
    i32,
    statistics::max_absolute_i32_buffer_size,
    statistics::max_absolute_i32_to_device_with_scratch,
    statistics::min_absolute_i32_buffer_size,
    statistics::min_absolute_i32_to_device_with_scratch
);

impl_pair_statistic_signal!(
    i16,
    statistics::min_max_i16_buffer_size,
    statistics::min_max_i16_to_device_with_scratch
);
impl_pair_statistic_signal!(
    i32,
    statistics::min_max_i32_buffer_size,
    statistics::min_max_i32_to_device_with_scratch
);
impl_pair_statistic_signal!(
    f32,
    statistics::min_max_f32_buffer_size,
    statistics::min_max_f32_to_device_with_scratch
);
impl_pair_statistic_signal!(
    f64,
    statistics::min_max_f64_buffer_size,
    statistics::min_max_f64_to_device_with_scratch
);

impl_mean_standard_deviation_statistic_signal!(
    f32,
    statistics::mean_standard_deviation_f32_buffer_size,
    statistics::mean_standard_deviation_f32_to_device_with_scratch
);
impl_mean_standard_deviation_statistic_signal!(
    f64,
    statistics::mean_standard_deviation_f64_buffer_size,
    statistics::mean_standard_deviation_f64_to_device_with_scratch
);

impl_indexed_statistic_signal!(
    i16,
    statistics::max_index_i16_buffer_size,
    statistics::max_index_i16_to_device_with_scratch,
    statistics::min_index_i16_buffer_size,
    statistics::min_index_i16_to_device_with_scratch
);
impl_indexed_statistic_signal!(
    i32,
    statistics::max_index_i32_buffer_size,
    statistics::max_index_i32_to_device_with_scratch,
    statistics::min_index_i32_buffer_size,
    statistics::min_index_i32_to_device_with_scratch
);
impl_indexed_statistic_signal!(
    f32,
    statistics::max_index_f32_buffer_size,
    statistics::max_index_f32_to_device_with_scratch,
    statistics::min_index_f32_buffer_size,
    statistics::min_index_f32_to_device_with_scratch
);
impl_indexed_statistic_signal!(
    f64,
    statistics::max_index_f64_buffer_size,
    statistics::max_index_f64_to_device_with_scratch,
    statistics::min_index_f64_buffer_size,
    statistics::min_index_f64_to_device_with_scratch
);

impl_absolute_indexed_statistic_signal!(
    i16,
    statistics::max_absolute_index_i16_buffer_size,
    statistics::max_absolute_index_i16_to_device_with_scratch,
    statistics::min_absolute_index_i16_buffer_size,
    statistics::min_absolute_index_i16_to_device_with_scratch
);
impl_absolute_indexed_statistic_signal!(
    i32,
    statistics::max_absolute_index_i32_buffer_size,
    statistics::max_absolute_index_i32_to_device_with_scratch,
    statistics::min_absolute_index_i32_buffer_size,
    statistics::min_absolute_index_i32_to_device_with_scratch
);

impl_every_statistic_signal!(
    u8,
    statistics::min_every_u8_in_place,
    statistics::max_every_u8_in_place
);
impl_every_statistic_signal!(
    u16,
    statistics::min_every_u16_in_place,
    statistics::max_every_u16_in_place
);
impl_every_statistic_signal!(
    i16,
    statistics::min_every_i16_in_place,
    statistics::max_every_i16_in_place
);
impl_every_statistic_signal!(
    i32,
    statistics::min_every_i32_in_place,
    statistics::max_every_i32_in_place
);
impl_every_statistic_signal!(
    f32,
    statistics::min_every_f32_in_place,
    statistics::max_every_f32_in_place
);

impl<'a, T> SignalPipeline<'a, T>
where
    T: DataTypeLike + statistics::MinEveryInPlace,
{
    pub fn min_every_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        statistics::min_every_in_place(stream_context, source, destination)
    }
}

impl_error_metric_statistic_signal!(
    u8,
    statistics::maximum_error_u8_buffer_size,
    statistics::maximum_error_u8_to_device_with_scratch,
    statistics::average_error_u8_buffer_size,
    statistics::average_error_u8_to_device_with_scratch,
    statistics::maximum_relative_error_u8_buffer_size,
    statistics::maximum_relative_error_u8_to_device_with_scratch,
    statistics::average_relative_error_u8_buffer_size,
    statistics::average_relative_error_u8_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    i8,
    statistics::maximum_error_i8_buffer_size,
    statistics::maximum_error_i8_to_device_with_scratch,
    statistics::average_error_i8_buffer_size,
    statistics::average_error_i8_to_device_with_scratch,
    statistics::maximum_relative_error_i8_buffer_size,
    statistics::maximum_relative_error_i8_to_device_with_scratch,
    statistics::average_relative_error_i8_buffer_size,
    statistics::average_relative_error_i8_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    u16,
    statistics::maximum_error_u16_buffer_size,
    statistics::maximum_error_u16_to_device_with_scratch,
    statistics::average_error_u16_buffer_size,
    statistics::average_error_u16_to_device_with_scratch,
    statistics::maximum_relative_error_u16_buffer_size,
    statistics::maximum_relative_error_u16_to_device_with_scratch,
    statistics::average_relative_error_u16_buffer_size,
    statistics::average_relative_error_u16_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    i16,
    statistics::maximum_error_i16_buffer_size,
    statistics::maximum_error_i16_to_device_with_scratch,
    statistics::average_error_i16_buffer_size,
    statistics::average_error_i16_to_device_with_scratch,
    statistics::maximum_relative_error_i16_buffer_size,
    statistics::maximum_relative_error_i16_to_device_with_scratch,
    statistics::average_relative_error_i16_buffer_size,
    statistics::average_relative_error_i16_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    ComplexI16,
    statistics::maximum_error_i16_complex_buffer_size,
    statistics::maximum_error_i16_complex_to_device_with_scratch,
    statistics::average_error_i16_complex_buffer_size,
    statistics::average_error_i16_complex_to_device_with_scratch,
    statistics::maximum_relative_error_i16_complex_buffer_size,
    statistics::maximum_relative_error_i16_complex_to_device_with_scratch,
    statistics::average_relative_error_i16_complex_buffer_size,
    statistics::average_relative_error_i16_complex_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    u32,
    statistics::maximum_error_u32_buffer_size,
    statistics::maximum_error_u32_to_device_with_scratch,
    statistics::average_error_u32_buffer_size,
    statistics::average_error_u32_to_device_with_scratch,
    statistics::maximum_relative_error_u32_buffer_size,
    statistics::maximum_relative_error_u32_to_device_with_scratch,
    statistics::average_relative_error_u32_buffer_size,
    statistics::average_relative_error_u32_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    i32,
    statistics::maximum_error_i32_buffer_size,
    statistics::maximum_error_i32_to_device_with_scratch,
    statistics::average_error_i32_buffer_size,
    statistics::average_error_i32_to_device_with_scratch,
    statistics::maximum_relative_error_i32_buffer_size,
    statistics::maximum_relative_error_i32_to_device_with_scratch,
    statistics::average_relative_error_i32_buffer_size,
    statistics::average_relative_error_i32_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    ComplexI32,
    statistics::maximum_error_i32_complex_buffer_size,
    statistics::maximum_error_i32_complex_to_device_with_scratch,
    statistics::average_error_i32_complex_buffer_size,
    statistics::average_error_i32_complex_to_device_with_scratch,
    statistics::maximum_relative_error_i32_complex_buffer_size,
    statistics::maximum_relative_error_i32_complex_to_device_with_scratch,
    statistics::average_relative_error_i32_complex_buffer_size,
    statistics::average_relative_error_i32_complex_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    i64,
    statistics::maximum_error_i64_buffer_size,
    statistics::maximum_error_i64_to_device_with_scratch,
    statistics::average_error_i64_buffer_size,
    statistics::average_error_i64_to_device_with_scratch,
    statistics::maximum_relative_error_i64_buffer_size,
    statistics::maximum_relative_error_i64_to_device_with_scratch,
    statistics::average_relative_error_i64_buffer_size,
    statistics::average_relative_error_i64_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    ComplexI64,
    statistics::maximum_error_i64_complex_buffer_size,
    statistics::maximum_error_i64_complex_to_device_with_scratch,
    statistics::average_error_i64_complex_buffer_size,
    statistics::average_error_i64_complex_to_device_with_scratch,
    statistics::maximum_relative_error_i64_complex_buffer_size,
    statistics::maximum_relative_error_i64_complex_to_device_with_scratch,
    statistics::average_relative_error_i64_complex_buffer_size,
    statistics::average_relative_error_i64_complex_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    f32,
    statistics::maximum_error_f32_buffer_size,
    statistics::maximum_error_f32_to_device_with_scratch,
    statistics::average_error_f32_buffer_size,
    statistics::average_error_f32_to_device_with_scratch,
    statistics::maximum_relative_error_f32_buffer_size,
    statistics::maximum_relative_error_f32_to_device_with_scratch,
    statistics::average_relative_error_f32_buffer_size,
    statistics::average_relative_error_f32_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    Complex32,
    statistics::maximum_error_f32_complex_buffer_size,
    statistics::maximum_error_f32_complex_to_device_with_scratch,
    statistics::average_error_f32_complex_buffer_size,
    statistics::average_error_f32_complex_to_device_with_scratch,
    statistics::maximum_relative_error_f32_complex_buffer_size,
    statistics::maximum_relative_error_f32_complex_to_device_with_scratch,
    statistics::average_relative_error_f32_complex_buffer_size,
    statistics::average_relative_error_f32_complex_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    f64,
    statistics::maximum_error_f64_buffer_size,
    statistics::maximum_error_f64_to_device_with_scratch,
    statistics::average_error_f64_buffer_size,
    statistics::average_error_f64_to_device_with_scratch,
    statistics::maximum_relative_error_f64_buffer_size,
    statistics::maximum_relative_error_f64_to_device_with_scratch,
    statistics::average_relative_error_f64_buffer_size,
    statistics::average_relative_error_f64_to_device_with_scratch
);
impl_error_metric_statistic_signal!(
    Complex64,
    statistics::maximum_error_f64_complex_buffer_size,
    statistics::maximum_error_f64_complex_to_device_with_scratch,
    statistics::average_error_f64_complex_buffer_size,
    statistics::average_error_f64_complex_to_device_with_scratch,
    statistics::maximum_relative_error_f64_complex_buffer_size,
    statistics::maximum_relative_error_f64_complex_to_device_with_scratch,
    statistics::average_relative_error_f64_complex_buffer_size,
    statistics::average_relative_error_f64_complex_to_device_with_scratch
);

impl_zero_crossing_statistic_signal!(
    i16,
    statistics::zero_crossing_i16_to_f32_buffer_size,
    statistics::zero_crossing_i16_to_f32_to_device_with_scratch
);
impl_zero_crossing_statistic_signal!(
    f32,
    statistics::zero_crossing_f32_buffer_size,
    statistics::zero_crossing_f32_to_device_with_scratch
);

impl<'a> BinaryScalarStatisticSignal<f32> for SignalPipeline<'a, f32> {
    fn norm_diff_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::norm_diff_inf_f32_buffer_size(stream_context, source)
    }

    fn norm_diff_inf_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, f32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::norm_diff_inf_f32_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    fn norm_diff_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::norm_diff_l1_f32_buffer_size(stream_context, source)
    }

    fn norm_diff_l1_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, f32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::norm_diff_l1_f32_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    fn norm_diff_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::norm_diff_l2_f32_buffer_size(stream_context, source)
    }

    fn norm_diff_l2_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, f32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::norm_diff_l2_f32_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    fn dot_product_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_buffer_size(stream_context, source)
    }

    fn dot_product_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, f32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::dot_product_f32_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }
}

impl<'a> BinaryScalarStatisticSignal<f64> for SignalPipeline<'a, f64> {
    fn norm_diff_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
    ) -> Result<usize> {
        statistics::norm_diff_inf_f64_buffer_size(stream_context, source)
    }

    fn norm_diff_inf_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f64>,
        source2: &SignalView<'_, f64>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::norm_diff_inf_f64_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    fn norm_diff_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
    ) -> Result<usize> {
        statistics::norm_diff_l1_f64_buffer_size(stream_context, source)
    }

    fn norm_diff_l1_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f64>,
        source2: &SignalView<'_, f64>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::norm_diff_l1_f64_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    fn norm_diff_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
    ) -> Result<usize> {
        statistics::norm_diff_l2_f64_buffer_size(stream_context, source)
    }

    fn norm_diff_l2_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f64>,
        source2: &SignalView<'_, f64>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::norm_diff_l2_f64_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    fn dot_product_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
    ) -> Result<usize> {
        statistics::dot_product_f64_buffer_size(stream_context, source)
    }

    fn dot_product_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f64>,
        source2: &SignalView<'_, f64>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::dot_product_f64_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }
}
