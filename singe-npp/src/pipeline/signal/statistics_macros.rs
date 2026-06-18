macro_rules! impl_scalar_statistic_signal {
    (
        $ty:ty,
        $sum_buffer_size:path,
        $sum_with_scratch:path,
        $sum_natural_logarithm_buffer_size:path,
        $sum_natural_logarithm_with_scratch:path,
        $mean_buffer_size:path,
        $mean_with_scratch:path,
        $standard_deviation_buffer_size:path,
        $standard_deviation_with_scratch:path,
        $norm_inf_buffer_size:path,
        $norm_inf_with_scratch:path,
        $norm_l1_buffer_size:path,
        $norm_l1_with_scratch:path,
        $norm_l2_buffer_size:path,
        $norm_l2_with_scratch:path
    ) => {
        impl<'a> ScalarStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn sum_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $sum_buffer_size(stream_context, source)
            }

            fn sum_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $sum_with_scratch(stream_context, source, destination, scratch)
            }

            fn sum_natural_logarithm_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $sum_natural_logarithm_buffer_size(stream_context, source)
            }

            fn sum_natural_logarithm_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $sum_natural_logarithm_with_scratch(stream_context, source, destination, scratch)
            }

            fn mean_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $mean_buffer_size(stream_context, source)
            }

            fn mean_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $mean_with_scratch(stream_context, source, destination, scratch)
            }

            fn standard_deviation_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $standard_deviation_buffer_size(stream_context, source)
            }

            fn standard_deviation_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $standard_deviation_with_scratch(stream_context, source, destination, scratch)
            }

            fn norm_inf_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $norm_inf_buffer_size(stream_context, source)
            }

            fn norm_inf_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $norm_inf_with_scratch(stream_context, source, destination, scratch)
            }

            fn norm_l1_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $norm_l1_buffer_size(stream_context, source)
            }

            fn norm_l1_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $norm_l1_with_scratch(stream_context, source, destination, scratch)
            }

            fn norm_l2_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $norm_l2_buffer_size(stream_context, source)
            }

            fn norm_l2_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $norm_l2_with_scratch(stream_context, source, destination, scratch)
            }
        }
    };
}

macro_rules! impl_extremum_statistic_signal {
    ($ty:ty, $max_buffer_size:path, $max_with_scratch:path, $min_buffer_size:path, $min_with_scratch:path) => {
        impl<'a> ExtremumStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn max_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $max_buffer_size(stream_context, source)
            }

            fn max_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $max_with_scratch(stream_context, source, destination, scratch)
            }

            fn min_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $min_buffer_size(stream_context, source)
            }

            fn min_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $min_with_scratch(stream_context, source, destination, scratch)
            }
        }
    };
}

macro_rules! impl_absolute_extremum_statistic_signal {
    ($ty:ty, $max_absolute_buffer_size:path, $max_absolute_with_scratch:path, $min_absolute_buffer_size:path, $min_absolute_with_scratch:path) => {
        impl<'a> AbsoluteExtremumStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn max_absolute_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $max_absolute_buffer_size(stream_context, source)
            }

            fn max_absolute_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $max_absolute_with_scratch(stream_context, source, destination, scratch)
            }

            fn min_absolute_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $min_absolute_buffer_size(stream_context, source)
            }

            fn min_absolute_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $min_absolute_with_scratch(stream_context, source, destination, scratch)
            }
        }
    };
}

macro_rules! impl_pair_statistic_signal {
    ($ty:ty, $min_max_buffer_size:path, $min_max_with_scratch:path) => {
        impl<'a> PairStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn min_max_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $min_max_buffer_size(stream_context, source)
            }

            fn min_max_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                min: &mut SignalViewMut<'_, $ty>,
                max: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $min_max_with_scratch(stream_context, source, min, max, scratch)
            }
        }
    };
}

macro_rules! impl_mean_standard_deviation_statistic_signal {
    ($ty:ty, $buffer_size:path, $with_scratch:path) => {
        impl<'a> MeanStandardDeviationStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn mean_standard_deviation_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $buffer_size(stream_context, source)
            }

            fn mean_standard_deviation_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                mean: &mut SignalViewMut<'_, $ty>,
                standard_deviation: &mut SignalViewMut<'_, $ty>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $with_scratch(stream_context, source, mean, standard_deviation, scratch)
            }
        }
    };
}

macro_rules! impl_indexed_statistic_signal {
    ($ty:ty, $max_index_buffer_size:path, $max_index_with_scratch:path, $min_index_buffer_size:path, $min_index_with_scratch:path) => {
        impl<'a> IndexedStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn max_index_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $max_index_buffer_size(stream_context, source)
            }

            fn max_index_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: &mut SignalViewMut<'_, $ty>,
                index: &mut SignalViewMut<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $max_index_with_scratch(stream_context, source, value, index, scratch)
            }

            fn min_index_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $min_index_buffer_size(stream_context, source)
            }

            fn min_index_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: &mut SignalViewMut<'_, $ty>,
                index: &mut SignalViewMut<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $min_index_with_scratch(stream_context, source, value, index, scratch)
            }
        }
    };
}

macro_rules! impl_absolute_indexed_statistic_signal {
    ($ty:ty, $max_absolute_index_buffer_size:path, $max_absolute_index_with_scratch:path, $min_absolute_index_buffer_size:path, $min_absolute_index_with_scratch:path) => {
        impl<'a> AbsoluteIndexedStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn max_absolute_index_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $max_absolute_index_buffer_size(stream_context, source)
            }

            fn max_absolute_index_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: &mut SignalViewMut<'_, $ty>,
                index: &mut SignalViewMut<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $max_absolute_index_with_scratch(stream_context, source, value, index, scratch)
            }

            fn min_absolute_index_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $min_absolute_index_buffer_size(stream_context, source)
            }

            fn min_absolute_index_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: &mut SignalViewMut<'_, $ty>,
                index: &mut SignalViewMut<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $min_absolute_index_with_scratch(stream_context, source, value, index, scratch)
            }
        }
    };
}

macro_rules! impl_every_statistic_signal {
    ($ty:ty, $min_every:path, $max_every:path) => {
        impl<'a> EveryStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn min_every_in_place(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $min_every(stream_context, source, destination)
            }

            fn max_every_in_place(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $max_every(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_error_metric_statistic_signal {
    (
        $ty:ty,
        $maximum_error_buffer_size:path,
        $maximum_error_with_scratch:path,
        $average_error_buffer_size:path,
        $average_error_with_scratch:path,
        $maximum_relative_error_buffer_size:path,
        $maximum_relative_error_with_scratch:path,
        $average_relative_error_buffer_size:path,
        $average_relative_error_with_scratch:path
    ) => {
        impl<'a> ErrorMetricStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn maximum_error_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $maximum_error_buffer_size(stream_context, source)
            }

            fn maximum_error_with_scratch(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, $ty>,
                source_2: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, f64>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $maximum_error_with_scratch(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    scratch,
                )
            }

            fn average_error_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $average_error_buffer_size(stream_context, source)
            }

            fn average_error_with_scratch(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, $ty>,
                source_2: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, f64>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $average_error_with_scratch(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    scratch,
                )
            }

            fn maximum_relative_error_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $maximum_relative_error_buffer_size(stream_context, source)
            }

            fn maximum_relative_error_with_scratch(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, $ty>,
                source_2: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, f64>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $maximum_relative_error_with_scratch(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    scratch,
                )
            }

            fn average_relative_error_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $average_relative_error_buffer_size(stream_context, source)
            }

            fn average_relative_error_with_scratch(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, $ty>,
                source_2: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, f64>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $average_relative_error_with_scratch(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    scratch,
                )
            }
        }
    };
}

macro_rules! impl_zero_crossing_statistic_signal {
    ($ty:ty, $buffer_size:path, $with_scratch:path) => {
        impl<'a> ZeroCrossingStatisticSignal<$ty> for SignalPipeline<'a, $ty> {
            fn zero_crossing_buffer_size(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
            ) -> Result<usize> {
                $buffer_size(stream_context, source)
            }

            fn zero_crossing_with_scratch(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, f32>,
                zero_crossing_type: ZeroCrossingType,
                scratch: &mut ScratchBuffer,
            ) -> Result<()> {
                $with_scratch(
                    stream_context,
                    source,
                    destination,
                    zero_crossing_type,
                    scratch,
                )
            }
        }
    };
}
