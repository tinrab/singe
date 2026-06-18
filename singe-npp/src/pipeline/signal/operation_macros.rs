macro_rules! impl_constant_signal {
    (
        $ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path,
        $divide:path,
        $divide_in_place:path
    ) => {
        impl<'a> ConstantSignal<$ty> for SignalPipeline<'a, $ty> {
            fn add_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $add(stream_context, source, value, destination)
            }

            fn add_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $add_in_place(stream_context, signal, value)
            }

            fn subtract_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $subtract(stream_context, source, value, destination)
            }

            fn subtract_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $subtract_in_place(stream_context, signal, value)
            }

            fn multiply_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $multiply(stream_context, source, value, destination)
            }

            fn multiply_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $multiply_in_place(stream_context, signal, value)
            }

            fn divide_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $divide(stream_context, source, value, destination)
            }

            fn divide_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $divide_in_place(stream_context, signal, value)
            }
        }
    };
}

macro_rules! impl_unary_signal {
    (
        $ty:ty,
        $exponent:path,
        $exponent_in_place:path,
        $natural_logarithm:path,
        $natural_logarithm_in_place:path,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        impl<'a> UnarySignal<$ty> for SignalPipeline<'a, $ty> {
            fn exponent_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $exponent(stream_context, source, destination)
            }

            fn exponent_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $exponent_in_place(stream_context, signal)
            }

            fn natural_logarithm_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $natural_logarithm(stream_context, source, destination)
            }

            fn natural_logarithm_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $natural_logarithm_in_place(stream_context, signal)
            }

            fn square_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $square(stream_context, source, destination)
            }

            fn square_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $square_in_place(stream_context, signal)
            }

            fn square_root_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $square_root(stream_context, source, destination)
            }

            fn square_root_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $square_root_in_place(stream_context, signal)
            }
        }
    };
}

macro_rules! impl_subtract_from_constant_signal {
    ($ty:ty, $subtract_from:path, $subtract_from_in_place:path) => {
        impl<'a> SubtractFromConstantSignal<$ty> for SignalPipeline<'a, $ty> {
            fn subtract_from_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $subtract_from(stream_context, source, value, destination)
            }

            fn subtract_from_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $subtract_from_in_place(stream_context, signal, value)
            }
        }
    };
}

macro_rules! impl_divide_into_constant_signal {
    ($ty:ty, $divide_into:path, $divide_into_in_place:path) => {
        impl<'a> DivideIntoConstantSignal<$ty> for SignalPipeline<'a, $ty> {
            fn divide_into_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $divide_into(stream_context, source, value, destination)
            }

            fn divide_into_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $divide_into_in_place(stream_context, signal, value)
            }
        }
    };
}

macro_rules! impl_threshold_signal {
    ($ty:ty, $threshold:path, $threshold_in_place:path) => {
        impl<'a> ThresholdSignal<$ty> for SignalPipeline<'a, $ty> {
            fn threshold_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold(stream_context, source, destination, level, operation)
            }

            fn threshold_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold_in_place(stream_context, signal, level, operation)
            }
        }
    };
}

macro_rules! impl_complex_threshold_signal {
    ($ty:ty, $level:ty, $threshold:path, $threshold_in_place:path) => {
        impl<'a> ComplexThresholdSignal<$ty, $level> for SignalPipeline<'a, $ty> {
            fn threshold_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $level,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold(stream_context, source, destination, level, operation)
            }

            fn threshold_complex_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $level,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold_in_place(stream_context, signal, level, operation)
            }
        }
    };
}

macro_rules! impl_fixed_threshold_signal {
    ($ty:ty, $less:path, $less_in_place:path, $greater:path, $greater_in_place:path) => {
        impl<'a> FixedThresholdSignal<$ty> for SignalPipeline<'a, $ty> {
            fn threshold_less_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $ty,
            ) -> Result<()> {
                $less(stream_context, source, destination, level)
            }

            fn threshold_less_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $ty,
            ) -> Result<()> {
                $less_in_place(stream_context, signal, level)
            }

            fn threshold_greater_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $ty,
            ) -> Result<()> {
                $greater(stream_context, source, destination, level)
            }

            fn threshold_greater_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $ty,
            ) -> Result<()> {
                $greater_in_place(stream_context, signal, level)
            }
        }
    };
}

macro_rules! impl_complex_fixed_threshold_signal {
    (
        $ty:ty,
        $level:ty,
        $less:path,
        $less_in_place:path,
        $greater:path,
        $greater_in_place:path
    ) => {
        impl<'a> ComplexFixedThresholdSignal<$ty, $level> for SignalPipeline<'a, $ty> {
            fn threshold_less_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $level,
            ) -> Result<()> {
                $less(stream_context, source, destination, level)
            }

            fn threshold_less_complex_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $level,
            ) -> Result<()> {
                $less_in_place(stream_context, signal, level)
            }

            fn threshold_greater_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $level,
            ) -> Result<()> {
                $greater(stream_context, source, destination, level)
            }

            fn threshold_greater_complex_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $level,
            ) -> Result<()> {
                $greater_in_place(stream_context, signal, level)
            }
        }
    };
}

macro_rules! impl_value_threshold_signal {
    (
        $ty:ty,
        $less:path,
        $less_in_place:path,
        $greater:path,
        $greater_in_place:path
    ) => {
        impl<'a> ValueThresholdSignal<$ty> for SignalPipeline<'a, $ty> {
            fn threshold_less_value_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $ty,
                value: $ty,
            ) -> Result<()> {
                $less(stream_context, source, destination, level, value)
            }

            fn threshold_less_value_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $ty,
                value: $ty,
            ) -> Result<()> {
                $less_in_place(stream_context, signal, level, value)
            }

            fn threshold_greater_value_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $ty,
                value: $ty,
            ) -> Result<()> {
                $greater(stream_context, source, destination, level, value)
            }

            fn threshold_greater_value_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $ty,
                value: $ty,
            ) -> Result<()> {
                $greater_in_place(stream_context, signal, level, value)
            }
        }
    };
}

macro_rules! impl_complex_value_threshold_signal {
    (
        $ty:ty,
        $level:ty,
        $less:path,
        $less_in_place:path,
        $greater:path,
        $greater_in_place:path
    ) => {
        impl<'a> ComplexValueThresholdSignal<$ty, $level> for SignalPipeline<'a, $ty> {
            fn threshold_less_value_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $level,
                value: $ty,
            ) -> Result<()> {
                $less(stream_context, source, destination, level, value)
            }

            fn threshold_less_value_complex_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $level,
                value: $ty,
            ) -> Result<()> {
                $less_in_place(stream_context, signal, level, value)
            }

            fn threshold_greater_value_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                level: $level,
                value: $ty,
            ) -> Result<()> {
                $greater(stream_context, source, destination, level, value)
            }

            fn threshold_greater_value_complex_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                level: $level,
                value: $ty,
            ) -> Result<()> {
                $greater_in_place(stream_context, signal, level, value)
            }
        }
    };
}
