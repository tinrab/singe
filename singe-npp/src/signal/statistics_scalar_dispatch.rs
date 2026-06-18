use super::*;

macro_rules! impl_signal_statistic_to_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($source_ty:ty => ($destination_ty:ty, $direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$destination_ty> for $source_ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
                    $direct_buffer(stream_context, source)
                }

                fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source, destination, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*

        pub fn $buffer_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
        ) -> Result<usize>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$scratch_method(stream_context, source, destination, scratch)
        }

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, destination)
        }
    };
}

macro_rules! impl_signal_scaled_statistic_to_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($source_ty:ty => ($destination_ty:ty, $direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
                scale_factor: i32,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$destination_ty> for $source_ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
                    $direct_buffer(stream_context, source)
                }

                fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source, destination, scale_factor, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, scale_factor)
                }
            }
        )*

        pub fn $buffer_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
        ) -> Result<usize>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$scratch_method(stream_context, source, destination, scale_factor, scratch)
        }

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, destination, scale_factor)
        }
    };
}

impl_signal_statistic_to_dispatch!(
    SumTo,
    sum_buffer_size,
    sum_to_device_with_scratch,
    sum_to_device,
    sum_buffer_size,
    sum_to_device_with_scratch,
    sum_to_device,
    [
        f32 => (f32, sum_f32_buffer_size, sum_f32_to_device_with_scratch, sum_f32_to_device),
        Complex32 => (Complex32, sum_f32_complex_buffer_size, sum_f32_complex_to_device_with_scratch, sum_f32_complex_to_device),
        f64 => (f64, sum_f64_buffer_size, sum_f64_to_device_with_scratch, sum_f64_to_device),
        Complex64 => (Complex64, sum_f64_complex_buffer_size, sum_f64_complex_to_device_with_scratch, sum_f64_complex_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    SumScaledTo,
    sum_scaled_buffer_size,
    sum_scaled_to_device_with_scratch,
    sum_scaled_to_device,
    sum_scaled_buffer_size,
    sum_scaled_to_device_with_scratch,
    sum_scaled_to_device,
    [
        i16 => (i16, sum_i16_scaled_buffer_size, sum_i16_scaled_to_device_with_scratch, sum_i16_scaled_to_device),
        i16 => (i32, sum_i16_to_i32_scaled_buffer_size, sum_i16_to_i32_scaled_to_device_with_scratch, sum_i16_to_i32_scaled_to_device),
        i32 => (i32, sum_i32_scaled_buffer_size, sum_i32_scaled_to_device_with_scratch, sum_i32_scaled_to_device),
        ComplexI16 => (ComplexI16, sum_i16_complex_scaled_buffer_size, sum_i16_complex_scaled_to_device_with_scratch, sum_i16_complex_scaled_to_device),
        ComplexI16 => (ComplexI32, sum_i16_complex_to_i32_complex_scaled_buffer_size, sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch, sum_i16_complex_to_i32_complex_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    SumNaturalLogarithmTo,
    sum_natural_logarithm_buffer_size,
    sum_natural_logarithm_to_device_with_scratch,
    sum_natural_logarithm_to_device,
    sum_natural_logarithm_buffer_size,
    sum_natural_logarithm_to_device_with_scratch,
    sum_natural_logarithm_to_device,
    [
        f32 => (f32, sum_natural_logarithm_f32_buffer_size, sum_natural_logarithm_f32_to_device_with_scratch, sum_natural_logarithm_f32_to_device),
        f32 => (f64, sum_natural_logarithm_f32_to_f64_buffer_size, sum_natural_logarithm_f32_to_f64_to_device_with_scratch, sum_natural_logarithm_f32_to_f64_to_device),
        f64 => (f64, sum_natural_logarithm_f64_buffer_size, sum_natural_logarithm_f64_to_device_with_scratch, sum_natural_logarithm_f64_to_device),
        i16 => (f32, sum_natural_logarithm_i16_to_f32_buffer_size, sum_natural_logarithm_i16_to_f32_to_device_with_scratch, sum_natural_logarithm_i16_to_f32_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    MeanTo,
    mean_buffer_size,
    mean_to_device_with_scratch,
    mean_to_device,
    mean_buffer_size,
    mean_to_device_with_scratch,
    mean_to_device,
    [
        f32 => (f32, mean_f32_buffer_size, mean_f32_to_device_with_scratch, mean_f32_to_device),
        Complex32 => (Complex32, mean_f32_complex_buffer_size, mean_f32_complex_to_device_with_scratch, mean_f32_complex_to_device),
        f64 => (f64, mean_f64_buffer_size, mean_f64_to_device_with_scratch, mean_f64_to_device),
        Complex64 => (Complex64, mean_f64_complex_buffer_size, mean_f64_complex_to_device_with_scratch, mean_f64_complex_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    MeanScaledTo,
    mean_scaled_buffer_size,
    mean_scaled_to_device_with_scratch,
    mean_scaled_to_device,
    mean_scaled_buffer_size,
    mean_scaled_to_device_with_scratch,
    mean_scaled_to_device,
    [
        i16 => (i16, mean_i16_scaled_buffer_size, mean_i16_scaled_to_device_with_scratch, mean_i16_scaled_to_device),
        i32 => (i32, mean_i32_scaled_buffer_size, mean_i32_scaled_to_device_with_scratch, mean_i32_scaled_to_device),
        ComplexI16 => (ComplexI16, mean_i16_complex_scaled_buffer_size, mean_i16_complex_scaled_to_device_with_scratch, mean_i16_complex_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    StandardDeviationTo,
    standard_deviation_buffer_size,
    standard_deviation_to_device_with_scratch,
    standard_deviation_to_device,
    standard_deviation_buffer_size,
    standard_deviation_to_device_with_scratch,
    standard_deviation_to_device,
    [
        f32 => (f32, standard_deviation_f32_buffer_size, standard_deviation_f32_to_device_with_scratch, standard_deviation_f32_to_device),
        f64 => (f64, standard_deviation_f64_buffer_size, standard_deviation_f64_to_device_with_scratch, standard_deviation_f64_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    StandardDeviationScaledTo,
    standard_deviation_scaled_buffer_size,
    standard_deviation_scaled_to_device_with_scratch,
    standard_deviation_scaled_to_device,
    standard_deviation_scaled_buffer_size,
    standard_deviation_scaled_to_device_with_scratch,
    standard_deviation_scaled_to_device,
    [
        i16 => (i16, standard_deviation_i16_scaled_buffer_size, standard_deviation_i16_scaled_to_device_with_scratch, standard_deviation_i16_scaled_to_device),
        i16 => (i32, standard_deviation_i16_to_i32_scaled_buffer_size, standard_deviation_i16_to_i32_scaled_to_device_with_scratch, standard_deviation_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    NormInfTo,
    norm_inf_buffer_size,
    norm_inf_to_device_with_scratch,
    norm_inf_to_device,
    norm_inf_buffer_size,
    norm_inf_to_device_with_scratch,
    norm_inf_to_device,
    [
        f32 => (f32, norm_inf_f32_buffer_size, norm_inf_f32_to_device_with_scratch, norm_inf_f32_to_device),
        f64 => (f64, norm_inf_f64_buffer_size, norm_inf_f64_to_device_with_scratch, norm_inf_f64_to_device),
        Complex32 => (f32, norm_inf_f32_complex_to_f32_buffer_size, norm_inf_f32_complex_to_f32_to_device_with_scratch, norm_inf_f32_complex_to_f32_to_device),
        Complex64 => (f64, norm_inf_f64_complex_to_f64_buffer_size, norm_inf_f64_complex_to_f64_to_device_with_scratch, norm_inf_f64_complex_to_f64_to_device),
        i16 => (f32, norm_inf_i16_to_f32_buffer_size, norm_inf_i16_to_f32_to_device_with_scratch, norm_inf_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    NormInfScaledTo,
    norm_inf_scaled_buffer_size,
    norm_inf_scaled_to_device_with_scratch,
    norm_inf_scaled_to_device,
    norm_inf_scaled_buffer_size,
    norm_inf_scaled_to_device_with_scratch,
    norm_inf_scaled_to_device,
    [
        i16 => (i32, norm_inf_i16_to_i32_scaled_buffer_size, norm_inf_i16_to_i32_scaled_to_device_with_scratch, norm_inf_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    NormL1To,
    norm_l1_buffer_size,
    norm_l1_to_device_with_scratch,
    norm_l1_to_device,
    norm_l1_buffer_size,
    norm_l1_to_device_with_scratch,
    norm_l1_to_device,
    [
        f32 => (f32, norm_l1_f32_buffer_size, norm_l1_f32_to_device_with_scratch, norm_l1_f32_to_device),
        f64 => (f64, norm_l1_f64_buffer_size, norm_l1_f64_to_device_with_scratch, norm_l1_f64_to_device),
        Complex32 => (f64, norm_l1_f32_complex_to_f64_buffer_size, norm_l1_f32_complex_to_f64_to_device_with_scratch, norm_l1_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_l1_f64_complex_to_f64_buffer_size, norm_l1_f64_complex_to_f64_to_device_with_scratch, norm_l1_f64_complex_to_f64_to_device),
        i16 => (f32, norm_l1_i16_to_f32_buffer_size, norm_l1_i16_to_f32_to_device_with_scratch, norm_l1_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    NormL1ScaledTo,
    norm_l1_scaled_buffer_size,
    norm_l1_scaled_to_device_with_scratch,
    norm_l1_scaled_to_device,
    norm_l1_scaled_buffer_size,
    norm_l1_scaled_to_device_with_scratch,
    norm_l1_scaled_to_device,
    [
        i16 => (i32, norm_l1_i16_to_i32_scaled_buffer_size, norm_l1_i16_to_i32_scaled_to_device_with_scratch, norm_l1_i16_to_i32_scaled_to_device),
        i16 => (i64, norm_l1_i16_to_i64_scaled_buffer_size, norm_l1_i16_to_i64_scaled_to_device_with_scratch, norm_l1_i16_to_i64_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    NormL2To,
    norm_l2_buffer_size,
    norm_l2_to_device_with_scratch,
    norm_l2_to_device,
    norm_l2_buffer_size,
    norm_l2_to_device_with_scratch,
    norm_l2_to_device,
    [
        f32 => (f32, norm_l2_f32_buffer_size, norm_l2_f32_to_device_with_scratch, norm_l2_f32_to_device),
        f64 => (f64, norm_l2_f64_buffer_size, norm_l2_f64_to_device_with_scratch, norm_l2_f64_to_device),
        Complex32 => (f64, norm_l2_f32_complex_to_f64_buffer_size, norm_l2_f32_complex_to_f64_to_device_with_scratch, norm_l2_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_l2_f64_complex_to_f64_buffer_size, norm_l2_f64_complex_to_f64_to_device_with_scratch, norm_l2_f64_complex_to_f64_to_device),
        i16 => (f32, norm_l2_i16_to_f32_buffer_size, norm_l2_i16_to_f32_to_device_with_scratch, norm_l2_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    NormL2ScaledTo,
    norm_l2_scaled_buffer_size,
    norm_l2_scaled_to_device_with_scratch,
    norm_l2_scaled_to_device,
    norm_l2_scaled_buffer_size,
    norm_l2_scaled_to_device_with_scratch,
    norm_l2_scaled_to_device,
    [
        i16 => (i32, norm_l2_i16_to_i32_scaled_buffer_size, norm_l2_i16_to_i32_scaled_to_device_with_scratch, norm_l2_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_scaled_statistic_to_dispatch!(
    NormL2SquaredScaledTo,
    norm_l2_squared_scaled_buffer_size,
    norm_l2_squared_scaled_to_device_with_scratch,
    norm_l2_squared_scaled_to_device,
    norm_l2_squared_scaled_buffer_size,
    norm_l2_squared_scaled_to_device_with_scratch,
    norm_l2_squared_scaled_to_device,
    [
        i16 => (i64, norm_l2_squared_i16_to_i64_scaled_buffer_size, norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch, norm_l2_squared_i16_to_i64_scaled_to_device)
    ]
);
