use super::*;

macro_rules! impl_signal_binary_statistic_to_dispatch {
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
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
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
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source_1, source_2, destination, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, destination)
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
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$scratch_method(stream_context, source_1, source_2, destination, scratch)
        }

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source_1, source_2, destination)
        }
    };
}

macro_rules! impl_signal_scaled_binary_statistic_to_dispatch {
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
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
                scale_factor: i32,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
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
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(
                        stream_context,
                        source_1,
                        source_2,
                        destination,
                        scale_factor,
                        scratch,
                    )
                }

                fn $method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, destination, scale_factor)
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
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$scratch_method(
                stream_context,
                source_1,
                source_2,
                destination,
                scale_factor,
                scratch,
            )
        }

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, Source>,
            source_2: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source_1, source_2, destination, scale_factor)
        }
    };
}

impl_signal_binary_statistic_to_dispatch!(
    NormDiffInfTo,
    norm_diff_inf_buffer_size,
    norm_diff_inf_to_device_with_scratch,
    norm_diff_inf_to_device,
    norm_diff_inf_buffer_size,
    norm_diff_inf_to_device_with_scratch,
    norm_diff_inf_to_device,
    [
        f32 => (f32, norm_diff_inf_f32_buffer_size, norm_diff_inf_f32_to_device_with_scratch, norm_diff_inf_f32_to_device),
        f64 => (f64, norm_diff_inf_f64_buffer_size, norm_diff_inf_f64_to_device_with_scratch, norm_diff_inf_f64_to_device),
        Complex32 => (f32, norm_diff_inf_f32_complex_to_f32_buffer_size, norm_diff_inf_f32_complex_to_f32_to_device_with_scratch, norm_diff_inf_f32_complex_to_f32_to_device),
        Complex64 => (f64, norm_diff_inf_f64_complex_to_f64_buffer_size, norm_diff_inf_f64_complex_to_f64_to_device_with_scratch, norm_diff_inf_f64_complex_to_f64_to_device),
        i16 => (f32, norm_diff_inf_i16_to_f32_buffer_size, norm_diff_inf_i16_to_f32_to_device_with_scratch, norm_diff_inf_i16_to_f32_to_device)
    ]
);

impl_signal_binary_statistic_to_dispatch!(
    NormDiffL1To,
    norm_diff_l1_buffer_size,
    norm_diff_l1_to_device_with_scratch,
    norm_diff_l1_to_device,
    norm_diff_l1_buffer_size,
    norm_diff_l1_to_device_with_scratch,
    norm_diff_l1_to_device,
    [
        f32 => (f32, norm_diff_l1_f32_buffer_size, norm_diff_l1_f32_to_device_with_scratch, norm_diff_l1_f32_to_device),
        f64 => (f64, norm_diff_l1_f64_buffer_size, norm_diff_l1_f64_to_device_with_scratch, norm_diff_l1_f64_to_device),
        Complex32 => (f64, norm_diff_l1_f32_complex_to_f64_buffer_size, norm_diff_l1_f32_complex_to_f64_to_device_with_scratch, norm_diff_l1_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_diff_l1_f64_complex_to_f64_buffer_size, norm_diff_l1_f64_complex_to_f64_to_device_with_scratch, norm_diff_l1_f64_complex_to_f64_to_device),
        i16 => (f32, norm_diff_l1_i16_to_f32_buffer_size, norm_diff_l1_i16_to_f32_to_device_with_scratch, norm_diff_l1_i16_to_f32_to_device)
    ]
);

impl_signal_binary_statistic_to_dispatch!(
    NormDiffL2To,
    norm_diff_l2_buffer_size,
    norm_diff_l2_to_device_with_scratch,
    norm_diff_l2_to_device,
    norm_diff_l2_buffer_size,
    norm_diff_l2_to_device_with_scratch,
    norm_diff_l2_to_device,
    [
        f32 => (f32, norm_diff_l2_f32_buffer_size, norm_diff_l2_f32_to_device_with_scratch, norm_diff_l2_f32_to_device),
        f64 => (f64, norm_diff_l2_f64_buffer_size, norm_diff_l2_f64_to_device_with_scratch, norm_diff_l2_f64_to_device),
        Complex32 => (f64, norm_diff_l2_f32_complex_to_f64_buffer_size, norm_diff_l2_f32_complex_to_f64_to_device_with_scratch, norm_diff_l2_f32_complex_to_f64_to_device),
        Complex64 => (f64, norm_diff_l2_f64_complex_to_f64_buffer_size, norm_diff_l2_f64_complex_to_f64_to_device_with_scratch, norm_diff_l2_f64_complex_to_f64_to_device),
        i16 => (f32, norm_diff_l2_i16_to_f32_buffer_size, norm_diff_l2_i16_to_f32_to_device_with_scratch, norm_diff_l2_i16_to_f32_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!(
    NormDiffInfScaledTo,
    norm_diff_inf_scaled_buffer_size,
    norm_diff_inf_scaled_to_device_with_scratch,
    norm_diff_inf_scaled_to_device,
    norm_diff_inf_scaled_buffer_size,
    norm_diff_inf_scaled_to_device_with_scratch,
    norm_diff_inf_scaled_to_device,
    [
        i16 => (i32, norm_diff_inf_i16_to_i32_scaled_buffer_size, norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch, norm_diff_inf_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!(
    NormDiffL1ScaledTo,
    norm_diff_l1_scaled_buffer_size,
    norm_diff_l1_scaled_to_device_with_scratch,
    norm_diff_l1_scaled_to_device,
    norm_diff_l1_scaled_buffer_size,
    norm_diff_l1_scaled_to_device_with_scratch,
    norm_diff_l1_scaled_to_device,
    [
        i16 => (i32, norm_diff_l1_i16_to_i32_scaled_buffer_size, norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch, norm_diff_l1_i16_to_i32_scaled_to_device),
        i16 => (i64, norm_diff_l1_i16_to_i64_scaled_buffer_size, norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch, norm_diff_l1_i16_to_i64_scaled_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!(
    NormDiffL2ScaledTo,
    norm_diff_l2_scaled_buffer_size,
    norm_diff_l2_scaled_to_device_with_scratch,
    norm_diff_l2_scaled_to_device,
    norm_diff_l2_scaled_buffer_size,
    norm_diff_l2_scaled_to_device_with_scratch,
    norm_diff_l2_scaled_to_device,
    [
        i16 => (i32, norm_diff_l2_i16_to_i32_scaled_buffer_size, norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch, norm_diff_l2_i16_to_i32_scaled_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!(
    NormDiffL2SquaredScaledTo,
    norm_diff_l2_squared_scaled_buffer_size,
    norm_diff_l2_squared_scaled_to_device_with_scratch,
    norm_diff_l2_squared_scaled_to_device,
    norm_diff_l2_squared_scaled_buffer_size,
    norm_diff_l2_squared_scaled_to_device_with_scratch,
    norm_diff_l2_squared_scaled_to_device,
    [
        i16 => (i64, norm_diff_l2_squared_i16_to_i64_scaled_buffer_size, norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch, norm_diff_l2_squared_i16_to_i64_scaled_to_device)
    ]
);

impl_signal_binary_statistic_to_dispatch!(
    DotProductTo,
    dot_product_buffer_size,
    dot_product_to_device_with_scratch,
    dot_product_to_device,
    dot_product_buffer_size,
    dot_product_to_device_with_scratch,
    dot_product_to_device,
    [
        f32 => (f32, dot_product_f32_buffer_size, dot_product_f32_to_device_with_scratch, dot_product_f32_to_device),
        f32 => (f64, dot_product_f32_to_f64_buffer_size, dot_product_f32_to_f64_to_device_with_scratch, dot_product_f32_to_f64_to_device),
        f64 => (f64, dot_product_f64_buffer_size, dot_product_f64_to_device_with_scratch, dot_product_f64_to_device),
        Complex32 => (Complex32, dot_product_f32_complex_buffer_size, dot_product_f32_complex_to_device_with_scratch, dot_product_f32_complex_to_device),
        Complex32 => (Complex64, dot_product_f32_complex_to_f64_complex_buffer_size, dot_product_f32_complex_to_f64_complex_to_device_with_scratch, dot_product_f32_complex_to_f64_complex_to_device),
        Complex64 => (Complex64, dot_product_f64_complex_buffer_size, dot_product_f64_complex_to_device_with_scratch, dot_product_f64_complex_to_device),
        i16 => (i64, dot_product_i16_to_i64_buffer_size, dot_product_i16_to_i64_to_device_with_scratch, dot_product_i16_to_i64_to_device),
        i16 => (f32, dot_product_i16_to_f32_buffer_size, dot_product_i16_to_f32_to_device_with_scratch, dot_product_i16_to_f32_to_device),
        ComplexI16 => (ComplexI64, dot_product_i16_complex_to_i64_complex_buffer_size, dot_product_i16_complex_to_i64_complex_to_device_with_scratch, dot_product_i16_complex_to_i64_complex_to_device),
        ComplexI16 => (Complex32, dot_product_i16_complex_to_f32_complex_buffer_size, dot_product_i16_complex_to_f32_complex_to_device_with_scratch, dot_product_i16_complex_to_f32_complex_to_device)
    ]
);

impl_signal_scaled_binary_statistic_to_dispatch!(
    DotProductScaledTo,
    dot_product_scaled_buffer_size,
    dot_product_scaled_to_device_with_scratch,
    dot_product_scaled_to_device,
    dot_product_scaled_buffer_size,
    dot_product_scaled_to_device_with_scratch,
    dot_product_scaled_to_device,
    [
        i16 => (i16, dot_product_i16_scaled_buffer_size, dot_product_i16_scaled_to_device_with_scratch, dot_product_i16_scaled_to_device),
        i16 => (i32, dot_product_i16_to_i32_scaled_buffer_size, dot_product_i16_to_i32_scaled_to_device_with_scratch, dot_product_i16_to_i32_scaled_to_device),
        i32 => (i32, dot_product_i32_scaled_buffer_size, dot_product_i32_scaled_to_device_with_scratch, dot_product_i32_scaled_to_device),
        ComplexI16 => (ComplexI16, dot_product_i16_complex_scaled_buffer_size, dot_product_i16_complex_scaled_to_device_with_scratch, dot_product_i16_complex_scaled_to_device),
        ComplexI16 => (ComplexI32, dot_product_i16_complex_to_i32_complex_scaled_buffer_size, dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch, dot_product_i16_complex_to_i32_complex_scaled_to_device),
        ComplexI32 => (ComplexI32, dot_product_i32_complex_scaled_buffer_size, dot_product_i32_complex_scaled_to_device_with_scratch, dot_product_i32_complex_scaled_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    MaxTo,
    max_buffer_size,
    max_to_device_with_scratch,
    max_to_device,
    max_buffer_size,
    max_to_device_with_scratch,
    max_to_device,
    [
        i16 => (i16, max_i16_buffer_size, max_i16_to_device_with_scratch, max_i16_to_device),
        i32 => (i32, max_i32_buffer_size, max_i32_to_device_with_scratch, max_i32_to_device),
        f32 => (f32, max_f32_buffer_size, max_f32_to_device_with_scratch, max_f32_to_device),
        f64 => (f64, max_f64_buffer_size, max_f64_to_device_with_scratch, max_f64_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    MinTo,
    min_buffer_size,
    min_to_device_with_scratch,
    min_to_device,
    min_buffer_size,
    min_to_device_with_scratch,
    min_to_device,
    [
        i16 => (i16, min_i16_buffer_size, min_i16_to_device_with_scratch, min_i16_to_device),
        i32 => (i32, min_i32_buffer_size, min_i32_to_device_with_scratch, min_i32_to_device),
        f32 => (f32, min_f32_buffer_size, min_f32_to_device_with_scratch, min_f32_to_device),
        f64 => (f64, min_f64_buffer_size, min_f64_to_device_with_scratch, min_f64_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    MaxAbsoluteTo,
    max_absolute_buffer_size,
    max_absolute_to_device_with_scratch,
    max_absolute_to_device,
    max_absolute_buffer_size,
    max_absolute_to_device_with_scratch,
    max_absolute_to_device,
    [
        i16 => (i16, max_absolute_i16_buffer_size, max_absolute_i16_to_device_with_scratch, max_absolute_i16_to_device),
        i32 => (i32, max_absolute_i32_buffer_size, max_absolute_i32_to_device_with_scratch, max_absolute_i32_to_device)
    ]
);

impl_signal_statistic_to_dispatch!(
    MinAbsoluteTo,
    min_absolute_buffer_size,
    min_absolute_to_device_with_scratch,
    min_absolute_to_device,
    min_absolute_buffer_size,
    min_absolute_to_device_with_scratch,
    min_absolute_to_device,
    [
        i16 => (i16, min_absolute_i16_buffer_size, min_absolute_i16_to_device_with_scratch, min_absolute_i16_to_device),
        i32 => (i32, min_absolute_i32_buffer_size, min_absolute_i32_to_device_with_scratch, min_absolute_i32_to_device)
    ]
);
