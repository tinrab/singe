use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        statistics::*,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike, ZeroCrossingType},
    workspace::ScratchBuffer,
};

#[path = "statistics_scalar_dispatch.rs"]
#[macro_use]
mod scalar_dispatch;
pub use scalar_dispatch::*;

#[path = "statistics_binary_dispatch.rs"]
mod binary_dispatch;
pub use binary_dispatch::*;

macro_rules! impl_signal_indexed_statistic_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait: DataTypeLike {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: &mut SignalViewMut<'_, Self>,
                index: &mut SignalViewMut<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: &mut SignalViewMut<'_, Self>,
                index: &mut SignalViewMut<'_, i32>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
                    $direct_buffer(stream_context, source)
                }

                fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: &mut SignalViewMut<'_, Self>,
                    index: &mut SignalViewMut<'_, i32>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source, value, index, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: &mut SignalViewMut<'_, Self>,
                    index: &mut SignalViewMut<'_, i32>,
                ) -> Result<()> {
                    $direct(stream_context, source, value, index)
                }
            }
        )*

        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
            T::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: &mut SignalViewMut<'_, T>,
            index: &mut SignalViewMut<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            T::$scratch_method(stream_context, source, value, index, scratch)
        }

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: &mut SignalViewMut<'_, T>,
            index: &mut SignalViewMut<'_, i32>,
        ) -> Result<()> {
            T::$method(stream_context, source, value, index)
        }
    };
}

impl_signal_indexed_statistic_dispatch!(
    MaxIndex,
    max_index_buffer_size,
    max_index_to_device_with_scratch,
    max_index_to_device,
    max_index_buffer_size,
    max_index_to_device_with_scratch,
    max_index_to_device,
    [
        i16 => (max_index_i16_buffer_size, max_index_i16_to_device_with_scratch, max_index_i16_to_device),
        i32 => (max_index_i32_buffer_size, max_index_i32_to_device_with_scratch, max_index_i32_to_device),
        f32 => (max_index_f32_buffer_size, max_index_f32_to_device_with_scratch, max_index_f32_to_device),
        f64 => (max_index_f64_buffer_size, max_index_f64_to_device_with_scratch, max_index_f64_to_device)
    ]
);

impl_signal_indexed_statistic_dispatch!(
    MinIndex,
    min_index_buffer_size,
    min_index_to_device_with_scratch,
    min_index_to_device,
    min_index_buffer_size,
    min_index_to_device_with_scratch,
    min_index_to_device,
    [
        i16 => (min_index_i16_buffer_size, min_index_i16_to_device_with_scratch, min_index_i16_to_device),
        i32 => (min_index_i32_buffer_size, min_index_i32_to_device_with_scratch, min_index_i32_to_device),
        f32 => (min_index_f32_buffer_size, min_index_f32_to_device_with_scratch, min_index_f32_to_device),
        f64 => (min_index_f64_buffer_size, min_index_f64_to_device_with_scratch, min_index_f64_to_device)
    ]
);

impl_signal_indexed_statistic_dispatch!(
    MaxAbsoluteIndex,
    max_absolute_index_buffer_size,
    max_absolute_index_to_device_with_scratch,
    max_absolute_index_to_device,
    max_absolute_index_buffer_size,
    max_absolute_index_to_device_with_scratch,
    max_absolute_index_to_device,
    [
        i16 => (max_absolute_index_i16_buffer_size, max_absolute_index_i16_to_device_with_scratch, max_absolute_index_i16_to_device),
        i32 => (max_absolute_index_i32_buffer_size, max_absolute_index_i32_to_device_with_scratch, max_absolute_index_i32_to_device)
    ]
);

impl_signal_indexed_statistic_dispatch!(
    MinAbsoluteIndex,
    min_absolute_index_buffer_size,
    min_absolute_index_to_device_with_scratch,
    min_absolute_index_to_device,
    min_absolute_index_buffer_size,
    min_absolute_index_to_device_with_scratch,
    min_absolute_index_to_device,
    [
        i16 => (min_absolute_index_i16_buffer_size, min_absolute_index_i16_to_device_with_scratch, min_absolute_index_i16_to_device),
        i32 => (min_absolute_index_i32_buffer_size, min_absolute_index_i32_to_device_with_scratch, min_absolute_index_i32_to_device)
    ]
);

macro_rules! impl_signal_pair_statistic_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait: DataTypeLike {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalViewMut<'_, Self>,
                max: &mut SignalViewMut<'_, Self>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalViewMut<'_, Self>,
                max: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
                    $direct_buffer(stream_context, source)
                }

                fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalViewMut<'_, Self>,
                    max: &mut SignalViewMut<'_, Self>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source, min, max, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalViewMut<'_, Self>,
                    max: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, min, max)
                }
            }
        )*

        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
            T::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalViewMut<'_, T>,
            max: &mut SignalViewMut<'_, T>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            T::$scratch_method(stream_context, source, min, max, scratch)
        }

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalViewMut<'_, T>,
            max: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, min, max)
        }
    };
}

macro_rules! impl_signal_indexed_pair_statistic_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait: DataTypeLike {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalViewMut<'_, Self>,
                min_index: &mut SignalViewMut<'_, i32>,
                max: &mut SignalViewMut<'_, Self>,
                max_index: &mut SignalViewMut<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalViewMut<'_, Self>,
                min_index: &mut SignalViewMut<'_, i32>,
                max: &mut SignalViewMut<'_, Self>,
                max_index: &mut SignalViewMut<'_, i32>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
                    $direct_buffer(stream_context, source)
                }

                fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalViewMut<'_, Self>,
                    min_index: &mut SignalViewMut<'_, i32>,
                    max: &mut SignalViewMut<'_, Self>,
                    max_index: &mut SignalViewMut<'_, i32>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(
                        stream_context,
                        source,
                        min,
                        min_index,
                        max,
                        max_index,
                        scratch,
                    )
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalViewMut<'_, Self>,
                    min_index: &mut SignalViewMut<'_, i32>,
                    max: &mut SignalViewMut<'_, Self>,
                    max_index: &mut SignalViewMut<'_, i32>,
                ) -> Result<()> {
                    $direct(stream_context, source, min, min_index, max, max_index)
                }
            }
        )*

        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
            T::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalViewMut<'_, T>,
            min_index: &mut SignalViewMut<'_, i32>,
            max: &mut SignalViewMut<'_, T>,
            max_index: &mut SignalViewMut<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            T::$scratch_method(
                stream_context,
                source,
                min,
                min_index,
                max,
                max_index,
                scratch,
            )
        }

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalViewMut<'_, T>,
            min_index: &mut SignalViewMut<'_, i32>,
            max: &mut SignalViewMut<'_, T>,
            max_index: &mut SignalViewMut<'_, i32>,
        ) -> Result<()> {
            T::$method(stream_context, source, min, min_index, max, max_index)
        }
    };
}

impl_signal_pair_statistic_dispatch!(
    MinMax,
    min_max_buffer_size,
    min_max_to_device_with_scratch,
    min_max_to_device,
    min_max_buffer_size,
    min_max_to_device_with_scratch,
    min_max_to_device,
    [
        u8 => (min_max_u8_buffer_size, min_max_u8_to_device_with_scratch, min_max_u8_to_device),
        u16 => (min_max_u16_buffer_size, min_max_u16_to_device_with_scratch, min_max_u16_to_device),
        i16 => (min_max_i16_buffer_size, min_max_i16_to_device_with_scratch, min_max_i16_to_device),
        u32 => (min_max_u32_buffer_size, min_max_u32_to_device_with_scratch, min_max_u32_to_device),
        i32 => (min_max_i32_buffer_size, min_max_i32_to_device_with_scratch, min_max_i32_to_device),
        f32 => (min_max_f32_buffer_size, min_max_f32_to_device_with_scratch, min_max_f32_to_device),
        f64 => (min_max_f64_buffer_size, min_max_f64_to_device_with_scratch, min_max_f64_to_device)
    ]
);

impl_signal_indexed_pair_statistic_dispatch!(
    MinMaxIndex,
    min_max_index_buffer_size,
    min_max_index_to_device_with_scratch,
    min_max_index_to_device,
    min_max_index_buffer_size,
    min_max_index_to_device_with_scratch,
    min_max_index_to_device,
    [
        u8 => (min_max_index_u8_buffer_size, min_max_index_u8_to_device_with_scratch, min_max_index_u8_to_device),
        u16 => (min_max_index_u16_buffer_size, min_max_index_u16_to_device_with_scratch, min_max_index_u16_to_device),
        i16 => (min_max_index_i16_buffer_size, min_max_index_i16_to_device_with_scratch, min_max_index_i16_to_device),
        u32 => (min_max_index_u32_buffer_size, min_max_index_u32_to_device_with_scratch, min_max_index_u32_to_device),
        i32 => (min_max_index_i32_buffer_size, min_max_index_i32_to_device_with_scratch, min_max_index_i32_to_device),
        f32 => (min_max_index_f32_buffer_size, min_max_index_f32_to_device_with_scratch, min_max_index_f32_to_device),
        f64 => (min_max_index_f64_buffer_size, min_max_index_f64_to_device_with_scratch, min_max_index_f64_to_device)
    ]
);

macro_rules! impl_signal_mean_standard_deviation_dispatch {
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
                mean: &mut SignalViewMut<'_, Destination>,
                standard_deviation: &mut SignalViewMut<'_, Destination>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                mean: &mut SignalViewMut<'_, Destination>,
                standard_deviation: &mut SignalViewMut<'_, Destination>,
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
                    mean: &mut SignalViewMut<'_, $destination_ty>,
                    standard_deviation: &mut SignalViewMut<'_, $destination_ty>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source, mean, standard_deviation, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    mean: &mut SignalViewMut<'_, $destination_ty>,
                    standard_deviation: &mut SignalViewMut<'_, $destination_ty>,
                ) -> Result<()> {
                    $direct(stream_context, source, mean, standard_deviation)
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
            mean: &mut SignalViewMut<'_, Destination>,
            standard_deviation: &mut SignalViewMut<'_, Destination>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$scratch_method(stream_context, source, mean, standard_deviation, scratch)
        }

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            mean: &mut SignalViewMut<'_, Destination>,
            standard_deviation: &mut SignalViewMut<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, mean, standard_deviation)
        }
    };
}

macro_rules! impl_signal_scaled_mean_standard_deviation_dispatch {
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
                mean: &mut SignalViewMut<'_, Destination>,
                standard_deviation: &mut SignalViewMut<'_, Destination>,
                scale_factor: i32,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                mean: &mut SignalViewMut<'_, Destination>,
                standard_deviation: &mut SignalViewMut<'_, Destination>,
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
                    mean: &mut SignalViewMut<'_, $destination_ty>,
                    standard_deviation: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(
                        stream_context,
                        source,
                        mean,
                        standard_deviation,
                        scale_factor,
                        scratch,
                    )
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    mean: &mut SignalViewMut<'_, $destination_ty>,
                    standard_deviation: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, mean, standard_deviation, scale_factor)
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
            mean: &mut SignalViewMut<'_, Destination>,
            standard_deviation: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$scratch_method(
                stream_context,
                source,
                mean,
                standard_deviation,
                scale_factor,
                scratch,
            )
        }

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            mean: &mut SignalViewMut<'_, Destination>,
            standard_deviation: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, mean, standard_deviation, scale_factor)
        }
    };
}

impl_signal_mean_standard_deviation_dispatch!(
    MeanStandardDeviationTo,
    mean_standard_deviation_buffer_size,
    mean_standard_deviation_to_device_with_scratch,
    mean_standard_deviation_to_device,
    mean_standard_deviation_buffer_size,
    mean_standard_deviation_to_device_with_scratch,
    mean_standard_deviation_to_device,
    [
        f32 => (f32, mean_standard_deviation_f32_buffer_size, mean_standard_deviation_f32_to_device_with_scratch, mean_standard_deviation_f32_to_device),
        f64 => (f64, mean_standard_deviation_f64_buffer_size, mean_standard_deviation_f64_to_device_with_scratch, mean_standard_deviation_f64_to_device)
    ]
);

impl_signal_scaled_mean_standard_deviation_dispatch!(
    MeanStandardDeviationScaledTo,
    mean_standard_deviation_scaled_buffer_size,
    mean_standard_deviation_scaled_to_device_with_scratch,
    mean_standard_deviation_scaled_to_device,
    mean_standard_deviation_scaled_buffer_size,
    mean_standard_deviation_scaled_to_device_with_scratch,
    mean_standard_deviation_scaled_to_device,
    [
        i16 => (i16, mean_standard_deviation_i16_scaled_buffer_size, mean_standard_deviation_i16_scaled_to_device_with_scratch, mean_standard_deviation_i16_scaled_to_device),
        i16 => (i32, mean_standard_deviation_i16_to_i32_scaled_buffer_size, mean_standard_deviation_i16_to_i32_scaled_to_device_with_scratch, mean_standard_deviation_i16_to_i32_scaled_to_device)
    ]
);

macro_rules! impl_signal_every_statistic_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, destination)
        }
    };
}

impl_signal_every_statistic_dispatch!(MinEveryInPlace, min_every_in_place, min_every_in_place, [
    u8 => min_every_u8_in_place,
    u16 => min_every_u16_in_place,
    i16 => min_every_i16_in_place,
    i32 => min_every_i32_in_place,
    f32 => min_every_f32_in_place,
    f64 => min_every_f64_in_place
]);

impl_signal_every_statistic_dispatch!(MaxEveryInPlace, max_every_in_place, max_every_in_place, [
    u8 => max_every_u8_in_place,
    u16 => max_every_u16_in_place,
    i16 => max_every_i16_in_place,
    i32 => max_every_i32_in_place,
    f32 => max_every_f32_in_place
]);

macro_rules! impl_signal_zero_crossing_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait: DataTypeLike {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, f32>,
                zero_crossing_type: ZeroCrossingType,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, f32>,
                zero_crossing_type: ZeroCrossingType,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
                    $direct_buffer(stream_context, source)
                }

                fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, f32>,
                    zero_crossing_type: ZeroCrossingType,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(
                        stream_context,
                        source,
                        destination,
                        zero_crossing_type,
                        scratch,
                    )
                }

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, f32>,
                    zero_crossing_type: ZeroCrossingType,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, zero_crossing_type)
                }
            }
        )*

        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
            T::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            T::$scratch_method(
                stream_context,
                source,
                destination,
                zero_crossing_type,
                scratch,
            )
        }

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, zero_crossing_type)
        }
    };
}

impl_signal_zero_crossing_dispatch!(
    ZeroCrossingToF32,
    zero_crossing_to_f32_buffer_size,
    zero_crossing_to_f32_to_device_with_scratch,
    zero_crossing_to_f32_to_device,
    zero_crossing_to_f32_buffer_size,
    zero_crossing_to_f32_to_device_with_scratch,
    zero_crossing_to_f32_to_device,
    [
        i16 => (zero_crossing_i16_to_f32_buffer_size, zero_crossing_i16_to_f32_to_device_with_scratch, zero_crossing_i16_to_f32_to_device),
        f32 => (zero_crossing_f32_buffer_size, zero_crossing_f32_to_device_with_scratch, zero_crossing_f32_to_device)
    ]
);

macro_rules! impl_signal_error_metric_dispatch {
    (
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        pub trait $trait: DataTypeLike {
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            fn $scratch_method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, f64>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            fn $method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, f64>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
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
                    destination: &mut SignalViewMut<'_, f64>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
                    $direct_scratch(stream_context, source_1, source_2, destination, scratch)
                }

                fn $method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, f64>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, destination)
                }
            }
        )*

        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
            T::$buffer_method(stream_context, source)
        }

        pub fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, T>,
            source_2: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, f64>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            T::$scratch_method(stream_context, source_1, source_2, destination, scratch)
        }

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, T>,
            source_2: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, f64>,
        ) -> Result<()> {
            T::$method(stream_context, source_1, source_2, destination)
        }
    };
}

impl_signal_error_metric_dispatch!(
    MaximumError,
    maximum_error_buffer_size,
    maximum_error_to_device_with_scratch,
    maximum_error_to_device,
    maximum_error_buffer_size,
    maximum_error_to_device_with_scratch,
    maximum_error_to_device,
    [
        u8 => (maximum_error_u8_buffer_size, maximum_error_u8_to_device_with_scratch, maximum_error_u8_to_device),
        i8 => (maximum_error_i8_buffer_size, maximum_error_i8_to_device_with_scratch, maximum_error_i8_to_device),
        u16 => (maximum_error_u16_buffer_size, maximum_error_u16_to_device_with_scratch, maximum_error_u16_to_device),
        i16 => (maximum_error_i16_buffer_size, maximum_error_i16_to_device_with_scratch, maximum_error_i16_to_device),
        ComplexI16 => (maximum_error_i16_complex_buffer_size, maximum_error_i16_complex_to_device_with_scratch, maximum_error_i16_complex_to_device),
        u32 => (maximum_error_u32_buffer_size, maximum_error_u32_to_device_with_scratch, maximum_error_u32_to_device),
        i32 => (maximum_error_i32_buffer_size, maximum_error_i32_to_device_with_scratch, maximum_error_i32_to_device),
        ComplexI32 => (maximum_error_i32_complex_buffer_size, maximum_error_i32_complex_to_device_with_scratch, maximum_error_i32_complex_to_device),
        i64 => (maximum_error_i64_buffer_size, maximum_error_i64_to_device_with_scratch, maximum_error_i64_to_device),
        ComplexI64 => (maximum_error_i64_complex_buffer_size, maximum_error_i64_complex_to_device_with_scratch, maximum_error_i64_complex_to_device),
        f32 => (maximum_error_f32_buffer_size, maximum_error_f32_to_device_with_scratch, maximum_error_f32_to_device),
        Complex32 => (maximum_error_f32_complex_buffer_size, maximum_error_f32_complex_to_device_with_scratch, maximum_error_f32_complex_to_device),
        f64 => (maximum_error_f64_buffer_size, maximum_error_f64_to_device_with_scratch, maximum_error_f64_to_device),
        Complex64 => (maximum_error_f64_complex_buffer_size, maximum_error_f64_complex_to_device_with_scratch, maximum_error_f64_complex_to_device)
    ]
);

impl_signal_error_metric_dispatch!(
    AverageError,
    average_error_buffer_size,
    average_error_to_device_with_scratch,
    average_error_to_device,
    average_error_buffer_size,
    average_error_to_device_with_scratch,
    average_error_to_device,
    [
        u8 => (average_error_u8_buffer_size, average_error_u8_to_device_with_scratch, average_error_u8_to_device),
        i8 => (average_error_i8_buffer_size, average_error_i8_to_device_with_scratch, average_error_i8_to_device),
        u16 => (average_error_u16_buffer_size, average_error_u16_to_device_with_scratch, average_error_u16_to_device),
        i16 => (average_error_i16_buffer_size, average_error_i16_to_device_with_scratch, average_error_i16_to_device),
        ComplexI16 => (average_error_i16_complex_buffer_size, average_error_i16_complex_to_device_with_scratch, average_error_i16_complex_to_device),
        u32 => (average_error_u32_buffer_size, average_error_u32_to_device_with_scratch, average_error_u32_to_device),
        i32 => (average_error_i32_buffer_size, average_error_i32_to_device_with_scratch, average_error_i32_to_device),
        ComplexI32 => (average_error_i32_complex_buffer_size, average_error_i32_complex_to_device_with_scratch, average_error_i32_complex_to_device),
        i64 => (average_error_i64_buffer_size, average_error_i64_to_device_with_scratch, average_error_i64_to_device),
        ComplexI64 => (average_error_i64_complex_buffer_size, average_error_i64_complex_to_device_with_scratch, average_error_i64_complex_to_device),
        f32 => (average_error_f32_buffer_size, average_error_f32_to_device_with_scratch, average_error_f32_to_device),
        Complex32 => (average_error_f32_complex_buffer_size, average_error_f32_complex_to_device_with_scratch, average_error_f32_complex_to_device),
        f64 => (average_error_f64_buffer_size, average_error_f64_to_device_with_scratch, average_error_f64_to_device),
        Complex64 => (average_error_f64_complex_buffer_size, average_error_f64_complex_to_device_with_scratch, average_error_f64_complex_to_device)
    ]
);

impl_signal_error_metric_dispatch!(
    MaximumRelativeError,
    maximum_relative_error_buffer_size,
    maximum_relative_error_to_device_with_scratch,
    maximum_relative_error_to_device,
    maximum_relative_error_buffer_size,
    maximum_relative_error_to_device_with_scratch,
    maximum_relative_error_to_device,
    [
        u8 => (maximum_relative_error_u8_buffer_size, maximum_relative_error_u8_to_device_with_scratch, maximum_relative_error_u8_to_device),
        i8 => (maximum_relative_error_i8_buffer_size, maximum_relative_error_i8_to_device_with_scratch, maximum_relative_error_i8_to_device),
        u16 => (maximum_relative_error_u16_buffer_size, maximum_relative_error_u16_to_device_with_scratch, maximum_relative_error_u16_to_device),
        i16 => (maximum_relative_error_i16_buffer_size, maximum_relative_error_i16_to_device_with_scratch, maximum_relative_error_i16_to_device),
        ComplexI16 => (maximum_relative_error_i16_complex_buffer_size, maximum_relative_error_i16_complex_to_device_with_scratch, maximum_relative_error_i16_complex_to_device),
        u32 => (maximum_relative_error_u32_buffer_size, maximum_relative_error_u32_to_device_with_scratch, maximum_relative_error_u32_to_device),
        i32 => (maximum_relative_error_i32_buffer_size, maximum_relative_error_i32_to_device_with_scratch, maximum_relative_error_i32_to_device),
        ComplexI32 => (maximum_relative_error_i32_complex_buffer_size, maximum_relative_error_i32_complex_to_device_with_scratch, maximum_relative_error_i32_complex_to_device),
        i64 => (maximum_relative_error_i64_buffer_size, maximum_relative_error_i64_to_device_with_scratch, maximum_relative_error_i64_to_device),
        ComplexI64 => (maximum_relative_error_i64_complex_buffer_size, maximum_relative_error_i64_complex_to_device_with_scratch, maximum_relative_error_i64_complex_to_device),
        f32 => (maximum_relative_error_f32_buffer_size, maximum_relative_error_f32_to_device_with_scratch, maximum_relative_error_f32_to_device),
        Complex32 => (maximum_relative_error_f32_complex_buffer_size, maximum_relative_error_f32_complex_to_device_with_scratch, maximum_relative_error_f32_complex_to_device),
        f64 => (maximum_relative_error_f64_buffer_size, maximum_relative_error_f64_to_device_with_scratch, maximum_relative_error_f64_to_device),
        Complex64 => (maximum_relative_error_f64_complex_buffer_size, maximum_relative_error_f64_complex_to_device_with_scratch, maximum_relative_error_f64_complex_to_device)
    ]
);

impl_signal_error_metric_dispatch!(
    AverageRelativeError,
    average_relative_error_buffer_size,
    average_relative_error_to_device_with_scratch,
    average_relative_error_to_device,
    average_relative_error_buffer_size,
    average_relative_error_to_device_with_scratch,
    average_relative_error_to_device,
    [
        u8 => (average_relative_error_u8_buffer_size, average_relative_error_u8_to_device_with_scratch, average_relative_error_u8_to_device),
        i8 => (average_relative_error_i8_buffer_size, average_relative_error_i8_to_device_with_scratch, average_relative_error_i8_to_device),
        u16 => (average_relative_error_u16_buffer_size, average_relative_error_u16_to_device_with_scratch, average_relative_error_u16_to_device),
        i16 => (average_relative_error_i16_buffer_size, average_relative_error_i16_to_device_with_scratch, average_relative_error_i16_to_device),
        ComplexI16 => (average_relative_error_i16_complex_buffer_size, average_relative_error_i16_complex_to_device_with_scratch, average_relative_error_i16_complex_to_device),
        u32 => (average_relative_error_u32_buffer_size, average_relative_error_u32_to_device_with_scratch, average_relative_error_u32_to_device),
        i32 => (average_relative_error_i32_buffer_size, average_relative_error_i32_to_device_with_scratch, average_relative_error_i32_to_device),
        ComplexI32 => (average_relative_error_i32_complex_buffer_size, average_relative_error_i32_complex_to_device_with_scratch, average_relative_error_i32_complex_to_device),
        i64 => (average_relative_error_i64_buffer_size, average_relative_error_i64_to_device_with_scratch, average_relative_error_i64_to_device),
        ComplexI64 => (average_relative_error_i64_complex_buffer_size, average_relative_error_i64_complex_to_device_with_scratch, average_relative_error_i64_complex_to_device),
        f32 => (average_relative_error_f32_buffer_size, average_relative_error_f32_to_device_with_scratch, average_relative_error_f32_to_device),
        Complex32 => (average_relative_error_f32_complex_buffer_size, average_relative_error_f32_complex_to_device_with_scratch, average_relative_error_f32_complex_to_device),
        f64 => (average_relative_error_f64_buffer_size, average_relative_error_f64_to_device_with_scratch, average_relative_error_f64_to_device),
        Complex64 => (average_relative_error_f64_complex_buffer_size, average_relative_error_f64_complex_to_device_with_scratch, average_relative_error_f64_complex_to_device)
    ]
);
