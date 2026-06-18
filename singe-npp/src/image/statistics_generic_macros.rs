macro_rules! impl_generic_statistic {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, output)
        }
    };
}

macro_rules! impl_generic_typed_statistic {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                output: &mut DeviceMemory<Self::Output>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceMemory<Self::Output>,
                ) -> Result<()> {
                    $direct(stream_context, source, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            output: &mut DeviceMemory<T::Output>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, output)
        }
    };
}

macro_rules! impl_generic_typed_pair_statistic {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                minimum: &mut DeviceMemory<Self::Output>,
                maximum: &mut DeviceMemory<Self::Output>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    minimum: &mut DeviceMemory<Self::Output>,
                    maximum: &mut DeviceMemory<Self::Output>,
                ) -> Result<()> {
                    $direct(stream_context, source, minimum, maximum)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            minimum: &mut DeviceMemory<T::Output>,
            maximum: &mut DeviceMemory<T::Output>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, minimum, maximum)
        }
    };
}

macro_rules! impl_generic_typed_indexed_statistic {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                output: &mut DeviceMemory<Self::Output>,
                index_x: &mut DeviceMemory<i32>,
                index_y: &mut DeviceMemory<i32>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceMemory<Self::Output>,
                    index_x: &mut DeviceMemory<i32>,
                    index_y: &mut DeviceMemory<i32>,
                ) -> Result<()> {
                    $direct(stream_context, source, output, index_x, index_y)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            output: &mut DeviceMemory<T::Output>,
            index_x: &mut DeviceMemory<i32>,
            index_y: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, output, index_x, index_y)
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum: &mut DeviceMemory<Self::Output>,
                maximum: &mut DeviceMemory<Self::Output>,
                minimum_index: &mut DeviceMemory<Point>,
                maximum_index: &mut DeviceMemory<Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    minimum: &mut DeviceMemory<Self::Output>,
                    maximum: &mut DeviceMemory<Self::Output>,
                    minimum_index: &mut DeviceMemory<Point>,
                    maximum_index: &mut DeviceMemory<Point>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        minimum,
                        maximum,
                        minimum_index,
                        maximum_index,
                    )
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            minimum: &mut DeviceMemory<T::Output>,
            maximum: &mut DeviceMemory<T::Output>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            T::dispatch(
                stream_context,
                source,
                minimum,
                maximum,
                minimum_index,
                maximum_index,
            )
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_c3 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                channel: usize,
                minimum: &mut DeviceMemory<Self::Output>,
                maximum: &mut DeviceMemory<Self::Output>,
                minimum_index: &mut DeviceMemory<Point>,
                maximum_index: &mut DeviceMemory<Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    channel: usize,
                    minimum: &mut DeviceMemory<Self::Output>,
                    maximum: &mut DeviceMemory<Self::Output>,
                    minimum_index: &mut DeviceMemory<Point>,
                    maximum_index: &mut DeviceMemory<Point>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        channel,
                        minimum,
                        maximum,
                        minimum_index,
                        maximum_index,
                    )
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            channel: usize,
            minimum: &mut DeviceMemory<T::Output>,
            maximum: &mut DeviceMemory<T::Output>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            T::dispatch(
                stream_context,
                source,
                channel,
                minimum,
                maximum,
                minimum_index,
                maximum_index,
            )
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_masked_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                minimum: &mut DeviceMemory<Self::Output>,
                maximum: &mut DeviceMemory<Self::Output>,
                minimum_index: &mut DeviceMemory<Point>,
                maximum_index: &mut DeviceMemory<Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    minimum: &mut DeviceMemory<Self::Output>,
                    maximum: &mut DeviceMemory<Self::Output>,
                    minimum_index: &mut DeviceMemory<Point>,
                    maximum_index: &mut DeviceMemory<Point>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        mask,
                        minimum,
                        maximum,
                        minimum_index,
                        maximum_index,
                    )
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            minimum: &mut DeviceMemory<T::Output>,
            maximum: &mut DeviceMemory<T::Output>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            T::dispatch(
                stream_context,
                source,
                mask,
                minimum,
                maximum,
                minimum_index,
                maximum_index,
            )
        }
    };
}

macro_rules! impl_generic_typed_pair_indexed_statistic_masked_c3 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $output_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            type Output: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                mask: &ImageView<'_, u8, C1>,
                channel: usize,
                minimum: &mut DeviceMemory<Self::Output>,
                maximum: &mut DeviceMemory<Self::Output>,
                minimum_index: &mut DeviceMemory<Point>,
                maximum_index: &mut DeviceMemory<Point>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                type Output = $output_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    mask: &ImageView<'_, u8, C1>,
                    channel: usize,
                    minimum: &mut DeviceMemory<Self::Output>,
                    maximum: &mut DeviceMemory<Self::Output>,
                    minimum_index: &mut DeviceMemory<Point>,
                    maximum_index: &mut DeviceMemory<Point>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        mask,
                        channel,
                        minimum,
                        maximum,
                        minimum_index,
                        maximum_index,
                    )
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            minimum: &mut DeviceMemory<T::Output>,
            maximum: &mut DeviceMemory<T::Output>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            T::dispatch(
                stream_context,
                source,
                mask,
                channel,
                minimum,
                maximum,
                minimum_index,
                maximum_index,
            )
        }
    };
}

macro_rules! impl_generic_statistic_masked_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    output: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source, mask, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, mask, output)
        }
    };
}

macro_rules! impl_generic_statistic_masked_c3 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                mask: &ImageView<'_, u8, C1>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    mask: &ImageView<'_, u8, C1>,
                    channel: usize,
                    output: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source, mask, channel, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, mask, channel, output)
        }
    };
}

macro_rules! impl_generic_mean_stddev_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mean: &mut DeviceMemory<f64>,
                standard_deviation: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mean: &mut DeviceMemory<f64>,
                    standard_deviation: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source, mean, standard_deviation)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mean: &mut DeviceMemory<f64>,
            standard_deviation: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, mean, standard_deviation)
        }
    };
}

macro_rules! impl_generic_mean_stddev_masked_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                mean: &mut DeviceMemory<f64>,
                standard_deviation: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    mean: &mut DeviceMemory<f64>,
                    standard_deviation: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source, mask, mean, standard_deviation)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            mean: &mut DeviceMemory<f64>,
            standard_deviation: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, mask, mean, standard_deviation)
        }
    };
}

macro_rules! impl_generic_dot_prod {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, Layout>,
                source_2: &ImageView<'_, Self, Layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, $layout>,
                    source_2: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, $layout>,
            source_2: &ImageView<'_, T, $layout>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source_1, source_2, output)
        }
    };
}

macro_rules! impl_generic_quality_index {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, Layout>,
                source_2: &ImageView<'_, Self, Layout>,
                output: &mut DeviceMemory<f32>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, $layout>,
                    source_2: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceMemory<f32>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, $layout>,
            source_2: &ImageView<'_, T, $layout>,
            output: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            T::dispatch(stream_context, source_1, source_2, output)
        }
    };
}

macro_rules! impl_generic_error_metric_masked_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, C1>,
                source_2: &ImageView<'_, Self, C1>,
                mask: &ImageView<'_, u8, C1>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, C1>,
                    source_2: &ImageView<'_, Self, C1>,
                    mask: &ImageView<'_, u8, C1>,
                    output: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, mask, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, C1>,
            source_2: &ImageView<'_, T, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source_1, source_2, mask, output)
        }
    };
}

macro_rules! impl_generic_error_metric_masked_c3 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, C3>,
                source_2: &ImageView<'_, Self, C3>,
                mask: &ImageView<'_, u8, C1>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, C3>,
                    source_2: &ImageView<'_, Self, C3>,
                    mask: &ImageView<'_, u8, C1>,
                    channel: usize,
                    output: &mut DeviceMemory<f64>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, mask, channel, output)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, C3>,
            source_2: &ImageView<'_, T, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            T::dispatch(stream_context, source_1, source_2, mask, channel, output)
        }
    };
}

macro_rules! impl_generic_every_statistic_in_place {
    ($trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )+

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, destination)
        }
    };
}
