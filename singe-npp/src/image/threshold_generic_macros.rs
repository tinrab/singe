macro_rules! impl_generic_threshold_scalar {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, threshold, operation)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, threshold, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_scalar_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, image, threshold, operation)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, image, threshold, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, thresholds, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, thresholds, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_array_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, thresholds, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, image, thresholds, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_scalar {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, threshold)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, threshold)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_scalar_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, image, threshold)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                ) -> Result<()> {
                    $direct(stream_context, image, threshold)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, thresholds)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, thresholds)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_array_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, thresholds)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                ) -> Result<()> {
                    $direct(stream_context, image, thresholds)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_scalar {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                value: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            value: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, threshold, value, operation)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, threshold, value, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_scalar_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                value: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            value: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, image, threshold, value, operation)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, image, threshold, value, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, thresholds, values, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, thresholds, values, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_array_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, thresholds, values, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, image, thresholds, values, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_scalar {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                value: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, threshold, value)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, threshold, value)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_scalar_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                value: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, image, threshold, value)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                ) -> Result<()> {
                    $direct(stream_context, image, threshold, value)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, thresholds, values)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, thresholds, values)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_array_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, thresholds, values)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                ) -> Result<()> {
                    $direct(stream_context, image, thresholds, values)
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_scalar {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                lower_threshold: Self,
                lower_value: Self,
                upper_threshold: Self,
                upper_value: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            lower_threshold: T,
            lower_value: T,
            upper_threshold: T,
            upper_value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                source,
                destination,
                lower_threshold,
                lower_value,
                upper_threshold,
                upper_value,
            )
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    lower_threshold: Self,
                    lower_value: Self,
                    upper_threshold: Self,
                    upper_value: Self,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        destination,
                        lower_threshold,
                        lower_value,
                        upper_threshold,
                        upper_value,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_scalar_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                lower_threshold: Self,
                lower_value: Self,
                upper_threshold: Self,
                upper_value: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            lower_threshold: T,
            lower_value: T,
            upper_threshold: T,
            upper_value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                image,
                lower_threshold,
                lower_value,
                upper_threshold,
                upper_value,
            )
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    lower_threshold: Self,
                    lower_value: Self,
                    upper_threshold: Self,
                    upper_value: Self,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        image,
                        lower_threshold,
                        lower_value,
                        upper_threshold,
                        upper_value,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                lower_thresholds: [Self; $channels],
                lower_values: [Self; $channels],
                upper_thresholds: [Self; $channels],
                upper_values: [Self; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            lower_thresholds: [T; $channels],
            lower_values: [T; $channels],
            upper_thresholds: [T; $channels],
            upper_values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source,
                destination,
                lower_thresholds,
                lower_values,
                upper_thresholds,
                upper_values,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    lower_thresholds: [Self; $channels],
                    lower_values: [Self; $channels],
                    upper_thresholds: [Self; $channels],
                    upper_values: [Self; $channels],
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        destination,
                        lower_thresholds,
                        lower_values,
                        upper_thresholds,
                        upper_values,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_array_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                lower_thresholds: [Self; $channels],
                lower_values: [Self; $channels],
                upper_thresholds: [Self; $channels],
                upper_values: [Self; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            lower_thresholds: [T; $channels],
            lower_values: [T; $channels],
            upper_thresholds: [T; $channels],
            upper_values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                image,
                lower_thresholds,
                lower_values,
                upper_thresholds,
                upper_values,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    lower_thresholds: [Self; $channels],
                    lower_values: [Self; $channels],
                    upper_thresholds: [Self; $channels],
                    upper_values: [Self; $channels],
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        image,
                        lower_thresholds,
                        lower_values,
                        upper_thresholds,
                        upper_values,
                    )
                }
            }
        )*
    };
}
