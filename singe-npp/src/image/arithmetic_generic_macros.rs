macro_rules! impl_generic_binary_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source1, source2, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source1, source2, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_binary_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, source_destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, source_destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_binary_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source1, source2, destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source1, source2, destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_binary_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, source_destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, source_destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_round_scaled_binary_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source1,
                source2,
                destination,
                scale_factor,
                round_mode,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source1,
                        source2,
                        destination,
                        scale_factor,
                        round_mode,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_round_scaled_binary_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source,
                source_destination,
                scale_factor,
                round_mode,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_destination,
                        scale_factor,
                        round_mode,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_unary_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_unary_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source_destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source_destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_unary_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_unary_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source_destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source_destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_alpha_comp_constant {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                alpha1: Self,
                source2: &ImageView<'_, Self, Layout>,
                alpha2: Self,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                operation: AlphaOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            alpha1: T,
            source2: &ImageView<'_, T, $layout>,
            alpha2: T,
            destination: &mut ImageViewMut<'_, T, $layout>,
            operation: AlphaOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source1,
                alpha1,
                source2,
                alpha2,
                destination,
                operation,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    alpha1: Self,
                    source2: &ImageView<'_, Self, $layout>,
                    alpha2: Self,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    operation: AlphaOperation,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source1,
                        alpha1,
                        source2,
                        alpha2,
                        destination,
                        operation,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_alpha_comp {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                operation: AlphaOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            operation: AlphaOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source1, source2, destination, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    operation: AlphaOperation,
                ) -> Result<()> {
                    $direct(stream_context, source1, source2, destination, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_constant_scalar_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Constant: Copy;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T::Constant,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constant, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self::Constant,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, constant, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_constant_scalar_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Constant: Copy;

            fn $method(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            constant: T::Constant,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, constant, source_destination)
        }

        $(
            impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                fn $method(
                    stream_context: &StreamContext,
                    constant: Self::Constant,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, constant, source_destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_constant_array_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Constant: Copy;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self::Constant; $channels],
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T::Constant; $channels],
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constants, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self::Constant; $channels],
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, constants, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_constant_array_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Constant: Copy;

            fn $method(
                stream_context: &StreamContext,
                constants: [Self::Constant; $channels],
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            constants: [T::Constant; $channels],
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, constants, source_destination)
        }

        $(
            impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                fn $method(
                    stream_context: &StreamContext,
                    constants: [Self::Constant; $channels],
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, constants, source_destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_scalar_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constant, destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, constant, destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_scalar_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                constant: Self,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            constant: T,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, constant, source_destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    constant: Self,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, constant, source_destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_array_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self; $channels],
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T; $channels],
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constants, destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self; $channels],
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, constants, destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_constant_array_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        $channels:literal,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                constants: [Self; $channels],
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            constants: [T; $channels],
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, constants, source_destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    constants: [Self; $channels],
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, constants, source_destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_device_constant_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Constant: Copy + 'static;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: &DeviceMemory<T::Constant>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constants, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: &DeviceMemory<Self::Constant>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, constants, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_device_constant_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty, $constant_ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Constant: Copy + 'static;

            fn $method(
                stream_context: &StreamContext,
                constants: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            constants: &DeviceMemory<T::Constant>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, constants, source_destination)
        }

        $(
            impl $trait<$layout> for $ty {
                type Constant = $constant_ty;

                fn $method(
                    stream_context: &StreamContext,
                    constants: &DeviceMemory<Self::Constant>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, constants, source_destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_device_constant_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: &DeviceMemory<Self>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: &DeviceMemory<T>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constants, destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: &DeviceMemory<Self>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, constants, destination, scale_factor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_scaled_device_constant_operation_in_place {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                constants: &DeviceMemory<Self>,
                source_destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            constants: &DeviceMemory<T>,
            source_destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, constants, source_destination, scale_factor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    constants: &DeviceMemory<Self>,
                    source_destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, constants, source_destination, scale_factor)
                }
            }
        )*
    };
}
