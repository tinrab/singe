macro_rules! impl_generic_convert_to_operation {
    ($trait:ident, $function:ident, $destination_ty:ty, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, Layout>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination)
        }
    };
}

macro_rules! impl_generic_convert_round_to_operation {
    ($trait:ident, $function:ident, $destination_ty:ty, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, Layout>,
                round_mode: RoundMode,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                    round_mode: RoundMode,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, round_mode)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            round_mode: RoundMode,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, round_mode)
        }
    };
}

macro_rules! impl_generic_convert_scaled_round_to_operation {
    ($trait:ident, $function:ident, $destination_ty:ty, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, Layout>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                    round_mode: RoundMode,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, round_mode, scale_factor)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            round_mode: RoundMode,
            scale_factor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, round_mode, scale_factor)
        }
    };
}

macro_rules! impl_scale_to_higher {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageViewMut<'_, $dst_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scale_to_higher_range {
    ($name:ident, $src_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_scale_range(min, max)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    min,
                    max,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scale_to_lower_hint {
    ($name:ident, $src_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageViewMut<'_, u8, $layout>,
            hint: HintAlgorithm,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    hint.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scale_to_lower_range {
    ($name:ident, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            destination: &mut ImageViewMut<'_, u8, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_scale_range(min, max)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    min,
                    max,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_scale_to_higher_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => ($dst_ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Destination: DataTypeLike;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self::Destination, Layout>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                type Destination = $dst_ty;

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T::Destination, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination)
        }
    };
}

macro_rules! impl_generic_scale_to_f32_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, f32, Layout>,
                min: f32,
                max: f32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, f32, $layout>,
                    min: f32,
                    max: f32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, min, max)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, min, max)
        }
    };
}

macro_rules! impl_generic_scale_to_u8_hint_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, u8, Layout>,
                hint: HintAlgorithm,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, u8, $layout>,
                    hint: HintAlgorithm,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, hint)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, u8, $layout>,
            hint: HintAlgorithm,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, hint)
        }
    };
}

macro_rules! impl_generic_scale_to_u8_range_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($src_ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, u8, Layout>,
                min: f32,
                max: f32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $src_ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, u8, $layout>,
                    min: f32,
                    max: f32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, min, max)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, u8, $layout>,
            min: f32,
            max: f32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, min, max)
        }
    };
}
