macro_rules! impl_lookup_table_c1 {
    ($name:ident, $ty:ty, $value_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            values: &DeviceMemory<$value_ty>,
            levels: &DeviceMemory<$value_ty>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let level_count = validate_lookup_table(values.len(), levels.len())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    values.as_ptr().cast(),
                    levels.as_ptr().cast(),
                    level_count,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_c1_in_place {
    ($name:ident, $ty:ty, $value_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            values: &DeviceMemory<$value_ty>,
            levels: &DeviceMemory<$value_ty>,
        ) -> Result<()> {
            let level_count = validate_lookup_table(values.len(), levels.len())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    values.as_ptr().cast(),
                    levels.as_ptr().cast(),
                    level_count,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_lookup_table_c1 {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            type Value: Copy;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                values: &DeviceMemory<Self::Value>,
                levels: &DeviceMemory<Self::Value>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            values: &DeviceMemory<T::Value>,
            levels: &DeviceMemory<T::Value>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, values, levels)
        }

        $(
            impl $trait for $ty {
                type Value = $value_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    values: &DeviceMemory<Self::Value>,
                    levels: &DeviceMemory<Self::Value>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, values, levels)
                }
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_c1_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            type Value: Copy;

            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                values: &DeviceMemory<Self::Value>,
                levels: &DeviceMemory<Self::Value>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            values: &DeviceMemory<T::Value>,
            levels: &DeviceMemory<T::Value>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, image, values, levels)
        }

        $(
            impl $trait for $ty {
                type Value = $value_ty;

                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    values: &DeviceMemory<Self::Value>,
                    levels: &DeviceMemory<Self::Value>,
                ) -> Result<()> {
                    $direct(stream_context, image, values, levels)
                }
            }
        )*
    };
}

macro_rules! impl_lookup_table_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            values: &[&DeviceMemory<i32>; $channels],
            levels: &[&DeviceMemory<i32>; $channels],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors(values, levels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_packed_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            values: &[&DeviceMemory<i32>; $channels],
            levels: &[&DeviceMemory<i32>; $channels],
        ) -> Result<()> {
            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors(values, levels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_packed_f32 {
    ($name:ident, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            values: &[&DeviceMemory<f32>; $channels],
            levels: &[&DeviceMemory<f32>; $channels],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors_f32(values, levels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_packed_f32_in_place {
    ($name:ident, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, f32, $layout>,
            values: &[&DeviceMemory<f32>; $channels],
            levels: &[&DeviceMemory<f32>; $channels],
        ) -> Result<()> {
            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors_f32(values, levels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_lookup_table_packed {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Value: Copy;

            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                values: &[&DeviceMemory<Self::Value>; $channels],
                levels: &[&DeviceMemory<Self::Value>; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            values: &[&DeviceMemory<T::Value>; $channels],
            levels: &[&DeviceMemory<T::Value>; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, values, levels)
        }

        $(
            impl $trait<$layout> for $ty {
                type Value = $value_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    values: &[&DeviceMemory<Self::Value>; $channels],
                    levels: &[&DeviceMemory<Self::Value>; $channels],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, values, levels)
                }
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_packed_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Value: Copy;

            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                values: &[&DeviceMemory<Self::Value>; $channels],
                levels: &[&DeviceMemory<Self::Value>; $channels],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            values: &[&DeviceMemory<T::Value>; $channels],
            levels: &[&DeviceMemory<T::Value>; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, values, levels)
        }

        $(
            impl $trait<$layout> for $ty {
                type Value = $value_ty;

                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    values: &[&DeviceMemory<Self::Value>; $channels],
                    levels: &[&DeviceMemory<Self::Value>; $channels],
                ) -> Result<()> {
                    $direct(stream_context, image, values, levels)
                }
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_trilinear {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                values: &DeviceMemory<u32>,
                levels: [&[Self]; 3],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            values: &DeviceMemory<u32>,
            levels: [&[T]; 3],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, values, levels)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    values: &DeviceMemory<u32>,
                    levels: [&[Self]; 3],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, values, levels)
                }
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_trilinear_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                values: &DeviceMemory<u32>,
                levels: [&[Self]; 3],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            values: &DeviceMemory<u32>,
            levels: [&[T]; 3],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, values, levels)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    values: &DeviceMemory<u32>,
                    levels: [&[Self]; 3],
                ) -> Result<()> {
                    $direct(stream_context, image, values, levels)
                }
            }
        )*
    };
}
