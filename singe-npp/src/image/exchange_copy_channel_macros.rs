macro_rules! impl_copy {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_operation {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
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

macro_rules! impl_copy_masked {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_mask_size(source.size(), mask.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_masked_operation {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                mask: &MaskView<'_>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, mask)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    mask: &MaskView<'_>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, mask)
                }
            }
        )*
    };
}

macro_rules! impl_set {
    ($name:ident, $ty:ty, $value_ty:ty, $layout:ty, $ffi:ident, $value_expr:expr) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            value: $value_ty,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    $value_expr(&value),
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

macro_rules! impl_copy_channel {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_channel: usize,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            destination_channel: usize,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_index(source_channel, $channels)?;
            validate_channel_index(destination_channel, $channels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().add(source_channel),
                    source.step(),
                    destination.as_mut_ptr().add(destination_channel),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_channel: usize,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                destination_channel: usize,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_channel: usize,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    destination_channel: usize,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_channel,
                        destination,
                        destination_channel,
                    )
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_channel: usize,
            destination: &mut ImageViewMut<'_, T, $layout>,
            destination_channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(
                stream_context,
                source,
                source_channel,
                destination,
                destination_channel,
            )
        }
    };
}

macro_rules! impl_extract_channel {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            channel: usize,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_index(channel, $channels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().add(channel),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_insert_channel {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            channel: usize,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_index(channel, $channels)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().add(channel),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_extract_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                channel: usize,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    channel: usize,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, channel)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, C1>,
            channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, channel)
        }
    };
}

macro_rules! impl_generic_insert_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                channel: usize,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    channel: usize,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, channel)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, channel)
        }
    };
}

macro_rules! impl_generic_set_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                value: Self,
                channel: usize,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    value: Self,
                    channel: usize,
                ) -> Result<()> {
                    $direct(stream_context, destination, value, channel)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            destination: &mut ImageViewMut<'_, T, $layout>,
            value: T,
            channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, destination, value, channel)
        }
    };
}

macro_rules! impl_copy_packed_to_planar {
    ($name:ident, $ty:ty, $layout:ty, $planes:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.planes()[0].size())?;

            let destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_ptr().cast(),
                    destination.planes()[0].step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_planar_to_packed {
    ($name:ident, $ty:ty, $layout:ty, $planes:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast(),
                    source.planes()[0].step(),
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

macro_rules! impl_generic_copy_packed_to_planar_operation {
    ($trait:ident, $function:ident, $layout:ty, $planes:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut PlanarImageViewMut<'_, Self, $planes>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut PlanarImageViewMut<'_, T, $planes>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination)
        }
    };
}

macro_rules! impl_generic_copy_planar_to_packed_operation {
    ($trait:ident, $function:ident, $layout:ty, $planes:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &PlanarImageView<'_, Self, $planes>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $planes>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination)
        }
    };
}
