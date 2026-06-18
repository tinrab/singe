macro_rules! impl_swap_channels {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            order: [i32; $channels],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_order(&order, $channels as i32)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            order: [i32; $channels],
        ) -> Result<()> {
            validate_channel_order(&order, $channels as i32)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_c4_to_c3 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C4>,
            destination: &mut ImageViewMut<'_, $ty, C3>,
            order: [i32; 3],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_order(&order, 4)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_c3_to_c4 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            destination: &mut ImageViewMut<'_, $ty, C4>,
            order: [i32; 4],
            value: $ty,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_order_with_fill(&order, 3, 4)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_ac4 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, AC4>,
            destination: &mut ImageViewMut<'_, $ty, AC4>,
            order: [i32; 3],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_channel_order(&order, 3)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_swap_channels_operation {
    ($trait:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                order: [i32; $channels],
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    order: [i32; $channels],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, order)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            order: [i32; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, order)
        }
    };
}

macro_rules! impl_generic_swap_channels_in_place_operation {
    ($trait:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                order: [i32; $channels],
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    order: [i32; $channels],
                ) -> Result<()> {
                    $direct(stream_context, image, order)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            order: [i32; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, image, order)
        }
    };
}

macro_rules! impl_generic_swap_channels_c4_to_c3_operation {
    ($trait:ident, $function:ident, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C4>,
                destination: &mut ImageViewMut<'_, Self, C3>,
                order: [i32; 3],
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C4>,
                    destination: &mut ImageViewMut<'_, Self, C3>,
                    order: [i32; 3],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, order)
                }
            }
        )+

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C4>,
            destination: &mut ImageViewMut<'_, T, C3>,
            order: [i32; 3],
        ) -> Result<()> {
            T::dispatch(stream_context, source, destination, order)
        }
    };
}

macro_rules! impl_generic_swap_channels_c3_to_c4_operation {
    ($trait:ident, $function:ident, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut ImageViewMut<'_, Self, C4>,
                order: [i32; 4],
                value: Self,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    destination: &mut ImageViewMut<'_, Self, C4>,
                    order: [i32; 4],
                    value: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, order, value)
                }
            }
        )+

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            destination: &mut ImageViewMut<'_, T, C4>,
            order: [i32; 4],
            value: T,
        ) -> Result<()> {
            T::dispatch(stream_context, source, destination, order, value)
        }
    };
}

macro_rules! impl_set_masked {
    ($name:ident, $ty:ty, $value_ty:ty, $layout:ty, $ffi:ident, $value_expr:expr) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            value: $value_ty,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()> {
            validate_mask_size(destination.size(), mask.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    $value_expr(&value),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_set_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => ($value_ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Value;

            fn dispatch(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                type Value = $value_ty;

                fn dispatch(
                    stream_context: &StreamContext,
                    value: Self::Value,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, value, destination)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            value: T::Value,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, value, destination)
        }
    };
}

macro_rules! impl_generic_set_masked_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => ($value_ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Value;

            fn dispatch(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                mask: &MaskView<'_>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                type Value = $value_ty;

                fn dispatch(
                    stream_context: &StreamContext,
                    value: Self::Value,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    mask: &MaskView<'_>,
                ) -> Result<()> {
                    $direct(stream_context, value, destination, mask)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            value: T::Value,
            destination: &mut ImageViewMut<'_, T, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, value, destination, mask)
        }
    };
}
