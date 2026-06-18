macro_rules! impl_transpose {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_transpose_size(source.size(), destination.size())?;

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

macro_rules! impl_generic_transpose_operation {
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

macro_rules! impl_copy_constant_border_scalar {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            top: usize,
            left: usize,
            value: $ty,
        ) -> Result<()> {
            let (top, left) =
                validate_border_placement(source.size(), destination.size(), top, left)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    top,
                    left,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_constant_border_packed {
    ($name:ident, $ty:ty, $layout:ty, $value_ty:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            top: usize,
            left: usize,
            value: $value_ty,
        ) -> Result<()> {
            let (top, left) =
                validate_border_placement(source.size(), destination.size(), top, left)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    top,
                    left,
                    value.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_border {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            top: usize,
            left: usize,
        ) -> Result<()> {
            let (top, left) =
                validate_border_placement(source.size(), destination.size(), top, left)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    top,
                    left,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_constant_border_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => ($value_ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            type Value;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                top: usize,
                left: usize,
                value: Self::Value,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                type Value = $value_ty;

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    top: usize,
                    left: usize,
                    value: Self::Value,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, top, left, value)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            top: usize,
            left: usize,
            value: T::Value,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, top, left, value)
        }
    };
}

macro_rules! impl_generic_copy_border_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                top: usize,
                left: usize,
            ) -> Result<()>;
        }

        $(
            impl $trait<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    top: usize,
                    left: usize,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, top, left)
                }
            }
        )+

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            top: usize,
            left: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::dispatch(stream_context, source, destination, top, left)
        }
    };
}
