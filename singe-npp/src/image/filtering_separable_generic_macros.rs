macro_rules! impl_generic_separable_integer_filter {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            kernel: &[i32],
            anchor: i32,
            divisor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, kernel, anchor, divisor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    kernel: &[i32],
                    anchor: i32,
                    divisor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, kernel, anchor, divisor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_separable_typed_filter {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                kernel: &[Self],
                anchor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            kernel: &[T],
            anchor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, kernel, anchor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    kernel: &[Self],
                    anchor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, kernel, anchor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_sum_window {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, f32, Layout>,
                mask_size: i32,
                anchor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, mask_size, anchor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, f32, $layout>,
                    mask_size: i32,
                    anchor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, mask_size, anchor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_sum_window_border {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, f32, Layout>,
                mask_size: i32,
                anchor: i32,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source,
                source_offset,
                destination,
                mask_size,
                anchor,
                border_type,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageViewMut<'_, f32, $layout>,
                    mask_size: i32,
                    anchor: i32,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        anchor,
                        border_type,
                    )
                }
            }
        )*
    };
}
