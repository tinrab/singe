macro_rules! impl_filter_integer {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_float_typed {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[$pixel_ty],
            kernel_size: Size,
            anchor: Point,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_integer_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    divisor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_float_typed_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[$pixel_ty],
            kernel_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_filter_integer {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                kernel: &[i32],
                kernel_size: Size,
                anchor: Point,
                divisor: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(
                stream_context,
                source,
                destination,
                kernel,
                kernel_size,
                anchor,
                divisor,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    kernel: &[i32],
                    kernel_size: Size,
                    anchor: Point,
                    divisor: i32,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        divisor,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_filter_typed {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                kernel: &[Self],
                kernel_size: Size,
                anchor: Point,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            kernel: &[T],
            kernel_size: Size,
            anchor: Point,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, kernel, kernel_size, anchor)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    kernel: &[Self],
                    kernel_size: Size,
                    anchor: Point,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, kernel, kernel_size, anchor)
                }
            }
        )*
    };
}

macro_rules! impl_generic_filter_integer_border {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                kernel: &[i32],
                kernel_size: Size,
                anchor: Point,
                divisor: i32,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
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
                kernel,
                kernel_size,
                anchor,
                divisor,
                border_type,
            )
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    kernel: &[i32],
                    kernel_size: Size,
                    anchor: Point,
                    divisor: i32,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        divisor,
                        border_type,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_generic_filter_typed_border {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                kernel: &[Self],
                kernel_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, T, $layout>,
            kernel: &[T],
            kernel_size: Size,
            anchor: Point,
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
                kernel,
                kernel_size,
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
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    kernel: &[Self],
                    kernel_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        border_type,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_filter32f {
    ($name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            kernel: &[f32],
            kernel_size: Size,
            anchor: Point,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter32f_border {
    ($name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            kernel: &[f32],
            kernel_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_kernel_f32 {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[f32],
            anchor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_f32(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            anchor: i32,
            divisor: i32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    divisor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_border_kernel_f32 {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[f32],
            anchor: i32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_f32(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_border_float {
    ($name:ident, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            kernel: &[f32],
            anchor: i32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_f32(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
