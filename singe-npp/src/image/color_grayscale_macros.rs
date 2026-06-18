macro_rules! impl_rgb_to_gray {
    ($name:ident, $ty:ty, $source_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
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

macro_rules! impl_color_to_gray {
    ($name:ident, $ty:ty, $source_layout:ty, $coefficients:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            coefficients: &[f32; $coefficients],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    coefficients.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_rgb_to_gray {
    ($trait:ident, $method:ident, $function:ident, $source_layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<SourceLayout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, SourceLayout>,
                destination: &mut ImageViewMut<'_, Self, C1>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $source_layout>,
            destination: &mut ImageViewMut<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait<$source_layout>,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait<$source_layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $source_layout>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_color_to_gray {
    ($trait:ident, $method:ident, $function:ident, $source_layout:ty, $coefficients:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<SourceLayout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, SourceLayout>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                coefficients: &[f32; $coefficients],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $source_layout>,
            destination: &mut ImageViewMut<'_, T, C1>,
            coefficients: &[f32; $coefficients],
        ) -> Result<()>
        where
            T: $trait<$source_layout>,
        {
            T::$method(stream_context, source, destination, coefficients)
        }

        $(
            impl $trait<$source_layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $source_layout>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    coefficients: &[f32; $coefficients],
                ) -> Result<()> {
                    $direct(stream_context, source, destination, coefficients)
                }
            }
        )*
    };
}

macro_rules! impl_gradient_color_to_gray {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            norm: ImageNormalization,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    norm.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_cfa_to_rgb {
    ($name:ident, $ty:ty, $destination_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            source_roi: Rectangle,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
            grid: BayerGridPosition,
        ) -> Result<()> {
            validate_cfa_roi(source.size(), source_roi, destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    grid.into(),
                    InterpolationMode::Undefined.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_cfa_to_rgba {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            source_roi: Rectangle,
            destination: &mut ImageViewMut<'_, $ty, AC4>,
            grid: BayerGridPosition,
            alpha: $ty,
        ) -> Result<()> {
            validate_cfa_roi(source.size(), source_roi, destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    grid.into(),
                    InterpolationMode::Undefined.into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
