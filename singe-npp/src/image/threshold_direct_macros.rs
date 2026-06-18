macro_rules! impl_threshold_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    threshold,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_c1_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    threshold,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    thresholds.as_ptr().cast(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_packed_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    thresholds.as_ptr().cast(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    threshold,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    threshold,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    thresholds.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_packed_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    thresholds.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    threshold,
                    value,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_c1_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    threshold,
                    value,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    thresholds.as_ptr().cast(),
                    values.as_ptr().cast(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_packed_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    thresholds.as_ptr().cast(),
                    values.as_ptr().cast(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_value_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    threshold,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_value_c1_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    threshold,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_value_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    thresholds.as_ptr().cast(),
                    values.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_value_packed_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    thresholds.as_ptr().cast(),
                    values.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_less_greater_value_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            lower_threshold: $ty,
            lower_value: $ty,
            upper_threshold: $ty,
            upper_value: $ty,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_less_greater_value_c1_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            lower_threshold: $ty,
            lower_value: $ty,
            upper_threshold: $ty,
            upper_value: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_less_greater_value_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            lower_thresholds: [$ty; $channels],
            lower_values: [$ty; $channels],
            upper_thresholds: [$ty; $channels],
            upper_values: [$ty; $channels],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    lower_thresholds.as_ptr().cast(),
                    lower_values.as_ptr().cast(),
                    upper_thresholds.as_ptr().cast(),
                    upper_values.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_less_greater_value_packed_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            lower_thresholds: [$ty; $channels],
            lower_values: [$ty; $channels],
            upper_thresholds: [$ty; $channels],
            upper_values: [$ty; $channels],
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    lower_thresholds.as_ptr().cast(),
                    lower_values.as_ptr().cast(),
                    upper_thresholds.as_ptr().cast(),
                    upper_values.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
