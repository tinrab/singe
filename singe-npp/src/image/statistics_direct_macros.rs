macro_rules! impl_statistic {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_dot_prod {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_same_size(source_1.size(), source_2.size())?;
            validate_statistic_output(output, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_every_statistic_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), source_destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_statistic_masked {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    coi,
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_typed_statistic {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_type:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<$output_type>,
        ) -> Result<()> {
            validate_typed_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_typed_pair_statistic {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_type:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            minimum: &mut DeviceMemory<$output_type>,
            maximum: &mut DeviceMemory<$output_type>,
        ) -> Result<()> {
            validate_typed_statistic_output(minimum, $output_channels)?;
            validate_typed_statistic_output(maximum, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    minimum.as_mut_ptr().cast(),
                    maximum.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_typed_indexed_statistic {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_type:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<$output_type>,
            index_x: &mut DeviceMemory<i32>,
            index_y: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            validate_typed_statistic_output(output, $output_channels)?;
            validate_index_output(index_x, $output_channels)?;
            validate_index_output(index_y, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    index_x.as_mut_ptr().cast(),
                    index_y.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_typed_pair_indexed_statistic {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            minimum: &mut DeviceMemory<$output_type>,
            maximum: &mut DeviceMemory<$output_type>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    minimum.as_mut_ptr().cast(),
                    maximum.as_mut_ptr().cast(),
                    minimum_index.as_mut_ptr().cast(),
                    maximum_index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            channel: usize,
            minimum: &mut DeviceMemory<$output_type>,
            maximum: &mut DeviceMemory<$output_type>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    coi,
                    minimum.as_mut_ptr().cast(),
                    maximum.as_mut_ptr().cast(),
                    minimum_index.as_mut_ptr().cast(),
                    maximum_index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_typed_pair_indexed_statistic_masked {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            minimum: &mut DeviceMemory<$output_type>,
            maximum: &mut DeviceMemory<$output_type>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            validate_mask_size(source.size(), mask.size())?;
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    minimum.as_mut_ptr().cast(),
                    maximum.as_mut_ptr().cast(),
                    minimum_index.as_mut_ptr().cast(),
                    maximum_index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            minimum: &mut DeviceMemory<$output_type>,
            maximum: &mut DeviceMemory<$output_type>,
            minimum_index: &mut DeviceMemory<Point>,
            maximum_index: &mut DeviceMemory<Point>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    coi,
                    minimum.as_mut_ptr().cast(),
                    maximum.as_mut_ptr().cast(),
                    minimum_index.as_mut_ptr().cast(),
                    maximum_index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_norm_statistic {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_norm_statistic_masked {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    coi,
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_error_metric {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_same_size(source_1.size(), source_2.size())?;
            validate_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_quality_index {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            validate_same_size(source_1.size(), source_2.size())?;
            validate_metric_output(output, 1, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_error_metric_masked {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_same_size(source_1.size(), source_2.size())?;
            validate_mask_size(source_1.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, C3>,
            source_2: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_same_size(source_1.size(), source_2.size())?;
            validate_mask_size(source_1.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_1.size().into(),
                    coi,
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mean_stddev {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mean: &mut DeviceMemory<f64>,
            standard_deviation: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            channel: usize,
            mean: &mut DeviceMemory<f64>,
            standard_deviation: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    coi,
                    scratch.as_mut_ptr().cast(),
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mean_stddev_masked {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            mean: &mut DeviceMemory<f64>,
            standard_deviation: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            mean: &mut DeviceMemory<f64>,
            standard_deviation: &mut DeviceMemory<f64>,
        ) -> Result<()> {
            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    coi,
                    scratch.as_mut_ptr().cast(),
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
