use super::*;

macro_rules! impl_histogram_range {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $level_ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: &[$level_ty],
        ) -> Result<usize> {
            let raw_levels = validate_histogram_range_levels(levels)?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    raw_levels,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            histogram: &mut DeviceMemory<i32>,
            levels: &[$level_ty],
        ) -> Result<()> {
            let raw_levels = validate_histogram_range_output(histogram, levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram.as_mut_ptr().cast(),
                    levels.as_ptr().cast(),
                    raw_levels,
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
        $level_ty:ty,
        C3,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[$level_ty]; $channels],
        ) -> Result<usize> {
            let mut raw_levels = validate_histogram_range_level_arrays(&levels)?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    raw_levels.as_mut_ptr().cast(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mut histograms: [&mut DeviceMemory<i32>; $channels],
            levels: [&[$level_ty]; $channels],
        ) -> Result<()> {
            let mut raw_levels = validate_histogram_range_outputs(&histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut level_ptrs = levels.map(|level| level.as_ptr());

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    level_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
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
        $level_ty:ty,
        C4,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[$level_ty]; $channels],
        ) -> Result<usize> {
            let mut raw_levels = validate_histogram_range_level_arrays(&levels)?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    raw_levels.as_mut_ptr().cast(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C4>,
            mut histograms: [&mut DeviceMemory<i32>; $channels],
            levels: [&[$level_ty]; $channels],
        ) -> Result<()> {
            let mut raw_levels = validate_histogram_range_outputs(&histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut level_ptrs = levels.map(|level| level.as_ptr());

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    level_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
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
        $level_ty:ty,
        AC4,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[$level_ty]; $channels],
        ) -> Result<usize> {
            let mut raw_levels = validate_histogram_range_level_arrays(&levels)?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    raw_levels.as_mut_ptr().cast(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, AC4>,
            mut histograms: [&mut DeviceMemory<i32>; $channels],
            levels: [&[$level_ty]; $channels],
        ) -> Result<()> {
            let mut raw_levels = validate_histogram_range_outputs(&histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut level_ptrs = levels.map(|level| level.as_ptr());

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    level_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_histogram_range_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $level_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            type Level: DataTypeLike;

            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: &[Self::Level],
            ) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                histogram: &mut DeviceMemory<i32>,
                levels: &[Self::Level],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                type Level = $level_ty;

                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: &[Self::Level],
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    histogram: &mut DeviceMemory<i32>,
                    levels: &[Self::Level],
                ) -> Result<()> {
                    $direct(stream_context, source, histogram, levels)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: &[T::Level],
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            histogram: &mut DeviceMemory<i32>,
            levels: &[T::Level],
        ) -> Result<()> {
            T::dispatch(stream_context, source, histogram, levels)
        }
    };
}

macro_rules! impl_generic_histogram_range_packed {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, $channels:expr, [$(($ty:ty, $level_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            type Level: DataTypeLike;

            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: [&[Self::Level]; $channels],
            ) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                histograms: [&mut DeviceMemory<i32>; $channels],
                levels: [&[Self::Level]; $channels],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                type Level = $level_ty;

                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: [&[Self::Level]; $channels],
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    histograms: [&mut DeviceMemory<i32>; $channels],
                    levels: [&[Self::Level]; $channels],
                ) -> Result<()> {
                    $direct(stream_context, source, histograms, levels)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[T::Level]; $channels],
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            histograms: [&mut DeviceMemory<i32>; $channels],
            levels: [&[T::Level]; $channels],
        ) -> Result<()> {
            T::dispatch(stream_context, source, histograms, levels)
        }
    };
}

impl_histogram_range!(
    histogram_range_u8_c1_buffer_size,
    histogram_range_u8_c1,
    u8,
    i32,
    C1,
    nppiHistogramRangeGetBufferSize_8u_C1R_Ctx,
    nppiHistogramRange_8u_C1R_Ctx
);
impl_histogram_range!(
    histogram_range_u8_c3_buffer_size,
    histogram_range_u8_c3,
    u8,
    i32,
    C3,
    3,
    nppiHistogramRangeGetBufferSize_8u_C3R_Ctx,
    nppiHistogramRange_8u_C3R_Ctx
);
impl_histogram_range!(
    histogram_range_u8_c4_buffer_size,
    histogram_range_u8_c4,
    u8,
    i32,
    C4,
    4,
    nppiHistogramRangeGetBufferSize_8u_C4R_Ctx,
    nppiHistogramRange_8u_C4R_Ctx
);
impl_histogram_range!(
    histogram_range_u8_ac4_buffer_size,
    histogram_range_u8_ac4,
    u8,
    i32,
    AC4,
    3,
    nppiHistogramRangeGetBufferSize_8u_AC4R_Ctx,
    nppiHistogramRange_8u_AC4R_Ctx
);
impl_histogram_range!(
    histogram_range_u16_c1_buffer_size,
    histogram_range_u16_c1,
    u16,
    i32,
    C1,
    nppiHistogramRangeGetBufferSize_16u_C1R_Ctx,
    nppiHistogramRange_16u_C1R_Ctx
);
impl_histogram_range!(
    histogram_range_u16_c3_buffer_size,
    histogram_range_u16_c3,
    u16,
    i32,
    C3,
    3,
    nppiHistogramRangeGetBufferSize_16u_C3R_Ctx,
    nppiHistogramRange_16u_C3R_Ctx
);
impl_histogram_range!(
    histogram_range_u16_c4_buffer_size,
    histogram_range_u16_c4,
    u16,
    i32,
    C4,
    4,
    nppiHistogramRangeGetBufferSize_16u_C4R_Ctx,
    nppiHistogramRange_16u_C4R_Ctx
);
impl_histogram_range!(
    histogram_range_u16_ac4_buffer_size,
    histogram_range_u16_ac4,
    u16,
    i32,
    AC4,
    3,
    nppiHistogramRangeGetBufferSize_16u_AC4R_Ctx,
    nppiHistogramRange_16u_AC4R_Ctx
);
impl_histogram_range!(
    histogram_range_i16_c1_buffer_size,
    histogram_range_i16_c1,
    i16,
    i32,
    C1,
    nppiHistogramRangeGetBufferSize_16s_C1R_Ctx,
    nppiHistogramRange_16s_C1R_Ctx
);
impl_histogram_range!(
    histogram_range_i16_c3_buffer_size,
    histogram_range_i16_c3,
    i16,
    i32,
    C3,
    3,
    nppiHistogramRangeGetBufferSize_16s_C3R_Ctx,
    nppiHistogramRange_16s_C3R_Ctx
);
impl_histogram_range!(
    histogram_range_i16_c4_buffer_size,
    histogram_range_i16_c4,
    i16,
    i32,
    C4,
    4,
    nppiHistogramRangeGetBufferSize_16s_C4R_Ctx,
    nppiHistogramRange_16s_C4R_Ctx
);
impl_histogram_range!(
    histogram_range_i16_ac4_buffer_size,
    histogram_range_i16_ac4,
    i16,
    i32,
    AC4,
    3,
    nppiHistogramRangeGetBufferSize_16s_AC4R_Ctx,
    nppiHistogramRange_16s_AC4R_Ctx
);
impl_histogram_range!(
    histogram_range_f32_c1_buffer_size,
    histogram_range_f32_c1,
    f32,
    f32,
    C1,
    nppiHistogramRangeGetBufferSize_32f_C1R_Ctx,
    nppiHistogramRange_32f_C1R_Ctx
);
impl_histogram_range!(
    histogram_range_f32_c3_buffer_size,
    histogram_range_f32_c3,
    f32,
    f32,
    C3,
    3,
    nppiHistogramRangeGetBufferSize_32f_C3R_Ctx,
    nppiHistogramRange_32f_C3R_Ctx
);
impl_histogram_range!(
    histogram_range_f32_c4_buffer_size,
    histogram_range_f32_c4,
    f32,
    f32,
    C4,
    4,
    nppiHistogramRangeGetBufferSize_32f_C4R_Ctx,
    nppiHistogramRange_32f_C4R_Ctx
);
impl_histogram_range!(
    histogram_range_f32_ac4_buffer_size,
    histogram_range_f32_ac4,
    f32,
    f32,
    AC4,
    3,
    nppiHistogramRangeGetBufferSize_32f_AC4R_Ctx,
    nppiHistogramRange_32f_AC4R_Ctx
);
impl_generic_histogram_range_c1!(
    HistogramRangeC1,
    histogram_range_c1,
    histogram_range_c1_buffer_size,
    [
        (
            u8,
            i32,
            histogram_range_u8_c1,
            histogram_range_u8_c1_buffer_size
        ),
        (
            u16,
            i32,
            histogram_range_u16_c1,
            histogram_range_u16_c1_buffer_size
        ),
        (
            i16,
            i32,
            histogram_range_i16_c1,
            histogram_range_i16_c1_buffer_size
        ),
        (
            f32,
            f32,
            histogram_range_f32_c1,
            histogram_range_f32_c1_buffer_size
        ),
    ]
);
impl_generic_histogram_range_packed!(
    HistogramRangeC3,
    histogram_range_c3,
    histogram_range_c3_buffer_size,
    C3,
    3,
    [
        (
            u8,
            i32,
            histogram_range_u8_c3,
            histogram_range_u8_c3_buffer_size
        ),
        (
            u16,
            i32,
            histogram_range_u16_c3,
            histogram_range_u16_c3_buffer_size
        ),
        (
            i16,
            i32,
            histogram_range_i16_c3,
            histogram_range_i16_c3_buffer_size
        ),
        (
            f32,
            f32,
            histogram_range_f32_c3,
            histogram_range_f32_c3_buffer_size
        ),
    ]
);
impl_generic_histogram_range_packed!(
    HistogramRangeC4,
    histogram_range_c4,
    histogram_range_c4_buffer_size,
    C4,
    4,
    [
        (
            u8,
            i32,
            histogram_range_u8_c4,
            histogram_range_u8_c4_buffer_size
        ),
        (
            u16,
            i32,
            histogram_range_u16_c4,
            histogram_range_u16_c4_buffer_size
        ),
        (
            i16,
            i32,
            histogram_range_i16_c4,
            histogram_range_i16_c4_buffer_size
        ),
        (
            f32,
            f32,
            histogram_range_f32_c4,
            histogram_range_f32_c4_buffer_size
        ),
    ]
);
impl_generic_histogram_range_packed!(
    HistogramRangeAC4,
    histogram_range_ac4,
    histogram_range_ac4_buffer_size,
    AC4,
    3,
    [
        (
            u8,
            i32,
            histogram_range_u8_ac4,
            histogram_range_u8_ac4_buffer_size
        ),
        (
            u16,
            i32,
            histogram_range_u16_ac4,
            histogram_range_u16_ac4_buffer_size
        ),
        (
            i16,
            i32,
            histogram_range_i16_ac4,
            histogram_range_i16_ac4_buffer_size
        ),
        (
            f32,
            f32,
            histogram_range_f32_ac4,
            histogram_range_f32_ac4_buffer_size
        ),
    ]
);
