use super::*;

macro_rules! impl_histogram_even {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: i32,
        ) -> Result<usize> {
            validate_histogram_levels(levels)?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    levels,
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
            levels: i32,
            lower_level: i32,
            upper_level: i32,
        ) -> Result<()> {
            validate_histogram_output(histogram, levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram.as_mut_ptr().cast(),
                    levels,
                    lower_level,
                    upper_level,
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
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            validate_histogram_levels_array(&levels)?;
            let mut raw_levels = levels;
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
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            validate_histogram_outputs(&histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut raw_levels = levels;
            let mut raw_lower_levels = lower_levels;
            let mut raw_upper_levels = upper_levels;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
                    raw_lower_levels.as_mut_ptr().cast(),
                    raw_upper_levels.as_mut_ptr().cast(),
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
        C4,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            validate_histogram_levels_array(&levels)?;
            let mut raw_levels = levels;
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
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            validate_histogram_outputs(&histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut raw_levels = levels;
            let mut raw_lower_levels = lower_levels;
            let mut raw_upper_levels = upper_levels;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
                    raw_lower_levels.as_mut_ptr().cast(),
                    raw_upper_levels.as_mut_ptr().cast(),
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
        AC4,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            validate_histogram_levels_array(&levels)?;
            let mut raw_levels = levels;
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
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            validate_histogram_outputs(&histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut raw_levels = levels;
            let mut raw_lower_levels = lower_levels;
            let mut raw_upper_levels = upper_levels;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
                    raw_lower_levels.as_mut_ptr().cast(),
                    raw_upper_levels.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_histogram_even_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: i32,
            ) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                histogram: &mut DeviceMemory<i32>,
                levels: i32,
                lower_level: i32,
                upper_level: i32,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: i32,
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    histogram: &mut DeviceMemory<i32>,
                    levels: i32,
                    lower_level: i32,
                    upper_level: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, histogram, levels, lower_level, upper_level)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: i32,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            histogram: &mut DeviceMemory<i32>,
            levels: i32,
            lower_level: i32,
            upper_level: i32,
        ) -> Result<()> {
            T::dispatch(
                stream_context,
                source,
                histogram,
                levels,
                lower_level,
                upper_level,
            )
        }
    };
}

macro_rules! impl_generic_histogram_even_packed {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, $channels:expr, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: [i32; $channels],
            ) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                histograms: [&mut DeviceMemory<i32>; $channels],
                levels: [i32; $channels],
                lower_levels: [i32; $channels],
                upper_levels: [i32; $channels],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: [i32; $channels],
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    histograms: [&mut DeviceMemory<i32>; $channels],
                    levels: [i32; $channels],
                    lower_levels: [i32; $channels],
                    upper_levels: [i32; $channels],
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        histograms,
                        levels,
                        lower_levels,
                        upper_levels,
                    )
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            histograms: [&mut DeviceMemory<i32>; $channels],
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            T::dispatch(
                stream_context,
                source,
                histograms,
                levels,
                lower_levels,
                upper_levels,
            )
        }
    };
}

impl_histogram_even!(
    histogram_even_u8_c1_buffer_size,
    histogram_even_u8_c1,
    u8,
    C1,
    nppiHistogramEvenGetBufferSize_8u_C1R_Ctx,
    nppiHistogramEven_8u_C1R_Ctx
);
impl_histogram_even!(
    histogram_even_u8_c3_buffer_size,
    histogram_even_u8_c3,
    u8,
    C3,
    3,
    nppiHistogramEvenGetBufferSize_8u_C3R_Ctx,
    nppiHistogramEven_8u_C3R_Ctx
);
impl_histogram_even!(
    histogram_even_u8_c4_buffer_size,
    histogram_even_u8_c4,
    u8,
    C4,
    4,
    nppiHistogramEvenGetBufferSize_8u_C4R_Ctx,
    nppiHistogramEven_8u_C4R_Ctx
);
impl_histogram_even!(
    histogram_even_u8_ac4_buffer_size,
    histogram_even_u8_ac4,
    u8,
    AC4,
    3,
    nppiHistogramEvenGetBufferSize_8u_AC4R_Ctx,
    nppiHistogramEven_8u_AC4R_Ctx
);
impl_histogram_even!(
    histogram_even_u16_c1_buffer_size,
    histogram_even_u16_c1,
    u16,
    C1,
    nppiHistogramEvenGetBufferSize_16u_C1R_Ctx,
    nppiHistogramEven_16u_C1R_Ctx
);
impl_histogram_even!(
    histogram_even_u16_c3_buffer_size,
    histogram_even_u16_c3,
    u16,
    C3,
    3,
    nppiHistogramEvenGetBufferSize_16u_C3R_Ctx,
    nppiHistogramEven_16u_C3R_Ctx
);
impl_histogram_even!(
    histogram_even_u16_c4_buffer_size,
    histogram_even_u16_c4,
    u16,
    C4,
    4,
    nppiHistogramEvenGetBufferSize_16u_C4R_Ctx,
    nppiHistogramEven_16u_C4R_Ctx
);
impl_histogram_even!(
    histogram_even_u16_ac4_buffer_size,
    histogram_even_u16_ac4,
    u16,
    AC4,
    3,
    nppiHistogramEvenGetBufferSize_16u_AC4R_Ctx,
    nppiHistogramEven_16u_AC4R_Ctx
);
impl_histogram_even!(
    histogram_even_i16_c1_buffer_size,
    histogram_even_i16_c1,
    i16,
    C1,
    nppiHistogramEvenGetBufferSize_16s_C1R_Ctx,
    nppiHistogramEven_16s_C1R_Ctx
);
impl_histogram_even!(
    histogram_even_i16_c3_buffer_size,
    histogram_even_i16_c3,
    i16,
    C3,
    3,
    nppiHistogramEvenGetBufferSize_16s_C3R_Ctx,
    nppiHistogramEven_16s_C3R_Ctx
);
impl_histogram_even!(
    histogram_even_i16_c4_buffer_size,
    histogram_even_i16_c4,
    i16,
    C4,
    4,
    nppiHistogramEvenGetBufferSize_16s_C4R_Ctx,
    nppiHistogramEven_16s_C4R_Ctx
);
impl_histogram_even!(
    histogram_even_i16_ac4_buffer_size,
    histogram_even_i16_ac4,
    i16,
    AC4,
    3,
    nppiHistogramEvenGetBufferSize_16s_AC4R_Ctx,
    nppiHistogramEven_16s_AC4R_Ctx
);
impl_generic_histogram_even_c1!(
    HistogramEvenC1,
    histogram_even_c1,
    histogram_even_c1_buffer_size,
    [
        (u8, histogram_even_u8_c1, histogram_even_u8_c1_buffer_size),
        (
            u16,
            histogram_even_u16_c1,
            histogram_even_u16_c1_buffer_size
        ),
        (
            i16,
            histogram_even_i16_c1,
            histogram_even_i16_c1_buffer_size
        ),
    ]
);
impl_generic_histogram_even_packed!(
    HistogramEvenC3,
    histogram_even_c3,
    histogram_even_c3_buffer_size,
    C3,
    3,
    [
        (u8, histogram_even_u8_c3, histogram_even_u8_c3_buffer_size),
        (
            u16,
            histogram_even_u16_c3,
            histogram_even_u16_c3_buffer_size
        ),
        (
            i16,
            histogram_even_i16_c3,
            histogram_even_i16_c3_buffer_size
        ),
    ]
);
impl_generic_histogram_even_packed!(
    HistogramEvenC4,
    histogram_even_c4,
    histogram_even_c4_buffer_size,
    C4,
    4,
    [
        (u8, histogram_even_u8_c4, histogram_even_u8_c4_buffer_size),
        (
            u16,
            histogram_even_u16_c4,
            histogram_even_u16_c4_buffer_size
        ),
        (
            i16,
            histogram_even_i16_c4,
            histogram_even_i16_c4_buffer_size
        ),
    ]
);
impl_generic_histogram_even_packed!(
    HistogramEvenAC4,
    histogram_even_ac4,
    histogram_even_ac4_buffer_size,
    AC4,
    3,
    [
        (u8, histogram_even_u8_ac4, histogram_even_u8_ac4_buffer_size),
        (
            u16,
            histogram_even_u16_ac4,
            histogram_even_u16_ac4_buffer_size
        ),
        (
            i16,
            histogram_even_i16_ac4,
            histogram_even_i16_ac4_buffer_size
        ),
    ]
);
