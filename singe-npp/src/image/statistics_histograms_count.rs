use super::*;

macro_rules! impl_count_in_range {
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
            counts: &mut DeviceMemory<i32>,
            lower_bound: $ty,
            upper_bound: $ty,
        ) -> Result<()> {
            validate_index_output(counts, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    counts.as_mut_ptr().cast(),
                    lower_bound,
                    upper_bound,
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
            counts: &mut DeviceMemory<i32>,
            lower_bound: [$ty; 3],
            upper_bound: [$ty; 3],
        ) -> Result<()> {
            validate_index_output(counts, 3)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    counts.as_mut_ptr().cast(),
                    lower_bound.as_ptr().cast_mut(),
                    upper_bound.as_ptr().cast_mut(),
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
            source: &ImageView<'_, $ty, AC4>,
            counts: &mut DeviceMemory<i32>,
            lower_bound: [$ty; 3],
            upper_bound: [$ty; 3],
        ) -> Result<()> {
            validate_index_output(counts, 3)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    counts.as_mut_ptr().cast(),
                    lower_bound.as_ptr().cast_mut(),
                    upper_bound.as_ptr().cast_mut(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_count_in_range_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                counts: &mut DeviceMemory<i32>,
                lower_bound: Self,
                upper_bound: Self,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    counts: &mut DeviceMemory<i32>,
                    lower_bound: Self,
                    upper_bound: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, counts, lower_bound, upper_bound)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            counts: &mut DeviceMemory<i32>,
            lower_bound: T,
            upper_bound: T,
        ) -> Result<()> {
            T::dispatch(stream_context, source, counts, lower_bound, upper_bound)
        }
    };
}

macro_rules! impl_generic_count_in_range_packed {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                counts: &mut DeviceMemory<i32>,
                lower_bound: [Self; 3],
                upper_bound: [Self; 3],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    counts: &mut DeviceMemory<i32>,
                    lower_bound: [Self; 3],
                    upper_bound: [Self; 3],
                ) -> Result<()> {
                    $direct(stream_context, source, counts, lower_bound, upper_bound)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            counts: &mut DeviceMemory<i32>,
            lower_bound: [T; 3],
            upper_bound: [T; 3],
        ) -> Result<()> {
            T::dispatch(stream_context, source, counts, lower_bound, upper_bound)
        }
    };
}

impl_count_in_range!(
    count_in_range_u8_c1_buffer_size,
    count_in_range_u8_c1,
    u8,
    C1,
    nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx,
    nppiCountInRange_8u_C1R_Ctx
);
impl_count_in_range!(
    count_in_range_f32_c1_buffer_size,
    count_in_range_f32_c1,
    f32,
    C1,
    nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx,
    nppiCountInRange_32f_C1R_Ctx
);
impl_count_in_range!(
    count_in_range_u8_c3_buffer_size,
    count_in_range_u8_c3,
    u8,
    C3,
    nppiCountInRangeGetBufferHostSize_8u_C3R_Ctx,
    nppiCountInRange_8u_C3R_Ctx
);
impl_count_in_range!(
    count_in_range_f32_c3_buffer_size,
    count_in_range_f32_c3,
    f32,
    C3,
    nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx,
    nppiCountInRange_32f_C3R_Ctx
);
impl_count_in_range!(
    count_in_range_u8_ac4_buffer_size,
    count_in_range_u8_ac4,
    u8,
    AC4,
    nppiCountInRangeGetBufferHostSize_8u_AC4R_Ctx,
    nppiCountInRange_8u_AC4R_Ctx
);
impl_count_in_range!(
    count_in_range_f32_ac4_buffer_size,
    count_in_range_f32_ac4,
    f32,
    AC4,
    nppiCountInRangeGetBufferHostSize_32f_AC4R_Ctx,
    nppiCountInRange_32f_AC4R_Ctx
);
impl_generic_count_in_range_c1!(
    CountInRangeC1,
    count_in_range_c1,
    count_in_range_c1_buffer_size,
    [
        (u8, count_in_range_u8_c1, count_in_range_u8_c1_buffer_size),
        (
            f32,
            count_in_range_f32_c1,
            count_in_range_f32_c1_buffer_size
        ),
    ]
);
impl_generic_count_in_range_packed!(
    CountInRangeC3,
    count_in_range_c3,
    count_in_range_c3_buffer_size,
    C3,
    [
        (u8, count_in_range_u8_c3, count_in_range_u8_c3_buffer_size),
        (
            f32,
            count_in_range_f32_c3,
            count_in_range_f32_c3_buffer_size
        ),
    ]
);
impl_generic_count_in_range_packed!(
    CountInRangeAC4,
    count_in_range_ac4,
    count_in_range_ac4_buffer_size,
    AC4,
    [
        (u8, count_in_range_u8_ac4, count_in_range_u8_ac4_buffer_size),
        (
            f32,
            count_in_range_f32_ac4,
            count_in_range_f32_ac4_buffer_size
        ),
    ]
);
