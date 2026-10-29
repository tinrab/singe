use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::validate_index_output,
        view::{AC4, C1, C3, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

macro_rules! impl_count_in_range {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch size in bytes for counting in-range samples over `roi`.
        ///
        /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the workspace query.
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        /// Counts source samples inside each channel's inclusive intensity range.
        /// Processes the single source channel.
        /// `counts` must contain exactly one element.
        ///
        /// Both the lower and upper bounds are included.
        /// Each output element counts samples in the corresponding source channel.
        /// The wrapper passes bounds to NPP without checking their order or floating-point values.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
        /// NPP reports a range error if a lower bound exceeds its upper bound.
        /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
        /// Keep all image and output allocations valid until this call completes its native work.
        /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            counts: &mut DeviceOutputView<'_, i32>,
            lower_bound: $ty,
            upper_bound: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(counts)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_index_output(counts, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    counts.as_mut_ptr().cast(),
                    lower_bound,
                    upper_bound,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

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
        /// Returns the scratch size in bytes for counting in-range samples over `roi`.
        ///
        /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the workspace query.
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        /// Counts source samples inside each channel's inclusive intensity range.
        /// Processes all three source channels.
        /// `counts` must contain exactly three elements in source channel order.
        ///
        /// Both the lower and upper bounds are included.
        /// Each output element counts samples in the corresponding source channel.
        /// The wrapper passes bounds to NPP without checking their order or floating-point values.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
        /// NPP reports a range error if a lower bound exceeds its upper bound.
        /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
        /// Keep all image and output allocations valid until this call completes its native work.
        /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            counts: &mut DeviceOutputView<'_, i32>,
            lower_bound: [$ty; 3],
            upper_bound: [$ty; 3],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(counts)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_index_output(counts, 3)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    counts.as_mut_ptr().cast(),
                    lower_bound.as_ptr().cast_mut(),
                    upper_bound.as_ptr().cast_mut(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

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
        /// Returns the scratch size in bytes for counting in-range samples over `roi`.
        ///
        /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the workspace query.
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        /// Counts source samples inside each channel's inclusive intensity range.
        /// Ignores source alpha and processes the three color channels.
        /// `counts` must contain exactly three elements in source channel order.
        ///
        /// Both the lower and upper bounds are included.
        /// Each output element counts samples in the corresponding source channel.
        /// The wrapper passes bounds to NPP without checking their order or floating-point values.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
        /// NPP reports a range error if a lower bound exceeds its upper bound.
        /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
        /// Keep all image and output allocations valid until this call completes its native work.
        /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, AC4>,
            counts: &mut DeviceOutputView<'_, i32>,
            lower_bound: [$ty; 3],
            upper_bound: [$ty; 3],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(counts)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_index_output(counts, 3)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    counts.as_mut_ptr().cast(),
                    lower_bound.as_ptr().cast_mut(),
                    upper_bound.as_ptr().cast_mut(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_count_in_range_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        /// Dispatches range count queries and computation for a native source element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation and channel layout and uphold the documented execution contract.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch size in bytes for counting in-range samples over `roi`.
            ///
            /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
            ///
            /// # Errors
            ///
            /// Returns an error if NPP rejects the workspace query.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            /// Counts source samples inside each channel's inclusive intensity range.
            /// Processes the single source channel and writes one count.
            ///
            /// Both the lower and upper bounds are included.
            /// Each output element counts samples in the corresponding source channel.
            /// The wrapper passes bounds to NPP without checking their order or floating-point values.
            /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
            /// NPP reports a range error if a lower bound exceeds its upper bound.
            /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// The source must be initialized and ready for native execution.
            /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
            /// Keep all image and output allocations valid until this call completes its native work.
            /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                counts: &mut DeviceOutputView<'_, i32>,
                lower_bound: Self,
                upper_bound: Self,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    counts: &mut DeviceOutputView<'_, i32>,
                    lower_bound: Self,
                    upper_bound: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, counts, lower_bound, upper_bound)
                    }
}
            }
        )+

        /// Returns the scratch size in bytes for counting in-range samples over `roi`.
        ///
        /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the workspace query.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        /// Counts source samples inside each channel's inclusive intensity range.
        /// Processes the single source channel and writes one count.
        ///
        /// Both the lower and upper bounds are included.
        /// Each output element counts samples in the corresponding source channel.
        /// The wrapper passes bounds to NPP without checking their order or floating-point values.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
        /// NPP reports a range error if a lower bound exceeds its upper bound.
        /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
        /// Keep all image and output allocations valid until this call completes its native work.
        /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            counts: &mut DeviceOutputView<'_, i32>,
            lower_bound: T,
            upper_bound: T,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, counts, lower_bound, upper_bound)
            }
}
    };
}

macro_rules! impl_generic_count_in_range_packed {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        /// Dispatches range count queries and computation for a native source element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation and channel layout and uphold the documented execution contract.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch size in bytes for counting in-range samples over `roi`.
            ///
            /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
            ///
            /// # Errors
            ///
            /// Returns an error if NPP rejects the workspace query.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            /// Counts source samples inside each channel's inclusive intensity range.
            /// Processes three channels in source order and writes three counts; AC4 excludes source alpha.
            ///
            /// Both the lower and upper bounds are included.
            /// Each output element counts samples in the corresponding source channel.
            /// The wrapper passes bounds to NPP without checking their order or floating-point values.
            /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
            /// NPP reports a range error if a lower bound exceeds its upper bound.
            /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// The source must be initialized and ready for native execution.
            /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
            /// Keep all image and output allocations valid until this call completes its native work.
            /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                counts: &mut DeviceOutputView<'_, i32>,
                lower_bound: [Self; 3],
                upper_bound: [Self; 3],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    counts: &mut DeviceOutputView<'_, i32>,
                    lower_bound: [Self; 3],
                    upper_bound: [Self; 3],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, counts, lower_bound, upper_bound)
                    }
}
            }
        )+

        /// Returns the scratch size in bytes for counting in-range samples over `roi`.
        ///
        /// The returned size describes a device allocation, despite the native query name containing `HostSize`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the workspace query.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        /// Counts source samples inside each channel's inclusive intensity range.
        /// Processes three channels in source order and writes three counts; AC4 excludes source alpha.
        ///
        /// Both the lower and upper bounds are included.
        /// Each output element counts samples in the corresponding source channel.
        /// The wrapper passes bounds to NPP without checking their order or floating-point values.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source or counts belong to another CUDA context, or the counts length does not match the processed channel count.
        /// NPP reports a range error if a lower bound exceeds its upper bound.
        /// Capture checks, workspace queries, allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Keep outputs exclusively writable and disjoint from the source and each other during native use.
        /// Keep all image and output allocations valid until this call completes its native work.
        /// Image byte strides, alignment, element representations, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            counts: &mut DeviceOutputView<'_, i32>,
            lower_bound: [T; 3],
            upper_bound: [T; 3],
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, counts, lower_bound, upper_bound)
            }
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
