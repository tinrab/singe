use singe_cuda::{memory::DeviceMemory, view::DeviceOutput};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{
            validate_histogram_levels, validate_histogram_levels_array, validate_histogram_output,
            validate_histogram_outputs,
        },
        view::{AC4, C1, C3, C4, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

macro_rules! impl_histogram_even {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
        ///
        /// Each level count specifies one more boundary than histogram bins.
        ///
        /// # Errors
        ///
        /// Returns an error if any level count is less than two or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: i32,
        ) -> Result<usize> {
            validate_histogram_levels(levels)?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
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

        /// Computes per-channel histograms with evenly spaced bins.
        /// Processes the single source channel.
        ///
        /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
        /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
        /// The wrapper passes range bounds to NPP without checking their order.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level count must be at least two, and each output length must equal its level count minus one.
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
            histogram: &mut (impl DeviceOutput<i32> + ?Sized),
            levels: i32,
            lower_level: i32,
            upper_level: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_histogram_output(stream_context, histogram, levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
        ///
        /// Each level count specifies one more boundary than histogram bins.
        ///
        /// # Errors
        ///
        /// Returns an error if any level count is less than two or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            validate_histogram_levels_array(&levels)?;
            let mut raw_levels = levels;
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
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

        /// Computes per-channel histograms with evenly spaced bins.
        /// Processes all three source channels.
        ///
        /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
        /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
        /// The wrapper passes range bounds to NPP without checking their order.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level count must be at least two, and each output length must equal its level count minus one.
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
            mut histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_histogram_outputs(stream_context, &histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut raw_levels = levels;
            let mut raw_lower_levels = lower_levels;
            let mut raw_upper_levels = upper_levels;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

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
        /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
        ///
        /// Each level count specifies one more boundary than histogram bins.
        ///
        /// # Errors
        ///
        /// Returns an error if any level count is less than two or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            validate_histogram_levels_array(&levels)?;
            let mut raw_levels = levels;
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
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

        /// Computes per-channel histograms with evenly spaced bins.
        /// Processes all four source channels.
        ///
        /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
        /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
        /// The wrapper passes range bounds to NPP without checking their order.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level count must be at least two, and each output length must equal its level count minus one.
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
            source: &ImageView<'_, $ty, C4>,
            mut histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_histogram_outputs(stream_context, &histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut raw_levels = levels;
            let mut raw_lower_levels = lower_levels;
            let mut raw_upper_levels = upper_levels;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
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
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
        ///
        /// Each level count specifies one more boundary than histogram bins.
        ///
        /// # Errors
        ///
        /// Returns an error if any level count is less than two or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            validate_histogram_levels_array(&levels)?;
            let mut raw_levels = levels;
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
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

        /// Computes per-channel histograms with evenly spaced bins.
        /// Ignores source alpha and processes the three color channels.
        ///
        /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
        /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
        /// The wrapper passes range bounds to NPP without checking their order.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level count must be at least two, and each output length must equal its level count minus one.
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
            mut histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_histogram_outputs(stream_context, &histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut raw_levels = levels;
            let mut raw_lower_levels = lower_levels;
            let mut raw_upper_levels = upper_levels;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_histogram_even_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        /// Dispatches even histogram queries and computation for a native source element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation and channel layout and uphold the documented execution contract.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
            ///
            /// Each level count specifies one more boundary than histogram bins.
            ///
            /// # Errors
            ///
            /// Returns an error if any level count is less than two or NPP rejects the workspace query.
            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: i32,
            ) -> Result<usize>;

            /// Computes per-channel histograms with evenly spaced bins.
            /// Processes the single source channel.
            ///
            /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
            /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
            /// The wrapper passes range bounds to NPP without checking their order.
            /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
            /// Each level count must be at least two, and each output length must equal its level count minus one.
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
                histogram: &mut (impl DeviceOutput<i32> + ?Sized),
                levels: i32,
                lower_level: i32,
                upper_level: i32,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: i32,
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    histogram: &mut (impl DeviceOutput<i32> + ?Sized),
                    levels: i32,
                    lower_level: i32,
                    upper_level: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, histogram, levels, lower_level, upper_level)
                    }
}
            }
        )+

        /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
        ///
        /// Each level count specifies one more boundary than histogram bins.
        ///
        /// # Errors
        ///
        /// Returns an error if any level count is less than two or NPP rejects the workspace query.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: i32,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        /// Computes per-channel histograms with evenly spaced bins.
        /// Processes the single source channel.
        ///
        /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
        /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
        /// The wrapper passes range bounds to NPP without checking their order.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level count must be at least two, and each output length must equal its level count minus one.
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
            histogram: &mut (impl DeviceOutput<i32> + ?Sized),
            levels: i32,
            lower_level: i32,
            upper_level: i32,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    histogram,
                    levels,
                    lower_level,
                    upper_level,
                )
            }
        }
    };
}

macro_rules! impl_generic_histogram_even_packed {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, $channels:expr, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        /// Dispatches even histogram queries and computation for a native source element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation and channel layout and uphold the documented execution contract.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
            ///
            /// Each level count specifies one more boundary than histogram bins.
            ///
            /// # Errors
            ///
            /// Returns an error if any level count is less than two or NPP rejects the workspace query.
            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: [i32; $channels],
            ) -> Result<usize>;

            /// Computes per-channel histograms with evenly spaced bins.
            /// Processes every source channel for C3 and C4; AC4 computes three histograms and excludes source alpha.
            ///
            /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
            /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
            /// The wrapper passes range bounds to NPP without checking their order.
            /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
            /// Each level count must be at least two, and each output length must equal its level count minus one.
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
                histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
                levels: [i32; $channels],
                lower_levels: [i32; $channels],
                upper_levels: [i32; $channels],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: [i32; $channels],
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
                    levels: [i32; $channels],
                    lower_levels: [i32; $channels],
                    upper_levels: [i32; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
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
            }
        )+

        /// Returns the scratch size in bytes for evenly spaced histograms over `roi`.
        ///
        /// Each level count specifies one more boundary than histogram bins.
        ///
        /// # Errors
        ///
        /// Returns an error if any level count is less than two or NPP rejects the workspace query.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: [i32; $channels],
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        /// Computes per-channel histograms with evenly spaced bins.
        /// Processes every source channel for C3 and C4; AC4 computes three histograms and excludes source alpha.
        ///
        /// For each processed channel, the lower bound is inclusive and the upper bound is exclusive.
        /// The range is divided into `levels - 1` bins; each output has exactly that many entries.
        /// The wrapper passes range bounds to NPP without checking their order.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level count must be at least two, and each output length must equal its level count minus one.
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
            histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [i32; $channels],
            lower_levels: [i32; $channels],
            upper_levels: [i32; $channels],
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    histograms,
                    levels,
                    lower_levels,
                    upper_levels,
                )
            }
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
