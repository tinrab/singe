use singe_cuda::{memory::DeviceMemory, view::DeviceOutput};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{
            validate_histogram_range_level_arrays, validate_histogram_range_levels,
            validate_histogram_range_output, validate_histogram_range_outputs,
        },
        view::{AC4, C1, C3, C4, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

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
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        /// Returns the scratch size in bytes for range histograms over `roi`.
        ///
        /// Only the lengths of the level arrays affect this query; their values are not read.
        ///
        /// # Errors
        ///
        /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: &[$level_ty],
        ) -> Result<usize> {
            let raw_levels = validate_histogram_range_levels(levels)?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
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

        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Computes per-channel histograms using caller-supplied bin boundaries.
        /// Processes the single source channel.
        ///
        /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
        /// Each histogram has exactly one fewer entry than its level array.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level array must have at least two entries and a length representable as `i32`.
        /// Each output length must equal its corresponding level count minus one.
        /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
        /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
        /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
        /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
        /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
        /// Keep source, level, and histogram allocations valid until this call completes its native work.
        /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            histogram: &mut (impl DeviceOutput<i32> + ?Sized),
            levels: &[$level_ty],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            let raw_levels = validate_histogram_range_output(stream_context, histogram, levels)?;
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
                    levels.as_ptr().cast(),
                    raw_levels,
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
        $level_ty:ty,
        C3,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        /// Returns the scratch size in bytes for range histograms over `roi`.
        ///
        /// Only the lengths of the level arrays affect this query; their values are not read.
        ///
        /// # Errors
        ///
        /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[$level_ty]; $channels],
        ) -> Result<usize> {
            let mut raw_levels = validate_histogram_range_level_arrays(&levels)?;
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

        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Computes per-channel histograms using caller-supplied bin boundaries.
        /// Processes all three source channels.
        ///
        /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
        /// Each histogram has exactly one fewer entry than its level array.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level array must have at least two entries and a length representable as `i32`.
        /// Each output length must equal its corresponding level count minus one.
        /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
        /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
        /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
        /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
        /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
        /// Keep source, level, and histogram allocations valid until this call completes its native work.
        /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mut histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [&[$level_ty]; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            let mut raw_levels =
                validate_histogram_range_outputs(stream_context, &histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut level_ptrs = levels.map(|level| level.as_ptr());

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    level_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
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
        $level_ty:ty,
        C4,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        /// Returns the scratch size in bytes for range histograms over `roi`.
        ///
        /// Only the lengths of the level arrays affect this query; their values are not read.
        ///
        /// # Errors
        ///
        /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[$level_ty]; $channels],
        ) -> Result<usize> {
            let mut raw_levels = validate_histogram_range_level_arrays(&levels)?;
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

        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Computes per-channel histograms using caller-supplied bin boundaries.
        /// Processes all four source channels.
        ///
        /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
        /// Each histogram has exactly one fewer entry than its level array.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level array must have at least two entries and a length representable as `i32`.
        /// Each output length must equal its corresponding level count minus one.
        /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
        /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
        /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
        /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
        /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
        /// Keep source, level, and histogram allocations valid until this call completes its native work.
        /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C4>,
            mut histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [&[$level_ty]; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            let mut raw_levels =
                validate_histogram_range_outputs(stream_context, &histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut level_ptrs = levels.map(|level| level.as_ptr());

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    level_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
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
        $level_ty:ty,
        AC4,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        /// Returns the scratch size in bytes for range histograms over `roi`.
        ///
        /// Only the lengths of the level arrays affect this query; their values are not read.
        ///
        /// # Errors
        ///
        /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[$level_ty]; $channels],
        ) -> Result<usize> {
            let mut raw_levels = validate_histogram_range_level_arrays(&levels)?;
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

        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Computes per-channel histograms using caller-supplied bin boundaries.
        /// Ignores source alpha and computes histograms for the three color channels.
        ///
        /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
        /// Each histogram has exactly one fewer entry than its level array.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level array must have at least two entries and a length representable as `i32`.
        /// Each output length must equal its corresponding level count minus one.
        /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
        /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
        /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
        /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
        /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
        /// Keep source, level, and histogram allocations valid until this call completes its native work.
        /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, AC4>,
            mut histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [&[$level_ty]; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.stream().ensure_not_capturing()?;

            let mut raw_levels =
                validate_histogram_range_outputs(stream_context, &histograms, &levels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), levels)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;
            let mut histogram_ptrs = histograms
                .each_mut()
                .map(|histogram| histogram.as_mut_ptr());
            let mut level_ptrs = levels.map(|level| level.as_ptr());

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    histogram_ptrs.as_mut_ptr().cast(),
                    level_ptrs.as_mut_ptr().cast(),
                    raw_levels.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_histogram_range_c1 {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, [$(($ty:ty, $level_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        /// Dispatches range histogram queries and computation for a native source element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native source and level element representations and uphold the documented histogram contract.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Native bin-boundary element type, `i32` for integer sources and `f32` for floating-point sources.
            type Level: DataTypeLike;

            /// Returns the scratch size in bytes for range histograms over `roi`.
            ///
            /// Only the lengths of the level arrays affect this query; their values are not read.
            ///
            /// # Errors
            ///
            /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: &[Self::Level],
            ) -> Result<usize>;

            /// Computes per-channel histograms using caller-supplied bin boundaries.
            /// The C1, C3, and C4 variants process every source channel; AC4 excludes source alpha.
            ///
            /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
            /// Each histogram has exactly one fewer entry than its level array.
            /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
            /// Each level array must have at least two entries and a length representable as `i32`.
            /// Each output length must equal its corresponding level count minus one.
            /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// The source must be initialized and ready for native execution.
            /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
            /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
            /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
            /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
            /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
            /// Keep source, level, and histogram allocations valid until this call completes its native work.
            /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                histogram: &mut (impl DeviceOutput<i32> + ?Sized),
                levels: &[Self::Level],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                type Level = $level_ty;

                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: &[Self::Level],
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    histogram: &mut (impl DeviceOutput<i32> + ?Sized),
                    levels: &[Self::Level],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, histogram, levels)
                    }
}
            }
        )+

        /// Returns the scratch size in bytes for range histograms over `roi`.
        ///
        /// Only the lengths of the level arrays affect this query; their values are not read.
        ///
        /// # Errors
        ///
        /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: &[T::Level],
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        /// Computes per-channel histograms using caller-supplied bin boundaries.
        /// The C1, C3, and C4 variants process every source channel; AC4 excludes source alpha.
        ///
        /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
        /// Each histogram has exactly one fewer entry than its level array.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level array must have at least two entries and a length representable as `i32`.
        /// Each output length must equal its corresponding level count minus one.
        /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
        /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
        /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
        /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
        /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
        /// Keep source, level, and histogram allocations valid until this call completes its native work.
        /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            histogram: &mut (impl DeviceOutput<i32> + ?Sized),
            levels: &[T::Level],
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, histogram, levels)
            }
}
    };
}

macro_rules! impl_generic_histogram_range_packed {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, $channels:expr, [$(($ty:ty, $level_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        /// Dispatches range histogram queries and computation for a native source element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native source and level element representations and uphold the documented histogram contract.
        pub unsafe trait $trait_name: DataTypeLike {
            /// Native bin-boundary element type, `i32` for integer sources and `f32` for floating-point sources.
            type Level: DataTypeLike;

            /// Returns the scratch size in bytes for range histograms over `roi`.
            ///
            /// Only the lengths of the level arrays affect this query; their values are not read.
            ///
            /// # Errors
            ///
            /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
            fn buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                levels: [&[Self::Level]; $channels],
            ) -> Result<usize>;

            /// Computes per-channel histograms using caller-supplied bin boundaries.
            /// The C1, C3, and C4 variants process every source channel; AC4 excludes source alpha.
            ///
            /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
            /// Each histogram has exactly one fewer entry than its level array.
            /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
            /// Each level array must have at least two entries and a length representable as `i32`.
            /// Each output length must equal its corresponding level count minus one.
            /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// The source must be initialized and ready for native execution.
            /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
            /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
            /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
            /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
            /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
            /// Keep source, level, and histogram allocations valid until this call completes its native work.
            /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
                levels: [&[Self::Level]; $channels],
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name for $ty {
                type Level = $level_ty;

                fn buffer_size(
                    stream_context: &StreamContext,
                    roi: Size,
                    levels: [&[Self::Level]; $channels],
                ) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi, levels)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
                    levels: [&[Self::Level]; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, histograms, levels)
                    }
}
            }
        )+

        /// Returns the scratch size in bytes for range histograms over `roi`.
        ///
        /// Only the lengths of the level arrays affect this query; their values are not read.
        ///
        /// # Errors
        ///
        /// Returns an error if a level array has fewer than two entries, its length exceeds `i32::MAX`, or NPP rejects the workspace query.
        pub fn $buffer_size_name<T: $trait_name>(
            stream_context: &StreamContext,
            roi: Size,
            levels: [&[T::Level]; $channels],
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi, levels)
        }

        /// Computes per-channel histograms using caller-supplied bin boundaries.
        /// The C1, C3, and C4 variants process every source channel; AC4 excludes source alpha.
        ///
        /// Bin `k` counts source samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
        /// Each histogram has exactly one fewer entry than its level array.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or the source or any histogram belongs to another CUDA context.
        /// Each level array must have at least two entries and a length representable as `i32`.
        /// Each output length must equal its corresponding level count minus one.
        /// Capture checks, workspace queries, allocation, the native histogram call, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// The source must be initialized and ready for native execution.
        /// Level arrays must reside in storage readable by CUDA in this stream context while also remaining valid Rust slices.
        /// Ordinary host allocation alone does not satisfy the native device-pointer requirement.
        /// This wrapper forwards level pointers without copying their contents or checking their CUDA context.
        /// Level values must satisfy the native bin-boundary requirements; their order and contents are not checked.
        /// Keep each histogram exclusively writable and disjoint from the source, level arrays, and every other histogram during native use.
        /// Keep source, level, and histogram allocations valid until this call completes its native work.
        /// Image byte strides, element representations, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            histograms: [&mut (impl DeviceOutput<i32> + ?Sized); $channels],
            levels: [&[T::Level]; $channels],
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, histograms, levels)
            }
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
