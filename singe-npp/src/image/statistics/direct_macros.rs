macro_rules! impl_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_dot_prod {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source_1.size(), source_2.size())?;
            validate_statistic_output(output, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_every_statistic_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Both images must contain initialized values ready for the selected stream.
        /// Keep the source read-only and the destination exclusively writable and disjoint
        /// through queued completion, including submission errors and captured graph replay.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), source_destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
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
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `output` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// `output` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_typed_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_type:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, $output_type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_typed_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_typed_pair_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_type:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `minimum` is ", stringify!($output_channels), ".")]
        #[doc = concat!("The required element count for `maximum` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            minimum: &mut DeviceOutputView<'_, $output_type>,
            maximum: &mut DeviceOutputView<'_, $output_type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(minimum)?;
            stream_context.validate_region(maximum)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_typed_statistic_output(minimum, $output_channels)?;
            validate_typed_statistic_output(maximum, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    minimum.as_mut_ptr().cast(),
                    maximum.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_typed_indexed_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_type:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($output_channels), ".")]
        #[doc = concat!("The required element count for `index_x` is ", stringify!($output_channels), ".")]
        #[doc = concat!("The required element count for `index_y` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, $output_type>,
            index_x: &mut DeviceOutputView<'_, i32>,
            index_y: &mut DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(output)?;
            stream_context.validate_region(index_x)?;
            stream_context.validate_region(index_y)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_typed_statistic_output(output, $output_channels)?;
            validate_index_output(index_x, $output_channels)?;
            validate_index_output(index_y, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    index_x.as_mut_ptr().cast(),
                    index_y.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_typed_pair_indexed_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// `minimum` must contain exactly 1 element.
        /// `maximum` must contain exactly 1 element.
        /// `minimum_index` must contain exactly 1 element.
        /// `maximum_index` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            minimum: &mut DeviceOutputView<'_, $output_type>,
            maximum: &mut DeviceOutputView<'_, $output_type>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(minimum)?;
            stream_context.validate_region(maximum)?;
            stream_context.validate_region(minimum_index)?;
            stream_context.validate_region(maximum_index)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// `channel` is zero-based and must be less than 3.
        /// `minimum` must contain exactly 1 element.
        /// `maximum` must contain exactly 1 element.
        /// `minimum_index` must contain exactly 1 element.
        /// `maximum_index` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            channel: usize,
            minimum: &mut DeviceOutputView<'_, $output_type>,
            maximum: &mut DeviceOutputView<'_, $output_type>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(minimum)?;
            stream_context.validate_region(maximum)?;
            stream_context.validate_region(minimum_index)?;
            stream_context.validate_region(maximum_index)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_typed_pair_indexed_statistic_masked {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `minimum` must contain exactly 1 element.
        /// `maximum` must contain exactly 1 element.
        /// `minimum_index` must contain exactly 1 element.
        /// `maximum_index` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            minimum: &mut DeviceOutputView<'_, $output_type>,
            maximum: &mut DeviceOutputView<'_, $output_type>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(minimum)?;
            stream_context.validate_region(maximum)?;
            stream_context.validate_region(minimum_index)?;
            stream_context.validate_region(maximum_index)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_mask_size(source.size(), mask.size())?;
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $output_type:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// `minimum` must contain exactly 1 element.
        /// `maximum` must contain exactly 1 element.
        /// `minimum_index` must contain exactly 1 element.
        /// `maximum_index` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            minimum: &mut DeviceOutputView<'_, $output_type>,
            maximum: &mut DeviceOutputView<'_, $output_type>,
            minimum_index: &mut DeviceOutputView<'_, Point>,
            maximum_index: &mut DeviceOutputView<'_, Point>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(minimum)?;
            stream_context.validate_region(maximum)?;
            stream_context.validate_region(minimum_index)?;
            stream_context.validate_region(maximum_index)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_typed_statistic_output(minimum, 1)?;
            validate_typed_statistic_output(maximum, 1)?;
            validate_point_output(minimum_index, 1)?;
            validate_point_output(maximum_index, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_norm_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_norm_statistic_masked {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `output` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// `output` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_error_metric {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source_1.size(), source_2.size())?;
            validate_statistic_output(output, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_quality_index {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $output_channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        #[doc = concat!("The required element count for `output` is ", stringify!($output_channels), ".")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source_1.size(), source_2.size())?;
            validate_metric_output(output, 1, $output_channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_error_metric_masked {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `output` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source_1.size(), source_2.size())?;
            validate_mask_size(source_1.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// `output` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Participating image ROI sizes must match.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, C3>,
            source_2: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            output: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_same_size(source_1.size(), source_2.size())?;
            validate_mask_size(source_1.size(), mask.size())?;
            validate_statistic_output(output, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_mean_stddev {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// `mean` must contain exactly 1 element.
        /// `standard_deviation` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mean: &mut DeviceOutputView<'_, f64>,
            standard_deviation: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    scratch.as_mut_ptr().cast(),
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// `channel` is zero-based and must be less than 3.
        /// `mean` must contain exactly 1 element.
        /// `standard_deviation` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            channel: usize,
            mean: &mut DeviceOutputView<'_, f64>,
            standard_deviation: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    coi,
                    scratch.as_mut_ptr().cast(),
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_mean_stddev_masked {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C1,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `mean` must contain exactly 1 element.
        /// `standard_deviation` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            mask: &ImageView<'_, u8, C1>,
            mean: &mut DeviceOutputView<'_, f64>,
            standard_deviation: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        C3,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for processing `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or cannot report the scratch size,
        /// or the returned size cannot be represented as `usize`.
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

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch storage and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// The mask must have the same ROI size as the source image.
        /// `channel` is zero-based and must be less than 3.
        /// `mean` must contain exactly 1 element.
        /// `standard_deviation` must contain exactly 1 element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image or result view belongs to another CUDA context.
        /// A mask with a different ROI size returns an error.
        /// An out-of-range `channel` returns an error.
        /// Result views must have the exact lengths documented above.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any mask must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all input and other result ranges
        /// until this call completes its native work.
        /// Pixel layout, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            mask: &ImageView<'_, u8, C1>,
            channel: usize,
            mean: &mut DeviceOutputView<'_, f64>,
            standard_deviation: &mut DeviceOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;
            stream_context.stream().ensure_not_capturing()?;

            let coi = validate_coi(channel, 3)?;
            validate_mask_size(source.size(), mask.size())?;
            validate_statistic_output(mean, 1)?;
            validate_statistic_output(standard_deviation, 1)?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                ))
            })?;

            Ok(())
        }
    };
}
