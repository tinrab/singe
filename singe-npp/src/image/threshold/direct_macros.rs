macro_rules! impl_threshold_c1 {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            threshold: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            threshold: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            threshold: $ty,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            thresholds: [$ty; $channels],
            values: [$ty; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            lower_threshold: $ty,
            lower_value: $ty,
            upper_threshold: $ty,
            upper_value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            lower_threshold: $ty,
            lower_value: $ty,
            upper_threshold: $ty,
            upper_value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            lower_thresholds: [$ty; $channels],
            lower_values: [$ty; $channels],
            upper_thresholds: [$ty; $channels],
            upper_values: [$ty; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the threshold or comparison parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            lower_thresholds: [$ty; $channels],
            lower_values: [$ty; $channels],
            upper_thresholds: [$ty; $channels],
            upper_values: [$ty; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
