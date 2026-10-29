macro_rules! impl_template_match_full_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;

            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_full {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;

            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_same_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_same {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_valid_scaled {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;

            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_valid {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues the operation on the selected stream.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;

            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_full_scaled {
    ($doc:literal, $buffer_size_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi:ident) => {
        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
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
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_full {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
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
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_same_scaled {
    ($doc:literal, $buffer_size_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi:ident) => {
        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
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
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source.size(), destination.size())?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_same {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
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
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source.size(), destination.size())?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_valid_scaled {
    ($doc:literal, $buffer_size_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi:ident) => {
        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
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
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_valid {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
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
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_advanced_template_match_buffer_size {
    ($name:ident, $ffi:ident) => {
        /// Returns advanced template-matching scratch capacity, in bytes.
        #[doc = concat!("Queries [`sys::", stringify!($ffi), "`].")]
        ///
        /// `source_size` and `template_size` are image ROI sizes in pixels.
        /// `destination_element_bytes` specifies the size of one result element;
        /// `channels` is the number of channels processed.
        ///
        /// # Errors
        ///
        /// Returns an error if `destination_element_bytes` exceeds the `i32` range,
        /// NPP rejects the geometry, element size, or channel count, or the returned scratch size cannot be represented as `usize`.
        pub fn $name(
            source_size: Size,
            template_size: Size,
            destination_element_bytes: usize,
            channels: i32,
        ) -> Result<usize> {
            let destination_element_bytes = i32::try_from(destination_element_bytes)
                .map_err(|_| Error::out_of_range("destination element bytes"))?;
            let mut bytes = 0;
            // SAFETY: the `ffi` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_size.into(),
                    template_size.into(),
                    destination_element_bytes,
                    channels,
                    &raw mut bytes,
                ))?;
            }
            to_usize(bytes, "buffer size")
        }
    };
}

macro_rules! impl_template_match_norm_level_advanced_full {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $channels:expr,
        $norm_level_buffer_size_name:ident,
        $advanced_buffer_size_name:ident,
        $ffi:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let norm_level_bytes = $norm_level_buffer_size_name(stream_context, source.size())?;
            let advanced_bytes = $advanced_buffer_size_name(
                source.size(),
                template.size(),
                size_of::<$destination_ty>(),
                $channels,
            )?;
            let norm_level_scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), norm_level_bytes)?;
            let advanced_scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), advanced_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    norm_level_scratch.as_mut_ptr().cast(),
                    advanced_scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_advanced_same {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $channels:expr,
        $norm_level_buffer_size_name:ident,
        $advanced_buffer_size_name:ident,
        $ffi:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source.size(), destination.size())?;
            let norm_level_bytes = $norm_level_buffer_size_name(stream_context, source.size())?;
            let advanced_bytes = $advanced_buffer_size_name(
                source.size(),
                template.size(),
                size_of::<$destination_ty>(),
                $channels,
            )?;
            let norm_level_scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), norm_level_bytes)?;
            let advanced_scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), advanced_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    norm_level_scratch.as_mut_ptr().cast(),
                    advanced_scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_advanced_valid {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $channels:expr,
        $norm_level_buffer_size_name:ident,
        $advanced_buffer_size_name:ident,
        $ffi:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic overflow also returns an error.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(template)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let norm_level_bytes = $norm_level_buffer_size_name(stream_context, source.size())?;
            let advanced_bytes = $advanced_buffer_size_name(
                source.size(),
                template.size(),
                size_of::<$destination_ty>(),
                $channels,
            )?;
            let norm_level_scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), norm_level_bytes)?;
            let advanced_scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), advanced_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    norm_level_scratch.as_mut_ptr().cast(),
                    advanced_scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_template_match_scaled {
    ($doc:literal, $trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and destination dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and destination ABI.
        /// They must preserve the access, destination-size, and completion contracts of `dispatch`.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            #[doc = $doc]
            ///
            /// Enqueues the operation on the selected stream.
            /// Applies native result scaling by `2^(-scale_factor)`.
            /// Channels and padding left untouched by the native routine are not initialized by this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image belongs to another CUDA context.
            /// The destination must have the size required by the selected full, same, or valid mode.
            /// Destination-size arithmetic errors from the selected implementation are propagated.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source and template pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and the destination exclusively writable and disjoint from them
            /// through queued completion, including submission errors and every captured graph replay.
            /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, template, destination, scale_factor)
                    }
}
            }
        )+

        #[doc = $doc]
        ///
        /// Enqueues the operation on the selected stream.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic errors from the selected implementation are propagated.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, template, destination, scale_factor)
            }
}
    };
}

macro_rules! impl_generic_template_match {
    ($doc:literal, $trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $destination_ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and destination dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and destination ABI.
        /// They must preserve the access, destination-size, and completion contracts of `dispatch`.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Native destination element type selected for this source type.
            type Destination: DataTypeLike;

            #[doc = $doc]
            ///
            /// Enqueues the operation on the selected stream.
            /// Channels and padding left untouched by the native routine are not initialized by this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image belongs to another CUDA context.
            /// The destination must have the size required by the selected full, same, or valid mode.
            /// Destination-size arithmetic errors from the selected implementation are propagated.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source and template pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and the destination exclusively writable and disjoint from them
            /// through queued completion, including submission errors and every captured graph replay.
            /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self::Destination, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                type Destination = $destination_ty;

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, template, destination)
                    }
}
            }
        )+

        #[doc = $doc]
        ///
        /// Enqueues the operation on the selected stream.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic errors from the selected implementation are propagated.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// through queued completion, including submission errors and every captured graph replay.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T::Destination, $layout>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, template, destination)
            }
}
    };
}

macro_rules! impl_generic_template_match_buffered_scaled {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and destination dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and destination ABI.
        /// They must preserve the access, destination-size, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Returns an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Allocates temporary GPU scratch and returns after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Applies native result scaling by `2^(-scale_factor)`.
            /// Channels and padding left untouched by the native routine are not initialized by this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image belongs to another CUDA context.
            /// The destination must have the size required by the selected full, same, or valid mode.
            /// Destination-size arithmetic errors from the selected implementation are propagated.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source and template pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and the destination exclusively writable and disjoint from them
            /// until this call completes its native work.
            /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, template, destination, scale_factor)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Applies native result scaling by `2^(-scale_factor)`.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic errors from the selected implementation are propagated.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, template, destination, scale_factor)
            }
}
    };
}

macro_rules! impl_generic_template_match_buffered {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $destination_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and destination dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and destination ABI.
        /// They must preserve the access, destination-size, and completion contracts of `dispatch`.
        /// Buffer-size queries must return the required capacity in bytes.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Native destination element type selected for this source type.
            type Destination: DataTypeLike;

            /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
            ///
            /// # Errors
            ///
            /// Returns an error if NPP rejects the ROI or scratch-size query,
            /// or the returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Allocates temporary GPU scratch and returns after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Channels and padding left untouched by the native routine are not initialized by this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image belongs to another CUDA context.
            /// The destination must have the size required by the selected full, same, or valid mode.
            /// Destination-size arithmetic errors from the selected implementation are propagated.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source and template pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and the destination exclusively writable and disjoint from them
            /// until this call completes its native work.
            /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self::Destination, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                type Destination = $destination_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, template, destination)
                    }
}
            }
        )+

        /// Returns the scratch capacity, in bytes, for the selected operation and `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or scratch-size query,
        /// or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic errors from the selected implementation are propagated.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T::Destination, $layout>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, template, destination)
            }
}
    };
}

macro_rules! impl_generic_template_match_advanced {
    ($doc:literal, $trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $destination_ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and destination dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element, layout, and destination ABI.
        /// They must preserve the access, destination-size, and completion contracts of `dispatch`.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Native destination element type selected for this source type.
            type Destination: DataTypeLike;

            #[doc = $doc]
            ///
            /// Allocates temporary GPU scratch and returns after successful stream completion.
            /// This scratch-owning form rejects stream capture.
            /// Channels and padding left untouched by the native routine are not initialized by this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or an image belongs to another CUDA context.
            /// The destination must have the size required by the selected full, same, or valid mode.
            /// Destination-size arithmetic errors from the selected implementation are propagated.
            /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
            /// NPP operation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source and template pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and the destination exclusively writable and disjoint from them
            /// until this call completes its native work.
            /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self::Destination, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                type Destination = $destination_ty;

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, template, destination)
                    }
}
            }
        )+

        #[doc = $doc]
        ///
        /// Allocates temporary GPU scratch and returns after successful stream completion.
        /// This scratch-owning form rejects stream capture.
        /// Channels and padding left untouched by the native routine are not initialized by this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// The destination must have the size required by the selected full, same, or valid mode.
        /// Destination-size arithmetic errors from the selected implementation are propagated.
        /// Capture checks, scratch-size queries or allocation, and stream synchronization can also fail.
        /// NPP operation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and template pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and the destination exclusively writable and disjoint from them
        /// until this call completes its native work.
        /// Pixel layouts, byte strides, ROI bounds, and address alignment must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T::Destination, $layout>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, template, destination)
            }
}
    };
}

mod basic;

pub use basic::*;

mod correlation;

pub use correlation::*;

mod norm_level;

pub use norm_level::*;

mod advanced;

pub use advanced::*;
