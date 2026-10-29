macro_rules! impl_filter_box_border_advanced {
    ($doc:literal, $buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident, $buffer_size_ffi:ident, $channels:expr) => {
        /// Returns the scratch size in bytes for advanced box filtering on `roi`.
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or channel count, or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name(roi: Size) -> Result<usize> {
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(roi.into(), $channels, &raw mut bytes,))?;
            }
            to_usize(bytes, "buffer size")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, and `anchor` must lie within the mask.
        /// Scratch-size queries, allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            let required_bytes = $buffer_size_name(destination.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    anchor.into(),
                    border_type.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_filter_advanced {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// Source and destination image sizes must match.
        /// `kernel` must be nonempty and its length must fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// The source allocation must include every neighboring sample read by the filter, including any halo outside the declared view.
        /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[f32],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_advanced_filter_kernel(kernel)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    kernel.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_advanced_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// `kernel` must be nonempty and its length must fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[f32],
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_advanced_filter_kernel(kernel)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    kernel.as_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_gauss_pyramid_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, and `rate` must be greater than `1.0` and at most `10.0`.
        /// `kernel` must be nonempty and its length must fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            rate: f32,
            kernel: &[f32],
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_gauss_pyramid_rate(rate)?;
            validate_advanced_filter_kernel(kernel)?;
            validate_source_offset(source.size(), source_offset)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    rate,
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    kernel.as_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_bilateral_gauss_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// `radius` must be in `1..=32`, and `step_between_source_pixels` must be positive.
        /// Both squared sigma parameters must compare greater than zero, and `border_type` must be `BorderType::Replicate`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            radius: i32,
            step_between_source_pixels: i32,
            value_square_sigma: f32,
            position_square_sigma: f32,
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_bilateral_gauss_parameters(
                radius,
                step_between_source_pixels,
                value_square_sigma,
                position_square_sigma,
                border_type,
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    radius,
                    step_between_source_pixels,
                    value_square_sigma,
                    position_square_sigma,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
