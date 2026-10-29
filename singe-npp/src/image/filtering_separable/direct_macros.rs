macro_rules! impl_filter_separable {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `anchor` locates the kernel origin along the filtered axis.
        /// The convolved sum is divided by `divisor` before conversion to destination pixels.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// The kernel must be nonempty, its length must fit in `i32`, and `anchor` must be in `0..kernel.len()`.
        /// Returns an error if NPP rejects the filter parameters or divisor.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// The source allocation must include every neighboring sample addressed by the kernel or window, including any halo outside the view.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            anchor: i32,
            divisor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_any(kernel, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    anchor,
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_float {
    ($doc:literal, $name:ident, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `anchor` locates the kernel origin along the filtered axis.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// The kernel must be nonempty, its length must fit in `i32`, and `anchor` must be in `0..kernel.len()`.
        /// Returns an error if NPP rejects the filter parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// The source allocation must include every neighboring sample addressed by the kernel or window, including any halo outside the view.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            kernel: &[f32],
            anchor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_f32(kernel, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_float_typed {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `anchor` locates the kernel origin along the filtered axis.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// The kernel must be nonempty, its length must fit in `i32`, and `anchor` must be in `0..kernel.len()`.
        /// Returns an error if NPP rejects the filter parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// The source allocation must include every neighboring sample addressed by the kernel or window, including any halo outside the view.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[$pixel_ty],
            anchor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_any(kernel, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_sum_window {
    ($doc:literal, $name:ident, $source_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// `mask_size` is the window length; `anchor` locates its origin along the filtered axis.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
        /// Returns an error if NPP rejects the filter parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// The source allocation must include every neighboring sample addressed by the kernel or window, including any halo outside the view.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_axis_mask(mask_size, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_sum_window_border {
    ($doc:literal, $name:ident, $source_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// `mask_size` is the window length; `anchor` locates its origin along the filtered axis.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
        /// Returns an error if NPP rejects the filter parameters or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
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
            validate_filter_axis_mask(mask_size, anchor)?;
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
                    mask_size,
                    anchor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
