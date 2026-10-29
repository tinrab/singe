macro_rules! impl_filter_integer {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// The convolved sum is divided by `divisor` before conversion to destination pixels.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_float_typed {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
            kernel_size: Size,
            anchor: Point,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_integer_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// The convolved sum is divided by `divisor` before conversion to destination pixels.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
        /// Returns an error if NPP rejects the filter parameters, divisor, or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
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
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
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
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    divisor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_float_typed_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
        /// Returns an error if NPP rejects the filter parameters or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[$pixel_ty],
            kernel_size: Size,
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

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
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
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_filter_integer {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible filtering routine.
        /// They must uphold the method's geometry checks, coefficient representation, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Supply `kernel` coefficients in the reverse order required by NPP.
            /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
            /// The convolved sum is divided by `divisor` before conversion to destination pixels.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
            /// Returns an error if source and destination sizes differ.
            /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                kernel: &[i32],
                kernel_size: Size,
                anchor: Point,
                divisor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// The convolved sum is divided by `divisor` before conversion to destination pixels.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    destination,
                    kernel,
                    kernel_size,
                    anchor,
                    divisor,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    kernel: &[i32],
                    kernel_size: Size,
                    anchor: Point,
                    divisor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            destination,
                            kernel,
                            kernel_size,
                            anchor,
                            divisor,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_filter_typed {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible filtering routine.
        /// They must uphold the method's geometry checks, coefficient representation, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Supply `kernel` coefficients in the reverse order required by NPP.
            /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
            /// Returns an error if source and destination sizes differ.
            /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                kernel: &[Self],
                kernel_size: Size,
                anchor: Point,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            kernel: &[T],
            kernel_size: Size,
            anchor: Point,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, kernel, kernel_size, anchor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    kernel: &[Self],
                    kernel_size: Size,
                    anchor: Point,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, kernel, kernel_size, anchor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_filter_integer_border {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible filtering routine.
        /// They must uphold the method's geometry checks, coefficient representation, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Supply `kernel` coefficients in the reverse order required by NPP.
            /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
            /// The convolved sum is divided by `divisor` before conversion to destination pixels.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
            /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
            /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
            /// Returns an error if NPP rejects the filter parameters, divisor, or border mode.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
            /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                kernel: &[i32],
                kernel_size: Size,
                anchor: Point,
                divisor: i32,
                border_type: BorderType,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// The convolved sum is divided by `divisor` before conversion to destination pixels.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
        /// Returns an error if NPP rejects the filter parameters, divisor, or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, T, $layout>,
            kernel: &[i32],
            kernel_size: Size,
            anchor: Point,
            divisor: i32,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    kernel,
                    kernel_size,
                    anchor,
                    divisor,
                    border_type,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    kernel: &[i32],
                    kernel_size: Size,
                    anchor: Point,
                    divisor: i32,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            kernel,
                            kernel_size,
                            anchor,
                            divisor,
                            border_type,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_filter_typed_border {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible filtering routine.
        /// They must uphold the method's geometry checks, coefficient representation, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Supply `kernel` coefficients in the reverse order required by NPP.
            /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
            /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
            /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
            /// Returns an error if NPP rejects the filter parameters or border mode.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
            /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                kernel: &[Self],
                kernel_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
        /// Returns an error if NPP rejects the filter parameters or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, T, $layout>,
            kernel: &[T],
            kernel_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    kernel,
                    kernel_size,
                    anchor,
                    border_type,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    kernel: &[Self],
                    kernel_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            kernel,
                            kernel_size,
                            anchor,
                            border_type,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_filter32f {
    ($doc:literal, $name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if source and destination sizes differ.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
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
            source: &ImageView<'_, $source_ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            kernel: &[f32],
            kernel_size: Size,
            anchor: Point,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter32f_border {
    ($doc:literal, $name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Supply `kernel` coefficients in the reverse order required by NPP.
        /// `kernel_size` gives its width and height; `anchor` locates its origin relative to the source pixel.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or either image belongs to another CUDA context.
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`, and `anchor` must lie within the kernel.
        /// Returns an error if NPP rejects the filter parameters or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            kernel: &[f32],
            kernel_size: Size,
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

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_kernel_2d(kernel, kernel_size, anchor)?;
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
                    kernel.as_ptr().cast(),
                    kernel_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_kernel_f32 {
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

macro_rules! impl_filter_separable_border {
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
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// The kernel must be nonempty, its length must fit in `i32`, and `anchor` must be in `0..kernel.len()`.
        /// Returns an error if NPP rejects the filter parameters, divisor, or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            anchor: i32,
            divisor: i32,
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
            validate_filter_kernel(kernel, anchor)?;
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
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    anchor,
                    divisor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_border_kernel_f32 {
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
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// The kernel must be nonempty, its length must fit in `i32`, and `anchor` must be in `0..kernel.len()`.
        /// Returns an error if NPP rejects the filter parameters or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            kernel: &[f32],
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
            validate_filter_kernel_f32(kernel, anchor)?;
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
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    anchor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_border_float {
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
        /// Returns an error if `source_offset` is outside the source image, its ROI endpoints overflow, or the destination-sized ROI extends past the source image.
        /// The kernel must be nonempty, its length must fit in `i32`, and `anchor` must be in `0..kernel.len()`.
        /// Returns an error if NPP rejects the filter parameters or border mode.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
        /// Pixel layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            kernel: &[f32],
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
            validate_filter_kernel_f32(kernel, anchor)?;
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
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len())
                        .map_err(|_| Error::out_of_range("kernel length"))?,
                    anchor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
