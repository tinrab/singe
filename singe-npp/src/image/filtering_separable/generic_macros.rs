macro_rules! impl_generic_separable_integer_filter {
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            kernel: &[i32],
            anchor: i32,
            divisor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, kernel, anchor, divisor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    kernel: &[i32],
                    anchor: i32,
                    divisor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, kernel, anchor, divisor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_separable_typed_filter {
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                kernel: &[Self],
                anchor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            kernel: &[T],
            anchor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, kernel, anchor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    kernel: &[Self],
                    anchor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, kernel, anchor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_sum_window {
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, f32, Layout>,
                mask_size: i32,
                anchor: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, mask_size, anchor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, f32, $layout>,
                    mask_size: i32,
                    anchor: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, mask_size, anchor)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_sum_window_border {
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, f32, Layout>,
                mask_size: i32,
                anchor: i32,
                border_type: BorderType,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
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
                    mask_size,
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
                    destination: &mut ImageOutputView<'_, f32, $layout>,
                    mask_size: i32,
                    anchor: i32,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            mask_size,
                            anchor,
                            border_type,
                        )
                    }
                }
            }
        )*
    };
}
