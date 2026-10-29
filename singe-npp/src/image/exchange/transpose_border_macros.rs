macro_rules! impl_transpose {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Destination width must equal source height, and destination height must equal source width.
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
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_transpose_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_transpose_operation {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's dimension and placement checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Destination width must equal source height, and destination height must equal source width.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Destination width must equal source height, and destination height must equal source width.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_copy_constant_border_scalar {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
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
            top: usize,
            left: usize,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            let (top, left) =
                validate_border_placement(source.size(), destination.size(), top, left)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    top,
                    left,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_constant_border_packed {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $value_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
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
            top: usize,
            left: usize,
            value: $value_ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            let (top, left) =
                validate_border_placement(source.size(), destination.size(), top, left)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    top,
                    left,
                    value.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_border {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
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
            top: usize,
            left: usize,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            let (top, left) =
                validate_border_placement(source.size(), destination.size(), top, left)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    top,
                    left,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_constant_border_operation {
    ($doc:literal, $trait:ident, $function:ident, $layout:ty, [$($ty:ty => ($value_ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's dimension and placement checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar or per-channel pixel value accepted by the native constant-border routine.
            type Value;

            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                top: usize,
                left: usize,
                value: Self::Value,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                type Value = $value_ty;

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    top: usize,
                    left: usize,
                    value: Self::Value,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, top, left, value)
                    }
}
            }
        )+

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            top: usize,
            left: usize,
            value: T::Value,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, top, left, value)
            }
}
    };
}

macro_rules! impl_generic_copy_border_operation {
    ($doc:literal, $trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's dimension and placement checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                top: usize,
                left: usize,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    top: usize,
                    left: usize,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, top, left)
                    }
}
            }
        )+

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `top` and `left` must fit in `i32`, their checked sums with the source dimensions must be representable, and the placed source must fit within the destination.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            top: usize,
            left: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, top, left)
            }
}
    };
}
