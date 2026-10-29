macro_rules! impl_copy {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Copies source pixels into the destination image.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
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
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_operation {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Copies source pixels into the destination image.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        /// Copies source pixels into the destination image.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
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

macro_rules! impl_copy_masked {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Copies source pixels into the destination where the corresponding `mask` pixel is nonzero.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Pixels selected by zero mask values remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// The mask must have the destination image size.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and mask samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.validate_region(mask)?;

            validate_same_size(source.size(), destination.size())?;
            validate_mask_size(source.size(), mask.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_masked_operation {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Copies source pixels into the destination where the corresponding `mask` pixel is nonzero.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Pixels selected by zero mask values remain unwritten.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// The mask must have the destination image size.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source and mask samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                mask: &MaskView<'_>,
            ) -> Result<()>;
        }

        /// Copies source pixels into the destination where the corresponding `mask` pixel is nonzero.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Pixels selected by zero mask values remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// The mask must have the destination image size.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source and mask samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, mask)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    mask: &MaskView<'_>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, mask)
                    }
}
            }
        )*
    };
}

macro_rules! impl_set {
    ($name:ident, $ty:ty, $value_ty:ty, $layout:ty, $ffi:ident, $value_expr:expr) => {
        /// Fills the destination image with `value`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Keep destination storage exclusively writable during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            value: $value_ty,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    $value_expr(&value),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_channel {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        /// Copies the zero-based `source_channel` into the zero-based `destination_channel`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_channel: usize,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            destination_channel: usize,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_index(source_channel, $channels)?;
            validate_channel_index(destination_channel, $channels)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().add(source_channel),
                    source.step(),
                    destination.as_mut_ptr().add(destination_channel),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Copies the zero-based `source_channel` into the zero-based `destination_channel`.
            ///
            /// Enqueues work on the selected stream.
            /// Other destination channels remain unwritten.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Channel indices must be smaller than the channel count of the corresponding packed image.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_channel: usize,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                destination_channel: usize,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_channel: usize,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    destination_channel: usize,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_channel,
                            destination,
                            destination_channel,
                        )
                    }
                }
            }
        )+

        /// Copies the zero-based `source_channel` into the zero-based `destination_channel`.
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_channel: usize,
            destination: &mut ImageOutputView<'_, T, $layout>,
            destination_channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(
                    stream_context,
                    source,
                    source_channel,
                    destination,
                    destination_channel,
                )
            }
        }
    };
}

macro_rules! impl_extract_channel {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        /// Copies the zero-based source `channel` into the single-channel destination.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            channel: usize,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_index(channel, $channels)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().add(channel),
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

macro_rules! impl_insert_channel {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        /// Copies the single-channel source into the zero-based destination `channel`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            channel: usize,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_index(channel, $channels)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().add(channel),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_extract_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Copies the zero-based source `channel` into the single-channel destination.
            ///
            /// Enqueues work on the selected stream.
            /// Other destination channels remain unwritten.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Channel indices must be smaller than the channel count of the corresponding packed image.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                channel: usize,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    channel: usize,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, channel)
                    }
}
            }
        )+

        /// Copies the zero-based source `channel` into the single-channel destination.
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, C1>,
            channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, channel)
            }
}
    };
}

macro_rules! impl_generic_insert_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Copies the single-channel source into the zero-based destination `channel`.
            ///
            /// Enqueues work on the selected stream.
            /// Other destination channels remain unwritten.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Channel indices must be smaller than the channel count of the corresponding packed image.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                channel: usize,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    channel: usize,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, channel)
                    }
}
            }
        )+

        /// Copies the single-channel source into the zero-based destination `channel`.
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, channel)
            }
}
    };
}

macro_rules! impl_generic_set_channel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Fills the zero-based destination `channel` with `value`.
            ///
            /// Enqueues work on the selected stream.
            /// Other destination channels remain unwritten.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Channel indices must be smaller than the channel count of the corresponding packed image.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Keep destination storage exclusively writable during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                value: Self,
                channel: usize,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    value: Self,
                    channel: usize,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, destination, value, channel)
                    }
}
            }
        )+

        /// Fills the zero-based destination `channel` with `value`.
        ///
        /// Enqueues work on the selected stream.
        /// Other destination channels remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Channel indices must be smaller than the channel count of the corresponding packed image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Keep destination storage exclusively writable during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            destination: &mut ImageOutputView<'_, T, $layout>,
            value: T,
            channel: usize,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, destination, value, channel)
            }
}
    };
}

macro_rules! impl_copy_packed_to_planar {
    ($name:ident, $ty:ty, $layout:ty, $planes:literal, $ffi:ident) => {
        /// Deinterleaves the packed source image into one destination plane per channel.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// All planes must satisfy the native common-size and common-byte-stride requirements.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.size(), destination.planes()[0].size())?;

            let destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_ptr().cast(),
                    destination.planes()[0].step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_copy_planar_to_packed {
    ($name:ident, $ty:ty, $layout:ty, $planes:literal, $ffi:ident) => {
        /// Interleaves the source planes into the packed destination image.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// All planes must satisfy the native common-size and common-byte-stride requirements.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(destination)?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.planes()[0].size(), destination.size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast(),
                    source.planes()[0].step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_packed_to_planar_operation {
    ($trait:ident, $function:ident, $layout:ty, $planes:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Deinterleaves the packed source image into one destination plane per channel.
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// All planes must satisfy the native common-size and common-byte-stride requirements.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut PlanarImageOutputView<'_, Self, $planes>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )+

        /// Deinterleaves the packed source image into one destination plane per channel.
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// All planes must satisfy the native common-size and common-byte-stride requirements.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut PlanarImageOutputView<'_, T, $planes>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination)
            }
}
    };
}

macro_rules! impl_generic_copy_planar_to_packed_operation {
    ($trait:ident, $function:ident, $layout:ty, $planes:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size and channel checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Interleaves the source planes into the packed destination image.
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// All planes must satisfy the native common-size and common-byte-stride requirements.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &PlanarImageView<'_, Self, $planes>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )+

        /// Interleaves the source planes into the packed destination image.
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destination planes during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// All planes must satisfy the native common-size and common-byte-stride requirements.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $planes>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination)
            }
}
    };
}
