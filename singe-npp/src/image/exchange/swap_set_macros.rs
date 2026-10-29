macro_rules! impl_swap_channels {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        /// Reorders image channels using the zero-based source indices in `order`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Destination channel `i` receives source channel `order[i]`.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be distinct indices below the processed channel count.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            order: [i32; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_order(&order, $channels as i32)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        /// Reorders image channels in place using the zero-based source indices in `order`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Destination channel `i` receives source channel `order[i]`.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Entries of `order` must be distinct indices below the processed channel count.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            order: [i32; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            validate_channel_order(&order, $channels as i32)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_c4_to_c3 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Reorders image channels using the zero-based source indices in `order`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Destination channel `i` receives source channel `order[i]`.
        /// Selects three of the four source channels.
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be distinct indices in `0..4`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C4>,
            destination: &mut ImageOutputView<'_, $ty, C3>,
            order: [i32; 3],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_order(&order, 4)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_c3_to_c4 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Reorders image channels using the zero-based source indices in `order`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Destination channel `i` receives source channel `order[i]`.
        /// An entry of `3` writes `value` to that destination channel instead of selecting a source channel.
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be in `0..=3`; source indices `0`, `1`, and `2` must not repeat, while fill entries may repeat.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            destination: &mut ImageOutputView<'_, $ty, C4>,
            order: [i32; 4],
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_order_with_fill(&order, 3, 4)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_swap_channels_ac4 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Reorders image channels using the zero-based source indices in `order`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Destination channel `i` receives source channel `order[i]`.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be distinct indices below the processed channel count.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, AC4>,
            destination: &mut ImageOutputView<'_, $ty, AC4>,
            order: [i32; 3],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_channel_order(&order, 3)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    order.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_swap_channels_operation {
    ($trait:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size, mask, and channel-order checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Reorders image channels using the zero-based source indices in `order`.
            /// Destination channel `i` receives source channel `order[i]`.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Entries of `order` must be distinct indices below the processed channel count.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                order: [i32; $channels],
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    order: [i32; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, order)
                    }
}
            }
        )+

        /// Reorders image channels using the zero-based source indices in `order`.
        /// Destination channel `i` receives source channel `order[i]`.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be distinct indices below the processed channel count.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            order: [i32; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, order)
            }
}
    };
}

macro_rules! impl_generic_swap_channels_in_place_operation {
    ($trait:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size, mask, and channel-order checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Reorders image channels in place using the zero-based source indices in `order`.
            /// Destination channel `i` receives source channel `order[i]`.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Entries of `order` must be distinct indices below the processed channel count.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image samples must be initialized and ready when read by the selected stream.
            /// Keep the image exclusively writable during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                order: [i32; $channels],
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    order: [i32; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, order)
                    }
}
            }
        )+

        /// Reorders image channels in place using the zero-based source indices in `order`.
        /// Destination channel `i` receives source channel `order[i]`.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Entries of `order` must be distinct indices below the processed channel count.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream.
        /// Keep the image exclusively writable during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            order: [i32; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, image, order)
            }
}
    };
}

macro_rules! impl_generic_swap_channels_c4_to_c3_operation {
    ($trait:ident, $function:ident, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size, mask, and channel-order checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            /// Reorders image channels using the zero-based source indices in `order`.
            /// Destination channel `i` receives source channel `order[i]`.
            /// Selects three of the four source channels.
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Entries of `order` must be distinct indices in `0..4`.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C4>,
                destination: &mut ImageOutputView<'_, Self, C3>,
                order: [i32; 3],
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C4>,
                    destination: &mut ImageOutputView<'_, Self, C3>,
                    order: [i32; 3],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, order)
                    }
}
            }
        )+

        /// Reorders image channels using the zero-based source indices in `order`.
        /// Destination channel `i` receives source channel `order[i]`.
        /// Selects three of the four source channels.
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be distinct indices in `0..4`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C4>,
            destination: &mut ImageOutputView<'_, T, C3>,
            order: [i32; 3],
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, order)
            }
}
    };
}

macro_rules! impl_generic_swap_channels_c3_to_c4_operation {
    ($trait:ident, $function:ident, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size, mask, and channel-order checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            /// Reorders image channels using the zero-based source indices in `order`.
            /// Destination channel `i` receives source channel `order[i]`.
            /// An entry of `3` writes `value` to that destination channel instead of selecting a source channel.
            ///
            /// Enqueues work on the selected stream.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Entries of `order` must be in `0..=3`; source indices `0`, `1`, and `2` must not repeat, while fill entries may repeat.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut ImageOutputView<'_, Self, C4>,
                order: [i32; 4],
                value: Self,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    destination: &mut ImageOutputView<'_, Self, C4>,
                    order: [i32; 4],
                    value: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, order, value)
                    }
}
            }
        )+

        /// Reorders image channels using the zero-based source indices in `order`.
        /// Destination channel `i` receives source channel `order[i]`.
        /// An entry of `3` writes `value` to that destination channel instead of selecting a source channel.
        ///
        /// Enqueues work on the selected stream.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Entries of `order` must be in `0..=3`; source indices `0`, `1`, and `2` must not repeat, while fill entries may repeat.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            destination: &mut ImageOutputView<'_, T, C4>,
            order: [i32; 4],
            value: T,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, order, value)
            }
}
    };
}

macro_rules! impl_set_masked {
    ($name:ident, $ty:ty, $value_ty:ty, $layout:ty, $ffi:ident, $value_expr:expr) => {
        /// Fills the destination image with `value` where the corresponding `mask` pixel is nonzero.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Pixels selected by zero mask values remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// The mask must have the destination image size.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Mask samples must be initialized and ready when read by the selected stream.
        /// Keep the mask read-only and destination storage exclusively writable and disjoint from the mask during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            value: $value_ty,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(destination)?;
            stream_context.validate_region(mask)?;

            validate_mask_size(destination.size(), mask.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    $value_expr(&value),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_set_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => ($value_ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size, mask, and channel-order checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar or per-channel pixel value accepted by the native fill routine.
            type Value;

            /// Fills the destination image with `value`.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the alpha channel is not processed.
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
            unsafe fn dispatch(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                type Value = $value_ty;

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    value: Self::Value,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe { $direct(stream_context, value, destination) }
                }
            }
        )+

        /// Fills the destination image with `value`.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            value: T::Value,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { T::dispatch(stream_context, value, destination) }
        }
    };
}

macro_rules! impl_generic_set_masked_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => ($value_ty:ty, $direct:ident)),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's size, mask, and channel-order checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Scalar or per-channel pixel value accepted by the native fill routine.
            type Value;

            /// Fills the destination image with `value` where the corresponding `mask` pixel is nonzero.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the alpha channel is not processed.
            /// Pixels selected by zero mask values remain unwritten.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// The mask must have the destination image size.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Mask samples must be initialized and ready when read by the selected stream.
            /// Keep the mask read-only and destination storage exclusively writable and disjoint from the mask during native use.
            /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                mask: &MaskView<'_>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                type Value = $value_ty;

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    value: Self::Value,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    mask: &MaskView<'_>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, value, destination, mask)
                    }
}
            }
        )+

        /// Fills the destination image with `value` where the corresponding `mask` pixel is nonzero.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the alpha channel is not processed.
        /// Pixels selected by zero mask values remain unwritten.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// The mask must have the destination image size.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Mask samples must be initialized and ready when read by the selected stream.
        /// Keep the mask read-only and destination storage exclusively writable and disjoint from the mask during native use.
        /// Retain participating image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            value: T::Value,
            destination: &mut ImageOutputView<'_, T, $layout>,
            mask: &MaskView<'_>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, value, destination, mask)
            }
}
    };
}
