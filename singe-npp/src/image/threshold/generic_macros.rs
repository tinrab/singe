macro_rules! impl_generic_threshold_scalar {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                threshold: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            threshold: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, threshold, operation)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    threshold: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, threshold, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_scalar_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, threshold, operation)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, threshold, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_array {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                thresholds: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            thresholds: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, thresholds, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, thresholds, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_array_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, thresholds, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, thresholds, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_scalar {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                threshold: Self,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            threshold: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, threshold)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    threshold: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, threshold)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_scalar_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, threshold)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, threshold)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_array {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                thresholds: [Self; $channels],
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            thresholds: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, thresholds)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, thresholds)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_array_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, thresholds)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, thresholds)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_scalar {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                threshold: Self,
                value: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            threshold: T,
            value: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, threshold, value, operation)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, threshold, value, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_scalar_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                value: Self,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            value: T,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, threshold, value, operation)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, threshold, value, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_array {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, thresholds, values, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, thresholds, values, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_value_array_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, thresholds, values, operation)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, thresholds, values, operation)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_scalar {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                threshold: Self,
                value: Self,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            threshold: T,
            value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, threshold, value)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, threshold, value)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_scalar_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                threshold: Self,
                value: Self,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            threshold: T,
            value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, threshold, value)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    threshold: Self,
                    value: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, threshold, value)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_array {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, thresholds, values)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, thresholds, values)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_fixed_value_array_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                thresholds: [Self; $channels],
                values: [Self; $channels],
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            thresholds: [T; $channels],
            values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, thresholds, values)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    thresholds: [Self; $channels],
                    values: [Self; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, thresholds, values)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_scalar {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                lower_threshold: Self,
                lower_value: Self,
                upper_threshold: Self,
                upper_value: Self,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            lower_threshold: T,
            lower_value: T,
            upper_threshold: T,
            upper_value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    destination,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )
            }
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    lower_threshold: Self,
                    lower_value: Self,
                    upper_threshold: Self,
                    upper_value: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            destination,
                            lower_threshold,
                            lower_value,
                            upper_threshold,
                            upper_value,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_scalar_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                lower_threshold: Self,
                lower_value: Self,
                upper_threshold: Self,
                upper_value: Self,
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            lower_threshold: T,
            lower_value: T,
            upper_threshold: T,
            upper_value: T,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    image,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )
            }
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    lower_threshold: Self,
                    lower_value: Self,
                    upper_threshold: Self,
                    upper_value: Self,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            image,
                            lower_threshold,
                            lower_value,
                            upper_threshold,
                            upper_value,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_array {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                lower_thresholds: [Self; $channels],
                lower_values: [Self; $channels],
                upper_thresholds: [Self; $channels],
                upper_values: [Self; $channels],
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            lower_thresholds: [T; $channels],
            lower_values: [T; $channels],
            upper_thresholds: [T; $channels],
            upper_values: [T; $channels],
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
                    lower_thresholds,
                    lower_values,
                    upper_thresholds,
                    upper_values,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    lower_thresholds: [Self; $channels],
                    lower_values: [Self; $channels],
                    upper_thresholds: [Self; $channels],
                    upper_values: [Self; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            destination,
                            lower_thresholds,
                            lower_values,
                            upper_thresholds,
                            upper_values,
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! impl_generic_threshold_less_greater_value_array_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible threshold routine.
        /// They must uphold the method's size checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
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
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                lower_thresholds: [Self; $channels],
                lower_values: [Self; $channels],
                upper_thresholds: [Self; $channels],
                upper_values: [Self; $channels],
            ) -> Result<()>;
        }

        #[doc = $doc]
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
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            lower_thresholds: [T; $channels],
            lower_values: [T; $channels],
            upper_thresholds: [T; $channels],
            upper_values: [T; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    image,
                    lower_thresholds,
                    lower_values,
                    upper_thresholds,
                    upper_values,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    lower_thresholds: [Self; $channels],
                    lower_values: [Self; $channels],
                    upper_thresholds: [Self; $channels],
                    upper_values: [Self; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            image,
                            lower_thresholds,
                            lower_values,
                            upper_thresholds,
                            upper_values,
                        )
                    }
                }
            }
        )*
    };
}
