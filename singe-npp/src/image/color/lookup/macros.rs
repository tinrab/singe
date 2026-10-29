macro_rules! impl_lookup_table_c1 {
    ($doc:literal, $name:ident, $ty:ty, $value_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Source and destination image sizes must match.
        /// Images and both device tables must belong to the selected CUDA context.
        /// Value and level counts must match, contain at least two entries, and fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            values: &DeviceMemory<$value_ty>,
            levels: &DeviceMemory<$value_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.validate_region(values)?;
            stream_context.validate_region(levels)?;

            validate_same_size(source.size(), destination.size())?;
            let level_count = validate_lookup_table(values.len(), levels.len())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    values.as_ptr().cast(),
                    levels.as_ptr().cast(),
                    level_count,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_c1_in_place {
    ($doc:literal, $name:ident, $ty:ty, $value_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Updates the image in place.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Images and both device tables must belong to the selected CUDA context.
        /// Value and level counts must match, contain at least two entries, and fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C1>,
            values: &DeviceMemory<$value_ty>,
            levels: &DeviceMemory<$value_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;
            stream_context.validate_region(values)?;
            stream_context.validate_region(levels)?;

            let level_count = validate_lookup_table(values.len(), levels.len())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    values.as_ptr().cast(),
                    levels.as_ptr().cast(),
                    level_count,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_lookup_table_c1 {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and table element ABI and channel layout.
        /// They must uphold the method's table-shape checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike + Sized {
            /// Native element type of both the lookup values and their mapping levels.
            type Value: Copy;

            /// Converts color samples through lookup values and levels without interpolation.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, destination alpha is not processed.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Source and destination image sizes must match.
            /// Images and both device tables must belong to the selected CUDA context.
            /// Value and level counts must match, contain at least two entries, and fit in `i32`.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image inputs and lookup tables must be initialized and ready when read by native work.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
            /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
            /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                values: &DeviceMemory<Self::Value>,
                levels: &DeviceMemory<Self::Value>,
            ) -> Result<()>;
        }

        /// Converts color samples through lookup values and levels without interpolation.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Source and destination image sizes must match.
        /// Images and both device tables must belong to the selected CUDA context.
        /// Value and level counts must match, contain at least two entries, and fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            values: &DeviceMemory<T::Value>,
            levels: &DeviceMemory<T::Value>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, values, levels)
            }
}

        $(
            unsafe impl $trait for $ty {
                type Value = $value_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    values: &DeviceMemory<Self::Value>,
                    levels: &DeviceMemory<Self::Value>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, values, levels)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_c1_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and table element ABI and channel layout.
        /// They must uphold the method's table-shape checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike + Sized {
            /// Native element type of both the lookup values and their mapping levels.
            type Value: Copy;

            /// Converts color samples through lookup values and levels without interpolation.
            /// Updates the image in place.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, destination alpha is not processed.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Images and both device tables must belong to the selected CUDA context.
            /// Value and level counts must match, contain at least two entries, and fit in `i32`.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image inputs and lookup tables must be initialized and ready when read by native work.
            /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
            /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
            /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C1>,
                values: &DeviceMemory<Self::Value>,
                levels: &DeviceMemory<Self::Value>,
            ) -> Result<()>;
        }

        /// Converts color samples through lookup values and levels without interpolation.
        /// Updates the image in place.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Images and both device tables must belong to the selected CUDA context.
        /// Value and level counts must match, contain at least two entries, and fit in `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C1>,
            values: &DeviceMemory<T::Value>,
            levels: &DeviceMemory<T::Value>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, values, levels)
            }
}

        $(
            unsafe impl $trait for $ty {
                type Value = $value_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C1>,
                    values: &DeviceMemory<Self::Value>,
                    levels: &DeviceMemory<Self::Value>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, values, levels)
                    }
}
            }
        )*
    };
}

macro_rules! impl_lookup_table_packed {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Rejects stream capture and returns after successful stream completion.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Source and destination image sizes must match.
        /// Participating images must belong to the selected CUDA context.
        /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
        /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
        /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
        /// Keep image and table allocations valid until this call completes its native work.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            values: &[&DeviceMemory<i32>; $channels],
            levels: &[&DeviceMemory<i32>; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source.size(), destination.size())?;
            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors(values, levels)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_packed_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Updates the image in place.
        ///
        /// Rejects stream capture and returns after successful stream completion.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
        /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
        /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
        /// Keep image and table allocations valid until this call completes its native work.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            values: &[&DeviceMemory<i32>; $channels],
            levels: &[&DeviceMemory<i32>; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;
            stream_context.stream().ensure_not_capturing()?;

            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors(values, levels)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers and valid output locations.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_packed_f32 {
    ($doc:literal, $name:ident, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Source and destination image sizes must match.
        /// Participating images must belong to the selected CUDA context.
        /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
        /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            destination: &mut ImageOutputView<'_, f32, $layout>,
            values: &[&DeviceMemory<f32>; $channels],
            levels: &[&DeviceMemory<f32>; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors_f32(values, levels)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_packed_f32_in_place {
    ($doc:literal, $name:ident, $layout:ty, $channels:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Updates the image in place.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
        /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, f32, $layout>,
            values: &[&DeviceMemory<f32>; $channels],
            levels: &[&DeviceMemory<f32>; $channels],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            let (mut value_pointers, mut level_pointers, mut level_counts) =
                lookup_table_channel_descriptors_f32(values, levels)?;

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    value_pointers.as_mut_ptr().cast(),
                    level_pointers.as_mut_ptr().cast(),
                    level_counts.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_lookup_table_packed {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and table element ABI and channel layout.
        /// They must uphold the method's table-shape checks, access rules, and completion contracts.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Native element type of both the lookup values and their mapping levels.
            type Value: Copy;

            /// Converts color samples through lookup values and levels without interpolation.
            ///
            /// Integer-table implementations reject stream capture and return after successful stream completion.
            /// Floating-point-table implementations enqueue work on the selected stream.
            /// For AC4 layouts, destination alpha is not processed.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Source and destination image sizes must match.
            /// Participating images must belong to the selected CUDA context.
            /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
            /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image inputs and lookup tables must be initialized and ready when read by native work.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
            /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
            /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
            /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                values: &[&DeviceMemory<Self::Value>; $channels],
                levels: &[&DeviceMemory<Self::Value>; $channels],
            ) -> Result<()>;
        }

        /// Converts color samples through lookup values and levels without interpolation.
        ///
        /// Integer-table implementations reject stream capture and return after successful stream completion.
        /// Floating-point-table implementations enqueue work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Source and destination image sizes must match.
        /// Participating images must belong to the selected CUDA context.
        /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
        /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
        /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            values: &[&DeviceMemory<T::Value>; $channels],
            levels: &[&DeviceMemory<T::Value>; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, values, levels)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Value = $value_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    values: &[&DeviceMemory<Self::Value>; $channels],
                    levels: &[&DeviceMemory<Self::Value>; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, values, levels)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_packed_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty, $value_ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and table element ABI and channel layout.
        /// They must uphold the method's table-shape checks, access rules, and completion contracts.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Native element type of both the lookup values and their mapping levels.
            type Value: Copy;

            /// Converts color samples through lookup values and levels without interpolation.
            /// Updates the image in place.
            ///
            /// Integer-table implementations reject stream capture and return after successful stream completion.
            /// Floating-point-table implementations enqueue work on the selected stream.
            /// For AC4 layouts, destination alpha is not processed.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Participating images must belong to the selected CUDA context.
            /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
            /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image inputs and lookup tables must be initialized and ready when read by native work.
            /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
            /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
            /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
            /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                values: &[&DeviceMemory<Self::Value>; $channels],
                levels: &[&DeviceMemory<Self::Value>; $channels],
            ) -> Result<()>;
        }

        /// Converts color samples through lookup values and levels without interpolation.
        /// Updates the image in place.
        ///
        /// Integer-table implementations reject stream capture and return after successful stream completion.
        /// Floating-point-table implementations enqueue work on the selected stream.
        /// For AC4 layouts, destination alpha is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Each channel must have equal value and level counts, at least two levels, and a level count representable as `i32`.
        /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
        /// All device tables must belong to the selected CUDA context; packed wrappers check image contexts only.
        /// Retain image and device-table allocations through queued completion, including submission errors and captured graph replay where supported.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            values: &[&DeviceMemory<T::Value>; $channels],
            levels: &[&DeviceMemory<T::Value>; $channels],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, values, levels)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                type Value = $value_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    values: &[&DeviceMemory<Self::Value>; $channels],
                    levels: &[&DeviceMemory<Self::Value>; $channels],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, values, levels)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_trilinear {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and table element ABI and channel layout.
        /// They must uphold the method's table-shape checks, access rules, and completion contracts.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
            ///
            /// Rejects stream capture and returns after successful stream completion.
            /// C4 copies source alpha unchanged; AC4 leaves destination alpha unprocessed.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Source and destination image sizes must match.
            /// Images and the device value table must belong to the selected CUDA context.
            /// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
            /// Their checked length product must not exceed `values.len()`.
            /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image inputs and lookup tables must be initialized and ready when read by native work.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
            /// Keep image and table allocations valid until this call completes its native work.
            /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                values: &DeviceMemory<u32>,
                levels: [&[Self]; 3],
            ) -> Result<()>;
        }

        /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
        ///
        /// Rejects stream capture and returns after successful stream completion.
        /// C4 copies source alpha unchanged; AC4 leaves destination alpha unprocessed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Source and destination image sizes must match.
        /// Images and the device value table must belong to the selected CUDA context.
        /// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
        /// Their checked length product must not exceed `values.len()`.
        /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from the source image and lookup inputs during native use.
        /// Keep image and table allocations valid until this call completes its native work.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            values: &DeviceMemory<u32>,
            levels: [&[T]; 3],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, values, levels)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    values: &DeviceMemory<u32>,
                    levels: [&[Self]; 3],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, values, levels)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_trilinear_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and table element ABI and channel layout.
        /// They must uphold the method's table-shape checks, access rules, and completion contracts.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
            /// Updates the image in place.
            ///
            /// Rejects stream capture and returns after successful stream completion.
            /// C4 copies source alpha unchanged; AC4 leaves destination alpha unprocessed.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Images and the device value table must belong to the selected CUDA context.
            /// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
            /// Their checked length product must not exceed `values.len()`.
            /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image inputs and lookup tables must be initialized and ready when read by native work.
            /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
            /// Keep image and table allocations valid until this call completes its native work.
            /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                values: &DeviceMemory<u32>,
                levels: [&[Self]; 3],
            ) -> Result<()>;
        }

        /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
        /// Updates the image in place.
        ///
        /// Rejects stream capture and returns after successful stream completion.
        /// C4 copies source alpha unchanged; AC4 leaves destination alpha unprocessed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Images and the device value table must belong to the selected CUDA context.
        /// Each of the three level arrays must contain at least two entries and have a length representable as `i32`.
        /// Their checked length product must not exceed `values.len()`.
        /// Capture checks and stream synchronization can also fail for implementations that wait for completion.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image inputs and lookup tables must be initialized and ready when read by native work.
        /// Keep the image exclusively writable and disjoint from all lookup storage during native use.
        /// Keep image and table allocations valid until this call completes its native work.
        /// Lookup levels must satisfy the native ordering and input-domain requirements; ordering and domain coverage are not validated.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            values: &DeviceMemory<u32>,
            levels: [&[T]; 3],
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, values, levels)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    values: &DeviceMemory<u32>,
                    levels: [&[Self]; 3],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, values, levels)
                    }
}
            }
        )*
    };
}
