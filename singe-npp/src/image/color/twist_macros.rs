macro_rules! impl_color_twist {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
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
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Each source and destination image pair must have matching sizes.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                twist: ColorTwistMatrix,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, twist)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    twist: ColorTwistMatrix,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, twist)
                    }
}
            }
        )*
    };
}

macro_rules! impl_color_twist_batch {
    ($doc:literal, $name:ident, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest image width and smallest image height in the batch.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        /// Batches must contain at least two images and have a count representable as `i32`.
        /// `twists` must contain one coefficient matrix per image.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Source and destination batch lengths must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, u8, $layout>],
            twists: &[ColorTwistMatrix],
            destinations: &mut [ImageOutputView<'_, u8, $layout>],
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, batch_memory, _twist_memory) = color_twist_batch_descriptors(
                stream_context.cuda_context(),
                sources,
                twists,
                destinations,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_in_place {
    ($doc:literal, $name:ident, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest image width and smallest image height in the batch.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Batches must contain at least two images and have a count representable as `i32`.
        /// `twists` must contain one coefficient matrix per image.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            images: &mut [ImageViewMut<'_, u8, $layout>],
            twists: &[ColorTwistMatrix],
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in images.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_in_place_descriptors(
                    stream_context.cuda_context(),
                    images,
                    twists,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_typed {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest image width and smallest image height in the batch.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        /// Batches must contain at least two images and have a count representable as `i32`.
        /// `twists` must contain one coefficient matrix per image.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Source and destination batch lengths must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, $layout>],
            twists: &[ColorTwistMatrix],
            destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, batch_memory, _twist_memory) = color_twist_batch_descriptors(
                stream_context.cuda_context(),
                sources,
                twists,
                destinations,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_in_place_typed {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest image width and smallest image height in the batch.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Batches must contain at least two images and have a count representable as `i32`.
        /// `twists` must contain one coefficient matrix per image.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            images: &mut [ImageViewMut<'_, $ty, $layout>],
            twists: &[ColorTwistMatrix],
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in images.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_in_place_descriptors(
                    stream_context.cuda_context(),
                    images,
                    twists,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_with_constants_typed {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest image width and smallest image height in the batch.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        /// Batches must contain at least two images and have a count representable as `i32`.
        /// `twists` must contain one coefficient matrix per image.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Source and destination batch lengths must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, C4>],
            twists: &[ColorTwistBatchConstantsMatrix4],
            destinations: &mut [ImageOutputView<'_, $ty, C4>],
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_with_constants_descriptors(
                    stream_context.cuda_context(),
                    sources,
                    twists,
                    destinations,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_with_constants_in_place_typed {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptors and coefficient arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest image width and smallest image height in the batch.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Batches must contain at least two images and have a count representable as `i32`.
        /// `twists` must contain one coefficient matrix per image.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            images: &mut [ImageViewMut<'_, $ty, C4>],
            twists: &[ColorTwistBatchConstantsMatrix4],
            min: f32,
            max: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in images.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_with_constants_in_place_descriptors(
                    stream_context.cuda_context(),
                    images,
                    twists,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_twist_different_layout {
    ($doc:literal, $name:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
            twist: ColorTwistMatrix,
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
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                twist: ColorTwistMatrix,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, twist)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    twist: ColorTwistMatrix,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, twist)
                    }
}
            }
        )*
    };
}

macro_rules! impl_color_twist_with_constants {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C4>,
            destination: &mut ImageOutputView<'_, $ty, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
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
                    twist.as_ptr().cast(),
                    constants.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist_with_constants {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Each source and destination image pair must have matching sizes.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C4>,
                destination: &mut ImageOutputView<'_, Self, C4>,
                twist: ColorTwistMatrix4,
                constants: ColorTwistConstants4,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C4>,
            destination: &mut ImageOutputView<'_, T, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, twist, constants)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C4>,
                    destination: &mut ImageOutputView<'_, Self, C4>,
                    twist: ColorTwistMatrix4,
                    constants: ColorTwistConstants4,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, twist, constants)
                    }
}
            }
        )*
    };
}

macro_rules! impl_color_twist_with_constants_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    twist.as_ptr().cast(),
                    constants.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist_with_constants_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C4>,
                twist: ColorTwistMatrix4,
                constants: ColorTwistConstants4,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image, twist, constants)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C4>,
                    twist: ColorTwistMatrix4,
                    constants: ColorTwistConstants4,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image, twist, constants)
                    }
}
            }
        )*
    };
}

macro_rules! impl_color_twist_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source.planes()[0].step(),
                    destination_planes.as_ptr(),
                    destination.planes()[0].step(),
                    source.planes()[0].size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_planar_in_place {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut PlanarImageViewMut<'_, $ty, $planes>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for plane in image.planes() {
                stream_context.validate_region(plane)?;
            }

            let step = image.planes()[0].step();
            let size = image.planes()[0].size();
            let planes = image
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    planes.as_ptr().cast(),
                    step,
                    size.into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_gamma_packed {
    ($doc:literal, $name:ident, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, $layout>,
            destination: &mut ImageOutputView<'_, u8, $layout>,
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

macro_rules! impl_generic_gamma {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Each source and destination image pair must have matching sizes.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
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

macro_rules! impl_gamma_packed_in_place {
    ($doc:literal, $name:ident, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, u8, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_gamma_in_place {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, image)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, image)
                    }
}
            }
        )*
    };
}

macro_rules! impl_gamma_planar {
    ($doc:literal, $name:ident, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Each source and destination image pair must have matching sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, u8, 3>,
            destination: &mut PlanarImageOutputView<'_, u8, 3>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source.planes()[0].step(),
                    destination_planes.as_mut_ptr(),
                    destination.planes()[0].step(),
                    source.planes()[0].size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_gamma_planar_in_place {
    ($doc:literal, $name:ident, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image samples must be initialized and ready when read by the selected stream, and images must remain exclusively writable and mutually disjoint during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut PlanarImageViewMut<'_, u8, 3>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for plane in image.planes() {
                stream_context.validate_region(plane)?;
            }

            let step = image.planes()[0].step();
            let size = image.planes()[0].size();
            let planes = image
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    planes.as_ptr().cast(),
                    step,
                    size.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
