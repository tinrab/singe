macro_rules! impl_color_convert_same_layout {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
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

macro_rules! impl_generic_color_convert_same_layout {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible color-conversion routine.
        /// They must uphold the method's size checks, component order, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
            /// Source and destination image sizes must match.
            /// Returns an error if NPP rejects the image geometry or conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
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

macro_rules! impl_color_convert_different_layout {
    ($doc:literal, $name:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
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

macro_rules! impl_color_convert_different_layout_constant_alpha {
    ($doc:literal, $name:ident, $ty:ty, $source_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, C4>,
            alpha: u8,
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
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_same_layout {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Converts the common ROI defined by the smallest source width and smallest source height in the batch.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Capture checks, descriptor allocation or upload, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageOutputView<'_, $ty, $layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, source_descriptors, destination_descriptors) =
                color_batch_descriptors(stream_context.cuda_context(), sources, destinations)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    roi.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_same_layout_advanced {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Converts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Capture checks, descriptor allocation or upload, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageOutputView<'_, $ty, $layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, max_roi, source_descriptors, destination_descriptors) =
                color_batch_advanced_descriptors(
                    stream_context.cuda_context(),
                    sources,
                    destinations,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_planar_to_packed {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $packed_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Converts the common ROI defined by the smallest source width and smallest source height in the batch.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a destination belongs to another CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Capture checks, descriptor allocation or upload, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Every source plane must belong to the selected CUDA context; this wrapper checks destination contexts only.
        /// Keep image allocations valid until this call completes its native work.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, $ty, $planes>],
            destinations: &mut [ImageOutputView<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, source_batch_list, destination_descriptors, _source_descriptors) =
                color_planar_batch_descriptors(
                    stream_context.cuda_context(),
                    sources,
                    destinations,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    roi.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_planar_to_packed_advanced {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $packed_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Converts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a destination belongs to another CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Capture checks, descriptor allocation or upload, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Every source plane must belong to the selected CUDA context; this wrapper checks destination contexts only.
        /// Keep image allocations valid until this call completes its native work.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, $ty, $planes>],
            destinations: &mut [ImageOutputView<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (
                batch_size,
                max_roi,
                source_batch_list,
                destination_descriptors,
                _source_descriptors,
            ) = color_planar_batch_advanced_descriptors(
                stream_context.cuda_context(),
                sources,
                destinations,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_convert_batch_planar_to_packed {
    ($doc:literal, $name:ident, $ty:ty, $packed_layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Converts the common ROI defined by the smallest source width and smallest source height in the batch.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// All three source-plane batches must have equal lengths.
        /// Both chroma planes must have equal sizes matching the full-resolution plane divided by the horizontal and vertical subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        /// Capture checks, descriptor allocation or upload, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, $ty, C1>],
            source_plane_1: &[ImageView<'_, $ty, C1>],
            source_plane_2: &[ImageView<'_, $ty, C1>],
            destinations: &mut [ImageOutputView<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in source_plane_0.iter() {
                stream_context.validate_region(image)?;
            }
            for image in source_plane_1.iter() {
                stream_context.validate_region(image)?;
            }
            for image in source_plane_2.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, source_batch_list, destination_descriptors, _source_descriptors) =
                subsampled_color_batch_descriptors(
                    stream_context.cuda_context(),
                    source_plane_0,
                    source_plane_1,
                    source_plane_2,
                    destinations,
                    $horizontal_subsampling,
                    $vertical_subsampling,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    roi.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_convert_batch_planar_to_packed_advanced {
    ($doc:literal, $name:ident, $ty:ty, $packed_layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Converts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// All three source-plane batches must have equal lengths.
        /// Both chroma planes must have equal sizes matching the full-resolution plane divided by the horizontal and vertical subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        /// Capture checks, descriptor allocation or upload, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, $ty, C1>],
            source_plane_1: &[ImageView<'_, $ty, C1>],
            source_plane_2: &[ImageView<'_, $ty, C1>],
            destinations: &mut [ImageOutputView<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in source_plane_0.iter() {
                stream_context.validate_region(image)?;
            }
            for image in source_plane_1.iter() {
                stream_context.validate_region(image)?;
            }
            for image in source_plane_2.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (
                batch_size,
                max_roi,
                source_batch_list,
                destination_descriptors,
                _source_descriptors,
            ) = subsampled_color_batch_advanced_descriptors(
                stream_context.cuda_context(),
                source_plane_0,
                source_plane_1,
                source_plane_2,
                destinations,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_color_convert_packed_to_planar {
    ($doc:literal, $name:ident, $ty:ty, $packed_layout:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $packed_layout>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.size(), destination.planes()[0].size())?;

            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_mut_ptr(),
                    destination.planes()[0].step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_planar_to_packed {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $packed_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageOutputView<'_, $ty, $packed_layout>,
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
                    source_planes.as_ptr().cast_mut(),
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

macro_rules! impl_color_convert_planar_to_packed_constant_alpha {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageOutputView<'_, $ty, C4>,
            alpha: u8,
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
                    source_planes.as_ptr().cast_mut(),
                    source.planes()[0].step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.planes()[0].size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_planar_to_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
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
                    source_planes.as_ptr().cast_mut(),
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

macro_rules! impl_color_convert_planar_to_planar_different_planes {
    ($doc:literal, $name:ident, $ty:ty, $source_planes:literal, $destination_planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $source_planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $destination_planes>,
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
                    source_planes.as_ptr().cast_mut(),
                    source.planes()[0].step(),
                    destination_planes.as_ptr().cast_mut(),
                    destination.planes()[0].step(),
                    source.planes()[0].size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
