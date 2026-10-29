macro_rules! impl_mirror {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination sizes must match.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            axis: Axis,
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
                    axis.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mirror_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels and channels not processed by the native routine retain their existing values.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Images must contain initialized values ready for the selected stream and remain exclusively writable during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            axis: Axis,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(image)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    axis.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mirror_batch {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest width and smallest height in the batch.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Batches require at least two entries and a count representable by the native interface.
        /// Source and destination batch lengths must match.
        /// Each source/destination pair must have matching image sizes.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            axis: Axis,
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

            let (batch_size, roi, descriptors) =
                mirror_batch_descriptors(stream_context.cuda_context(), sources, destinations)?;
            let batch_size =
                i32::try_from(batch_size).map_err(|_| Error::out_of_range("batch size"))?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    roi.into(),
                    axis.into(),
                    descriptors.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_mirror_batch_in_place {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// The processed ROI uses the smallest width and smallest height in the batch.
        /// Pixels and channels not processed by the native routine retain their existing values.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Batches require at least two entries and a count representable by the native interface.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Images must contain initialized values ready for the selected stream and remain exclusively writable during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            axis: Axis,
            images: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in images.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, descriptors) =
                mirror_batch_in_place_descriptors(stream_context.cuda_context(), images)?;
            let batch_size =
                i32::try_from(batch_size).map_err(|_| Error::out_of_range("batch size"))?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    roi.into(),
                    axis.into(),
                    descriptors.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_resize_sqr_pixel {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            resize: &ResizeSqrPixel,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            resize.source_roi.validate()?;
            resize.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    resize.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    resize.destination_roi.into(),
                    resize.x_factor,
                    resize.y_factor,
                    resize.x_shift,
                    resize.y_shift,
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_sqr_pixel_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            resize: &ResizeSqrPixel,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            resize.source_roi.validate()?;
            for plane in destination.planes() {
                resize.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    resize.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    resize.destination_roi.into(),
                    resize.x_factor,
                    resize.y_factor,
                    resize.x_shift,
                    resize.y_shift,
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            resize: &Resize,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            resize.source_roi.validate()?;
            resize.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    resize.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    resize.destination_roi.into(),
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            resize: &Resize,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            resize.source_roi.validate()?;
            for plane in destination.planes() {
                resize.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].step(),
                    source.planes()[0].size().into(),
                    resize.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    destination.planes()[0].size().into(),
                    resize.destination_roi.into(),
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_batch {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Native source and destination bounds use the smallest image width and height in each batch.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Batches require at least two entries and a count representable by the native interface.
        /// Source and destination batch lengths must match.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            resize: &Resize,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageOutputView<'_, $ty, $layout>],
        ) -> Result<()> {
            resize.source_roi.validate()?;
            for image in destinations.iter() {
                resize.destination_roi.validate_within(image.size())?;
            }

            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, smallest_source_size, smallest_destination_size, descriptors) =
                resize_batch_descriptors(stream_context.cuda_context(), sources, destinations)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    smallest_source_size.into(),
                    resize.source_roi.into(),
                    smallest_destination_size.into(),
                    resize.destination_roi.into(),
                    i32::from(resize.interpolation),
                    descriptors.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_resize_batch_advanced {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Batches require at least two entries and a count representable by the native interface.
        /// Source and destination batch lengths must match.
        /// `rois` must contain one ROI pair per image pair.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            rois: &[ResizeBatchAdvanced],
            interpolation: InterpolationMode,
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

            let (
                batch_size,
                max_destination_width,
                max_destination_height,
                sources,
                destinations,
                rois,
            ) = resize_batch_advanced_descriptors(
                stream_context.cuda_context(),
                rois,
                sources,
                destinations,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    max_destination_width,
                    max_destination_height,
                    sources.as_mut_ptr().cast(),
                    destinations.as_mut_ptr().cast(),
                    rois.as_mut_ptr().cast(),
                    batch_size,
                    i32::from(interpolation),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_resize_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element and layout ABI and preserve
        /// the dispatch method's access, ROI, status, and queued-lifetime contracts.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or a participating image belongs to another CUDA context.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Each destination ROI must lie within its destination image.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                resize: &Resize,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            resize: &Resize,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, resize, source, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    resize: &Resize,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, resize, source, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_resize_sqr_pixel_operation {
    (
        $doc:literal,
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native type and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must match the native element and layout ABI and preserve
        /// the dispatch method's access, ROI, status, and queued-lifetime contracts.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or a participating image belongs to another CUDA context.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Each destination ROI must lie within its destination image.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when the selected stream reads them.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                resize: &ResizeSqrPixel,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            resize: &ResizeSqrPixel,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, resize, source, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    resize: &ResizeSqrPixel,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, resize, source, destination)
                    }
}
            }
        )*
    };
}

macro_rules! impl_rotate {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            rotate: &Rotate,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            rotate.source_roi.validate()?;
            rotate.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    rotate.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    rotate.destination_roi.into(),
                    rotate.angle,
                    rotate.x_shift,
                    rotate.y_shift,
                    i32::from(rotate.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_remap {
    ($doc:literal, $name:ident, $ty:ty, $map_ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Both coordinate maps must have the destination image size.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any coordinate maps must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image and coordinate-map allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, map coordinates, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            remap: &Remap,
            source: &ImageView<'_, $ty, $layout>,
            x_map: &ImageView<'_, $map_ty, C1>,
            y_map: &ImageView<'_, $map_ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            remap.source_roi.validate()?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(x_map)?;
            stream_context.validate_region(y_map)?;
            stream_context.validate_region(destination)?;

            validate_same_size(x_map.size(), destination.size())?;
            validate_same_size(y_map.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    remap.source_roi.into(),
                    x_map.as_ptr().cast(),
                    x_map.step(),
                    y_map.as_ptr().cast(),
                    y_map.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    i32::from(remap.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_remap_planar {
    ($doc:literal, $name:ident, $ty:ty, $map_ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Both coordinate maps must have the destination image size.
        /// Source and destination plane sizes must match.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels and any coordinate maps must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image and coordinate-map allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, map coordinates, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            remap: &Remap,
            source: &PlanarImageView<'_, $ty, $planes>,
            x_map: &ImageView<'_, $map_ty, C1>,
            y_map: &ImageView<'_, $map_ty, C1>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            remap.source_roi.validate()?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(x_map)?;
            stream_context.validate_region(y_map)?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(x_map.size(), destination.planes()[0].size())?;
            validate_same_size(y_map.size(), destination.planes()[0].size())?;
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    remap.source_roi.into(),
                    x_map.as_ptr().cast(),
                    x_map.step(),
                    y_map.as_ptr().cast(),
                    y_map.step(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    destination.planes()[0].size().into(),
                    i32::from(remap.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            warp.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_batch {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Native source bounds use the smallest source width and height in the batch.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Batches require at least two entries and a count representable by the native interface.
        /// Source and destination batch lengths must match.
        /// `coefficients` must contain one matrix per image pair.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffineBatch,
            coefficients: &[AffineCoefficients],
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageOutputView<'_, $ty, $layout>],
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for image in destinations.iter() {
                warp.destination_roi.validate_within(image.size())?;
            }

            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, smallest_source_size, batch_list, _coefficients) =
                warp_affine_batch_descriptors(
                    stream_context.cuda_context(),
                    coefficients,
                    sources,
                    destinations,
                )?;

            {
                // SAFETY: `nppiWarpAffineBatchInit_Ctx` receives live handles owned by the wrappers and valid output locations.
                stream_context.sync_scope(|| unsafe {
                    try_ffi!(sys::nppiWarpAffineBatchInit_Ctx(
                        batch_list.as_mut_ptr().cast(),
                        batch_size,
                        stream_context.as_raw(),
                    ))
                })?;
                // SAFETY: the caller upholds this function's safety contract.
                stream_context.sync_scope(|| unsafe {
                    try_ffi!(sys::$ffi(
                        smallest_source_size.into(),
                        warp.source_roi.into(),
                        warp.destination_roi.into(),
                        i32::from(warp.interpolation),
                        batch_list.as_mut_ptr().cast(),
                        batch_size,
                        stream_context.as_raw(),
                    ))
                })?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for plane in destination.planes() {
                warp.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_quad {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            warp.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_quad_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for plane in destination.planes() {
                warp.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            warp.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_batch {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU descriptor arrays, rejects stream capture, and returns after successful stream completion.
        /// Native source bounds use the smallest source width and height in the batch.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Batches require at least two entries and a count representable by the native interface.
        /// Source and destination batch lengths must match.
        /// `coefficients` must contain one matrix per image pair.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspectiveBatch,
            coefficients: &[PerspectiveCoefficients],
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageOutputView<'_, $ty, $layout>],
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for image in destinations.iter() {
                warp.destination_roi.validate_within(image.size())?;
            }

            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, smallest_source_size, batch_list, _coefficients) =
                warp_perspective_batch_descriptors(
                    stream_context.cuda_context(),
                    coefficients,
                    sources,
                    destinations,
                )?;

            {
                // SAFETY: `nppiWarpPerspectiveBatchInit_Ctx` receives live handles owned by the wrappers and valid output locations.
                stream_context.sync_scope(|| unsafe {
                    try_ffi!(sys::nppiWarpPerspectiveBatchInit_Ctx(
                        batch_list.as_mut_ptr().cast(),
                        batch_size,
                        stream_context.as_raw(),
                    ))
                })?;
                // SAFETY: the caller upholds this function's safety contract.
                stream_context.sync_scope(|| unsafe {
                    try_ffi!(sys::$ffi(
                        smallest_source_size.into(),
                        warp.source_roi.into(),
                        warp.destination_roi.into(),
                        i32::from(warp.interpolation),
                        batch_list.as_mut_ptr().cast(),
                        batch_size,
                        stream_context.as_raw(),
                    ))
                })?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for plane in destination.planes() {
                warp.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_quad {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            warp.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_quad_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for plane in destination.planes() {
                warp.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_back {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            warp.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_back_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for plane in destination.planes() {
                warp.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_back {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            warp.destination_roi.validate_within(destination.size())?;

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_back_planar {
    ($doc:literal, $name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Pixels outside the transformed source intersection, ignored channels, and padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source ROI dimensions must be positive and their endpoints representable.
        /// Each destination ROI must lie within its destination image.
        /// Source and destination plane sizes must match.
        /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when the selected stream reads them.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, byte-stride, and pointer-array requirements.
        /// Pixel layouts, byte strides, ROI footprints, and address alignment must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            warp.source_roi.validate()?;
            for plane in destination.planes() {
                warp.destination_roi.validate_within(plane.size())?;
            }

            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(|plane| plane.as_mut_ptr());

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
