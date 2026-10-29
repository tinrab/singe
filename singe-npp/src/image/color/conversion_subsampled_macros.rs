macro_rules! impl_subsampled_p2_to_p3 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Source chroma width and height must equal the full-resolution dimensions divided by ", stringify!($source_horizontal_subsampling), " and ", stringify!($source_vertical_subsampling), ", respectively.")]
        /// Source full-resolution dimensions must be divisible by those subsampling factors.
        #[doc = concat!("Destination chroma width and height must equal the full-resolution dimensions divided by ", stringify!($destination_horizontal_subsampling), " and ", stringify!($destination_vertical_subsampling), ", respectively.")]
        /// Destination full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cbcr: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cb: &mut ImageOutputView<'_, u8, C1>,
            destination_cr: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_cbcr)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cb)?;
            stream_context.validate_region(destination_cr)?;

            let roi = validate_subsampled_p2_source_planes(
                source_y,
                source_cbcr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_cb.as_mut_ptr().cast(),
                destination_cr.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_cb.step(),
                destination_cr.step(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_y.as_ptr().cast(),
                    source_y.step(),
                    source_cbcr.as_ptr().cast(),
                    source_cbcr.step(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_c2_to_p3 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Destination chroma width and height must equal the full-resolution dimensions divided by ", stringify!($destination_horizontal_subsampling), " and ", stringify!($destination_vertical_subsampling), ", respectively.")]
        /// Destination full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cb: &mut ImageOutputView<'_, u8, C1>,
            destination_cr: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cb)?;
            stream_context.validate_region(destination_cr)?;

            validate_jpeg_subsampled_destination_planes(
                source.size(),
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_cb.as_mut_ptr().cast(),
                destination_cr.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_cb.step(),
                destination_cr.step(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_c2_to_p2 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Destination chroma width and height must equal the full-resolution dimensions divided by ", stringify!($destination_horizontal_subsampling), " and ", stringify!($destination_vertical_subsampling), ", respectively.")]
        /// Destination full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cbcr: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cbcr)?;

            validate_subsampled_p2_destination_planes(
                source.size(),
                destination_y,
                destination_cbcr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_y.as_mut_ptr().cast(),
                    destination_y.step(),
                    destination_cbcr.as_mut_ptr().cast(),
                    destination_cbcr.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_p3_to_p2 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Source chroma width and height must equal the full-resolution dimensions divided by ", stringify!($source_horizontal_subsampling), " and ", stringify!($source_vertical_subsampling), ", respectively.")]
        /// Source full-resolution dimensions must be divisible by those subsampling factors.
        #[doc = concat!("Destination chroma width and height must equal the full-resolution dimensions divided by ", stringify!($destination_horizontal_subsampling), " and ", stringify!($destination_vertical_subsampling), ", respectively.")]
        /// Destination full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cbcr: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_cb)?;
            stream_context.validate_region(source_cr)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cbcr)?;

            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_subsampled_p2_destination_planes(
                roi,
                destination_y,
                destination_cbcr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source_steps.as_mut_ptr().cast(),
                    destination_y.as_mut_ptr().cast(),
                    destination_y.step(),
                    destination_cbcr.as_mut_ptr().cast(),
                    destination_cbcr.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_p3_to_c2 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Source chroma width and height must equal the full-resolution dimensions divided by ", stringify!($source_horizontal_subsampling), " and ", stringify!($source_vertical_subsampling), ", respectively.")]
        /// Source full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_cb)?;
            stream_context.validate_region(source_cr)?;
            stream_context.validate_region(destination)?;

            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source_steps.as_mut_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_p3_to_p3 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Source chroma width and height must equal the full-resolution dimensions divided by ", stringify!($source_horizontal_subsampling), " and ", stringify!($source_vertical_subsampling), ", respectively.")]
        /// Source full-resolution dimensions must be divisible by those subsampling factors.
        #[doc = concat!("Destination chroma width and height must equal the full-resolution dimensions divided by ", stringify!($destination_horizontal_subsampling), " and ", stringify!($destination_vertical_subsampling), ", respectively.")]
        /// Destination full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cb: &mut ImageOutputView<'_, u8, C1>,
            destination_cr: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_cb)?;
            stream_context.validate_region(source_cr)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cb)?;
            stream_context.validate_region(destination_cr)?;

            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];
            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_cb.as_mut_ptr().cast(),
                destination_cr.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_cb.step(),
                destination_cr.step(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source_steps.as_mut_ptr().cast(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_p2_to_c2 {
    (
        $doc:literal,
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Source chroma width and height must equal the full-resolution dimensions divided by ", stringify!($source_horizontal_subsampling), " and ", stringify!($source_vertical_subsampling), ", respectively.")]
        /// Source full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cbcr: &ImageView<'_, u8, C2>,
            destination: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_cbcr)?;
            stream_context.validate_region(destination)?;

            let roi = validate_subsampled_p2_source_planes(
                source_y,
                source_cbcr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_same_size(roi, destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_y.as_ptr().cast(),
                    source_y.step(),
                    source_cbcr.as_ptr().cast(),
                    source_cbcr.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_packed_to_p3 {
    (
        $doc:literal,
        $name:ident,
        $layout:ty,
        $ffi:ident,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Chroma dimensions count chroma positions; interleaved chroma has two components per position.
        /// Each plane's byte stride is passed independently.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Destination chroma width and height must equal the full-resolution dimensions divided by ", stringify!($destination_horizontal_subsampling), " and ", stringify!($destination_vertical_subsampling), ", respectively.")]
        /// Destination full-resolution dimensions must be divisible by those subsampling factors.
        /// Separate Cb and Cr planes must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, $layout>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cb: &mut ImageOutputView<'_, u8, C1>,
            destination_cr: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cb)?;
            stream_context.validate_region(destination_cr)?;

            validate_jpeg_subsampled_destination_planes(
                source.size(),
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_cb.as_mut_ptr().cast(),
                destination_cr.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_cb.step(),
                destination_cr.step(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
