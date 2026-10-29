macro_rules! impl_jpeg_subsampled_planar_forward {
    ($doc:literal, $name:ident, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Chroma width and height must equal the full-resolution dimensions divided by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
        /// The three full-resolution source color planes must have equal byte strides.
        /// Chroma plane strides are passed independently.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Color-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, u8, C1>,
            source_1: &ImageView<'_, u8, C1>,
            source_2: &ImageView<'_, u8, C1>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_cb: &mut ImageOutputView<'_, u8, C1>,
            destination_cr: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_0)?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_cb)?;
            stream_context.validate_region(destination_cr)?;

            let roi = validate_rgb_source_planes(source_0, source_1, source_2)?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_cb,
                destination_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;

            let source_planes = [
                source_0.as_ptr().cast(),
                source_1.as_ptr().cast(),
                source_2.as_ptr(),
            ];
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
                    source_planes.as_ptr(),
                    source_0.step(),
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

macro_rules! impl_jpeg_subsampled_packed_forward {
    ($doc:literal, $name:ident, $layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Chroma width and height must equal the full-resolution dimensions divided by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
        /// Chroma plane strides are passed independently.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Color-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
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
                $horizontal_subsampling,
                $vertical_subsampling,
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

macro_rules! impl_jpeg_subsampled_planar_inverse {
    ($doc:literal, $name:ident, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Chroma width and height must equal the full-resolution dimensions divided by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
        /// The three full-resolution destination color planes must have equal byte strides.
        /// Chroma plane strides are passed independently.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Color-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination_0: &mut ImageOutputView<'_, u8, C1>,
            destination_1: &mut ImageOutputView<'_, u8, C1>,
            destination_2: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_cb)?;
            stream_context.validate_region(source_cr)?;
            stream_context.validate_region(destination_0)?;
            stream_context.validate_region(destination_1)?;
            stream_context.validate_region(destination_2)?;

            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;
            validate_rgb_destination_planes(roi, destination_0, destination_1, destination_2)?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];
            let mut destination_planes = [
                destination_0.as_mut_ptr().cast(),
                destination_1.as_mut_ptr().cast(),
                destination_2.as_mut_ptr().cast(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination_planes.as_mut_ptr(),
                    destination_0.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_jpeg_subsampled_packed_inverse {
    ($doc:literal, $name:ident, $layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 destination layouts, the alpha channel is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Chroma width and height must equal the full-resolution dimensions divided by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
        /// Chroma plane strides are passed independently.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Color-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination: &mut ImageOutputView<'_, u8, $layout>,
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
                $horizontal_subsampling,
                $vertical_subsampling,
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
                    source_planes.as_ptr(),
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

macro_rules! impl_jpeg_subsampled_packed_inverse_constant_alpha {
    ($doc:literal, $name:ident, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Writes `alpha` to every destination alpha component.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// Source and destination full-resolution dimensions must match.
        #[doc = concat!("Chroma width and height must equal the full-resolution dimensions divided by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        /// Full-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.
        /// Chroma plane strides are passed independently.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Color-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination: &mut ImageOutputView<'_, u8, C4>,
            alpha: u8,
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
                $horizontal_subsampling,
                $vertical_subsampling,
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
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
