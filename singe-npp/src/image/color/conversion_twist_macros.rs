macro_rules! impl_subsampled_color_twist_planar_forward {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Chroma plane byte strides are passed independently.
        /// Row padding remains unwritten.
        ///
        #[doc = concat!("Chroma width and height divide the full-resolution dimensions by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.
        /// Full-resolution source and destination dimensions must match.
        /// Chroma planes must have matching dimensions and the exact size for the selected subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        /// The three full-resolution source color planes must have equal dimensions and byte strides.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready for native execution.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain allocations through queued completion, including submission errors and every captured replay.
        /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination_y: &mut ImageOutputView<'_, $ty, C1>,
            destination_cb: &mut ImageOutputView<'_, $ty, C1>,
            destination_cr: &mut ImageOutputView<'_, $ty, C1>,
            twist: ColorTwistMatrix,
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
            let mut source_steps = [source_0.step(), source_1.step(), source_2.step()];
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
                    source_steps.as_mut_ptr().cast(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    roi.into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_twist_packed_forward {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Chroma plane byte strides are passed independently.
        /// Row padding remains unwritten.
        ///
        #[doc = concat!("Chroma width and height divide the full-resolution dimensions by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.
        /// Full-resolution source and destination dimensions must match.
        /// Chroma planes must have matching dimensions and the exact size for the selected subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready for native execution.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain allocations through queued completion, including submission errors and every captured replay.
        /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination_y: &mut ImageOutputView<'_, $ty, C1>,
            destination_cb: &mut ImageOutputView<'_, $ty, C1>,
            destination_cr: &mut ImageOutputView<'_, $ty, C1>,
            twist: ColorTwistMatrix,
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
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_twist_planar_inverse {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Chroma plane byte strides are passed independently.
        /// Row padding remains unwritten.
        ///
        #[doc = concat!("Chroma width and height divide the full-resolution dimensions by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.
        /// Full-resolution source and destination dimensions must match.
        /// Chroma planes must have matching dimensions and the exact size for the selected subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        /// The three full-resolution destination color planes must have equal dimensions and byte strides.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready for native execution.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain allocations through queued completion, including submission errors and every captured replay.
        /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_cb: &ImageView<'_, $ty, C1>,
            source_cr: &ImageView<'_, $ty, C1>,
            destination_0: &mut ImageOutputView<'_, $ty, C1>,
            destination_1: &mut ImageOutputView<'_, $ty, C1>,
            destination_2: &mut ImageOutputView<'_, $ty, C1>,
            twist: ColorTwistMatrix,
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
            let mut destination_steps = [
                destination_0.step(),
                destination_1.step(),
                destination_2.step(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    roi.into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_twist_packed_inverse {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Chroma plane byte strides are passed independently.
        /// Row padding remains unwritten.
        ///
        #[doc = concat!("Chroma width and height divide the full-resolution dimensions by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.
        /// Full-resolution source and destination dimensions must match.
        /// Chroma planes must have matching dimensions and the exact size for the selected subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready for native execution.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain allocations through queued completion, including submission errors and every captured replay.
        /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_cb: &ImageView<'_, $ty, C1>,
            source_cr: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
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
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_twist_packed_inverse_constant_alpha {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Chroma plane byte strides are passed independently.
        /// Row padding remains unwritten.
        ///
        #[doc = concat!("Chroma width and height divide the full-resolution dimensions by ", stringify!($horizontal_subsampling), " and ", stringify!($vertical_subsampling), ", respectively.")]
        /// Writes `alpha` to the fourth destination channel.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.
        /// Full-resolution source and destination dimensions must match.
        /// Chroma planes must have matching dimensions and the exact size for the selected subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready for native execution.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain allocations through queued completion, including submission errors and every captured replay.
        /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_cb: &ImageView<'_, $ty, C1>,
            source_cr: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, AC4>,
            twist: ColorTwistMatrix,
            alpha: $ty,
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
                    twist.as_ptr().cast(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
