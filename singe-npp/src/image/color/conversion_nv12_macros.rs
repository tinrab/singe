macro_rules! impl_nv12_color_twist_planar_forward {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work on the selected stream without waiting for completion.
        /// Chroma dimensions count positions; each interleaved chroma position stores two components.
        /// Row padding remains unwritten.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Destination plane strides are passed independently.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The full-resolution source and destination dimensions must match.
        /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
        /// The three full-resolution source color planes must have equal dimensions and byte strides.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination_y: &mut ImageOutputView<'_, $ty, C1>,
            destination_uv: &mut ImageOutputView<'_, $ty, C2>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_0)?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_uv)?;

            let roi = validate_rgb_source_planes(source_0, source_1, source_2)?;
            validate_nv12_destination_planes(roi, destination_y, destination_uv)?;

            let source_planes = [
                source_0.as_ptr().cast(),
                source_1.as_ptr().cast(),
                source_2.as_ptr(),
            ];
            let mut source_steps = [source_0.step(), source_1.step(), source_2.step()];
            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_uv.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [destination_y.step(), destination_uv.step()];

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

macro_rules! impl_nv12_color_twist_packed_forward {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work on the selected stream without waiting for completion.
        /// Chroma dimensions count positions; each interleaved chroma position stores two components.
        /// Row padding remains unwritten.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        /// Destination plane strides are passed independently.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The full-resolution source and destination dimensions must match.
        /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination_y: &mut ImageOutputView<'_, $ty, C1>,
            destination_uv: &mut ImageOutputView<'_, $ty, C2>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_uv)?;

            validate_nv12_destination_planes(source.size(), destination_y, destination_uv)?;

            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_uv.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [destination_y.step(), destination_uv.step()];

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

macro_rules! impl_nv12_color_twist_packed_inverse {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work on the selected stream without waiting for completion.
        /// Chroma dimensions count positions; each interleaved chroma position stores two components.
        /// Row padding remains unwritten.
        /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The full-resolution source and destination dimensions must match.
        /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
        /// The source luma and interleaved-chroma planes must have equal byte strides.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_uv: &ImageView<'_, $ty, C2>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_uv)?;
            stream_context.validate_region(destination)?;

            let roi = validate_nv12_source_planes(source_y, source_uv)?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [source_y.as_ptr().cast(), source_uv.as_ptr().cast()];
            let mut source_steps = [source_y.step(), source_uv.step()];

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

macro_rules! impl_nv12_packed_inverse {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work on the selected stream without waiting for completion.
        /// Chroma dimensions count positions; each interleaved chroma position stores two components.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The full-resolution source and destination dimensions must match.
        /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
        /// The source luma and interleaved-chroma planes must have equal byte strides.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_uv: &ImageView<'_, $ty, C2>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_uv)?;
            stream_context.validate_region(destination)?;

            let roi = validate_nv12_source_planes(source_y, source_uv)?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [source_y.as_ptr().cast(), source_uv.as_ptr().cast()];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_y.step(),
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

macro_rules! impl_nv12_to_planar {
    ($doc:literal, $name:ident, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work on the selected stream without waiting for completion.
        /// Chroma dimensions count positions; each interleaved chroma position stores two components.
        /// Row padding remains unwritten.
        /// The three destination strides are passed independently.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
        /// The full-resolution source and destination dimensions must match.
        /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
        /// The source luma and interleaved-chroma planes must have equal byte strides.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Plane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_uv: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageOutputView<'_, u8, C1>,
            destination_u: &mut ImageOutputView<'_, u8, C1>,
            destination_v: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_y)?;
            stream_context.validate_region(source_uv)?;
            stream_context.validate_region(destination_y)?;
            stream_context.validate_region(destination_u)?;
            stream_context.validate_region(destination_v)?;

            let roi = validate_nv12_source_planes(source_y, source_uv)?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_u,
                destination_v,
                2,
                2,
            )?;

            let source_planes = [source_y.as_ptr().cast(), source_uv.as_ptr().cast()];
            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_u.as_mut_ptr().cast(),
                destination_v.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_u.step(),
                destination_v.step(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_y.step(),
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
