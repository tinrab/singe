macro_rules! impl_rgb_to_uyvp_packed {
    ($doc:literal, $name:ident, $source_ty:ty, $source_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// `source_offset` must be `(0, 0)`.
        /// `color_space` selects the native color-conversion standard.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.
        /// The video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.
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
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut VideoOutputView<'_, u8>,
            color_space: ColorSpace,
        ) -> Result<()> {
            if destination.format() != VideoFormat::Uyvp10
                || source_offset.x != 0
                || source_offset.y != 0
            {
                return Err(Error::out_of_range("uyvp format or source offset"));
            }
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination.plane_mut(0)?.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_rgb_to_uyvp_planar {
    ($doc:literal, $name:ident, $source_ty:ty, $source_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// `source_offset` must be `(0, 0)`.
        /// `color_space` selects the native color-conversion standard.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.
        /// `destination_step` must be positive.
        /// Raw destination pointers are not checked for context, allocation bounds, or overlap.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready for native execution.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from inputs and other destinations during native use.
        /// Retain allocations through queued completion, including submission errors and every captured replay.
        /// Native pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.
        /// Each raw destination pointer must address native planar UYVP storage with the supplied pitch for the source ROI in the stream context.
        /// The caller must establish access ordering for these untracked destinations.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination_0: *mut u8,
            destination_1: *mut u8,
            destination_2: *mut u8,
            destination_step: i32,
            color_space: ColorSpace,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;

            if source_offset.x != 0 || source_offset.y != 0 || destination_step <= 0 {
                return Err(Error::out_of_range("uyvp planar offset or pitch"));
            }

            let mut destination_planes = [
                destination_0.cast(),
                destination_1.cast(),
                destination_2.cast(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination_planes.as_mut_ptr(),
                    destination_step,
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_rgb_planar_to_uyvp_packed {
    ($doc:literal, $name:ident, $source_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// `source_offset` must be `(0, 0)`.
        /// `color_space` selects the native color-conversion standard.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.
        /// The video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.
        /// The three RGB source planes must have equal dimensions and byte strides.
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
            source_0: &ImageView<'_, $source_ty, C1>,
            source_1: &ImageView<'_, $source_ty, C1>,
            source_2: &ImageView<'_, $source_ty, C1>,
            source_offset: Point,
            destination: &mut VideoOutputView<'_, u8>,
            color_space: ColorSpace,
        ) -> Result<()> {
            if destination.format() != VideoFormat::Uyvp10
                || source_offset.x != 0
                || source_offset.y != 0
            {
                return Err(Error::out_of_range("uyvp format or source offset"));
            }
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_0)?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            for plane in destination.planes() {
                stream_context.validate_region(plane)?;
            }

            let roi = validate_rgb_source_planes(source_0, source_1, source_2)?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [
                source_0.as_ptr().cast(),
                source_1.as_ptr().cast(),
                source_2.as_ptr(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_0.step(),
                    source_offset.into(),
                    destination.plane_mut(0)?.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    roi.into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_uyvp_to_rgb_planar {
    ($doc:literal, $name:ident, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// `source_offset` must be `(0, 0)`.
        /// `color_space` selects the native color-conversion standard.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.
        /// The video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.
        /// The three RGB destination planes must have equal dimensions and byte strides.
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
            source: &VideoView<'_, u8>,
            source_offset: Point,
            destination_0: &mut ImageOutputView<'_, $destination_ty, C1>,
            destination_1: &mut ImageOutputView<'_, $destination_ty, C1>,
            destination_2: &mut ImageOutputView<'_, $destination_ty, C1>,
            color_space: ColorSpace,
        ) -> Result<()> {
            if source.format() != VideoFormat::Uyvp10
                || source_offset.x != 0
                || source_offset.y != 0
            {
                return Err(Error::out_of_range("uyvp format or source offset"));
            }
            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            stream_context.validate_region(destination_0)?;
            stream_context.validate_region(destination_1)?;
            stream_context.validate_region(destination_2)?;

            validate_rgb_destination_planes(
                source.size(),
                destination_0,
                destination_1,
                destination_2,
            )?;

            let mut destination_planes = [
                destination_0.as_mut_ptr().cast(),
                destination_1.as_mut_ptr().cast(),
                destination_2.as_mut_ptr().cast(),
            ];

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.planes()[0].as_ptr().cast(),
                    source.planes()[0].step(),
                    source_offset.into(),
                    destination_planes.as_mut_ptr(),
                    destination_0.step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_uyvp_to_rgb_packed {
    ($doc:literal, $name:ident, $destination_ty:ty, $destination_layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// `source_offset` must be `(0, 0)`.
        /// `color_space` selects the native color-conversion standard.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.
        /// The video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.
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
            source: &VideoView<'_, u8>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            color_space: ColorSpace,
        ) -> Result<()> {
            if source.format() != VideoFormat::Uyvp10
                || source_offset.x != 0
                || source_offset.y != 0
            {
                return Err(Error::out_of_range("uyvp format or source offset"));
            }
            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.planes()[0].as_ptr().cast(),
                    source.planes()[0].step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_uyvp_to_rgb_packed_constant_alpha {
    ($doc:literal, $name:ident, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Enqueues work without waiting for stream completion.
        /// `source_offset` must be `(0, 0)`.
        /// `color_space` selects the native color-conversion standard.
        /// Writes `alpha` to the fourth destination channel.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.
        /// The video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.
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
            source: &VideoView<'_, u8>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $destination_ty, AC4>,
            color_space: ColorSpace,
            alpha: $destination_ty,
        ) -> Result<()> {
            if source.format() != VideoFormat::Uyvp10
                || source_offset.x != 0
                || source_offset.y != 0
            {
                return Err(Error::out_of_range("uyvp format or source offset"));
            }
            stream_context.cuda_context().bind()?;
            for plane in source.planes() {
                stream_context.validate_region(plane)?;
            }
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.planes()[0].as_ptr().cast(),
                    source.planes()[0].step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    color_space.into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
