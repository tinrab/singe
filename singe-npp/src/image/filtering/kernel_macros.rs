macro_rules! impl_filter_wiener_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $noise_len:expr, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Uses local statistics from `mask_size` around `anchor` and one noise variance level per processed channel.
        /// NPP expects noise levels in `0.0..=1.0`; this wrapper does not validate their values.
        /// AC4 alpha and row padding remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, `anchor` must lie within the mask, and `border_type` must be `BorderType::Replicate`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Noise storage must satisfy the native address-space requirements and remain exclusively available for as long as the native routine uses it.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
            noise: &mut [f32; $noise_len],
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            if border_type != BorderType::Replicate {
                return Err(Error::out_of_range("border type"));
            }
            {
                // SAFETY: the `ffi_name` entry point receives live handles owned by the wrappers and valid output locations.
                unsafe {
                    try_ffi!(sys::$ffi_name(
                        source_pointer.cast(),
                        source.step(),
                        source.size().into(),
                        source_offset.into(),
                        destination.as_mut_ptr().cast(),
                        destination.step(),
                        destination.size().into(),
                        mask_size.into(),
                        anchor.into(),
                        noise.as_mut_ptr().cast(),
                        border_type.into(),
                        stream_context.as_raw(),
                    ))
                }?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Applies the selected `mask_size` and enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            mask_size: MaskSize,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel_border {
    ($doc:literal, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Applies the selected `mask_size` and enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
        /// `source_offset` must address a source pixel, its ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// NPP can also reject the selected `border_type`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            mask_size: MaskSize,
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel_typed {
    ($doc:literal, $name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Applies the selected `mask_size` and enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            mask_size: MaskSize,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_masked_kernel_typed_border {
    ($doc:literal, $name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Applies the selected `mask_size` and enqueues work on the selected stream.
        /// Ignored channels and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the mask, image geometry, or filter parameters.
        /// `source_offset` must address a source pixel, its ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// NPP can also reject the selected `border_type`.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            mask_size: MaskSize,
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_gradient_vector_border {
    (
        $doc:literal,
        $name:ident,
        $source_ty:ty,
        $source_layout:ty,
        $gradient_ty:ty,
        $ffi_name:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
        /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination_x: Option<&mut ImageOutputView<'_, $gradient_ty, C1>>,
            destination_y: Option<&mut ImageOutputView<'_, $gradient_ty, C1>>,
            destination_magnitude: Option<&mut ImageOutputView<'_, $gradient_ty, C1>>,
            destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
            mask_size: MaskSize,
            norm: ImageNormalization,
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            if let Some(image) = destination_x.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_y.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_magnitude.as_ref() {
                stream_context.validate_region(*image)?;
            }
            if let Some(image) = destination_angle.as_ref() {
                stream_context.validate_region(*image)?;
            }

                let (destination_x_ptr, destination_x_step, destination_x_size) =
                    optional_image_output(destination_x);
                let (destination_y_ptr, destination_y_step, destination_y_size) =
                    optional_image_output(destination_y);
                let (
                    destination_magnitude_ptr,
                    destination_magnitude_step,
                    destination_magnitude_size,
                ) = optional_image_output(destination_magnitude);
                let (destination_angle_ptr, destination_angle_step, destination_angle_size) =
                    optional_image_output(destination_angle);
                let roi = validate_gradient_vector_outputs(
                    destination_x_size,
                    destination_y_size,
                    destination_magnitude_size,
                    destination_angle_size,
                )?;
                validate_border_roi(source.size(), source_offset, roi)?;

                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    try_ffi!(sys::$ffi_name(
                        source_pointer.cast(),
                        source.step(),
                        source.size().into(),
                        source_offset.into(),
                        destination_x_ptr.cast(),
                        destination_x_step,
                        destination_y_ptr.cast(),
                        destination_y_step,
                        destination_magnitude_ptr.cast(),
                        destination_magnitude_step,
                        destination_angle_ptr.cast(),
                        destination_angle_step,
                        roi.into(),
                        mask_size.into(),
                        norm.into(),
                        border_type.into(),
                        stream_context.as_raw(),
                    ))?;
                }
                Ok(())

        }
    };
}
