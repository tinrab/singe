macro_rules! impl_filter_median {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi_name:ident) => {
        /// Returns the scratch size in bytes for median filtering of `roi` using `mask_size`.
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if mask dimensions are nonpositive, NPP rejects the ROI or mask, or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            mask_size: Size,
        ) -> Result<usize> {
            validate_filter_mask(mask_size, Point { x: 0, y: 0 })?;
            let mut bytes = 0_u32;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    mask_size.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        /// Applies a median filter over the neighborhood defined by `mask_size` and `anchor`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// Source and destination image sizes must match.
        /// Mask dimensions must be positive, and `anchor` must lie within the mask.
        /// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// The source allocation must include all neighboring samples read by the filter mask, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source.size(), destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), mask_size)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size.into(),
                    anchor.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_filter_median_border {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi_name:ident) => {
        /// Returns the scratch size in bytes for median filtering of `roi` using `mask_size`.
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if mask dimensions are nonpositive, NPP rejects the ROI or mask or border mode, or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            mask_size: Size,
            border_type: BorderType,
        ) -> Result<usize> {
            validate_filter_mask(mask_size, Point { x: 0, y: 0 })?;
            let mut bytes = 0_u32;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    mask_size.into(),
                    &raw mut bytes,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        /// Applies a median filter over the neighborhood defined by `mask_size` and `anchor` with border control.
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, and `anchor` must lie within the mask.
        /// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
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
            stream_context.stream().ensure_not_capturing()?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            let required_bytes =
                $buffer_size_name(stream_context, destination.size(), mask_size, border_type)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
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
                    scratch.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_filter_unsharp_border {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi_name:ident) => {
        /// Returns the scratch size in bytes for unsharp-mask sharpening with the given `radius` and `sigma`.
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if `radius` or `sigma` does not compare greater than zero, NPP rejects the query, or the returned size cannot be represented as `usize`.
        pub fn $buffer_size_name(radius: f32, sigma: f32) -> Result<usize> {
            validate_unsharp_parameters(radius, sigma)?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(radius, sigma, &raw mut bytes,))?;
            }
            to_usize(bytes, "buffer size")
        }

        /// Sharpens the source using an unsharp mask with border control.
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        ///
        /// `radius` is the Gaussian radius in pixels excluding the center pixel; `sigma` is its standard deviation in pixels.
        /// `weight` scales the high-pass contribution, and `threshold` selects the minimum absolute high-pass value to add back.
        /// Allocates temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// `radius` and `sigma` must compare greater than zero; the wrapper does not validate `weight` or `threshold`.
        /// The native unsharp filter supports `BorderType::Replicate`; the wrapper forwards `border_type` without validating it.
        /// Scratch-size queries, device allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $pixel_ty, $layout>,
            radius: f32,
            sigma: f32,
            weight: f32,
            threshold: f32,
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
            stream_context.stream().ensure_not_capturing()?;

            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_unsharp_parameters(radius, sigma)?;
            let required_bytes = $buffer_size_name(radius, sigma)?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    radius,
                    sigma,
                    weight,
                    threshold,
                    border_type.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}
