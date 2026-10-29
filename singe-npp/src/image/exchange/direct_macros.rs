macro_rules! impl_convert {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        /// Converts source samples to the destination element type.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Preserves numeric values subject to native rounding and saturation; it does not scale their range.
        /// Integer results are clamped to the destination type's range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
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
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageOutputView<'_, $dst_ty, $layout>,
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
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert_round {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        /// Converts source samples to the destination element type.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Preserves numeric values subject to native rounding and saturation; it does not scale their range.
        /// Uses `round_mode` for conversion to integer values.
        /// Integer results are clamped to the destination type's range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
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
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageOutputView<'_, $dst_ty, $layout>,
            round_mode: RoundMode,
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
                    destination.size().into(),
                    round_mode.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert_scaled_round {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        /// Converts source samples to the destination element type.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        /// Multiplies samples by `2^(-scale_factor)` before rounding and saturation.
        /// Uses `round_mode` for conversion to integer values.
        /// Integer results are clamped to the destination type's range.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, source and destination sizes differ, or NPP rejects the conversion parameters.
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
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageOutputView<'_, $dst_ty, $layout>,
            round_mode: RoundMode,
            scale_factor: i32,
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
                    destination.size().into(),
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
