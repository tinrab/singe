macro_rules! impl_convert {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert_scaled {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Applies native scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert_scaled_round {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Applies native scaling using `scale_factor`.
        /// Uses `round_mode` when converting to the destination representation.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            round_mode: RoundMode,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?
                        .try_into()
                        .map_err(|_| $crate::error::Error::out_of_range("source length"))?, // can be `Npp32u`
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            level: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_complex {
    ($doc:literal, $name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            level: $level_ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_complex_in_place {
    ($doc:literal, $name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            level: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_complex {
    ($doc:literal, $name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            level: $level_ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_complex_in_place {
    ($doc:literal, $name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            level: $ty,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $ty,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_complex {
    ($doc:literal, $name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            level: $level_ty,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    value.into_npp(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_complex_in_place {
    ($doc:literal, $name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    value.into_npp(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
