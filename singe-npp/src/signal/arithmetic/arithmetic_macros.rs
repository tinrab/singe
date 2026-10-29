macro_rules! impl_constant_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            value: $ty,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            validate_same_len(source.len(), destination.len(), "destination length")?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value.into_npp(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_constant_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            value: $ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_constant_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            value: $ty,
            destination: &mut SignalOutputView<'_, $ty>,
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
                    value.into_npp(),
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

macro_rules! impl_scaled_constant_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            value: $ty,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_constant_operation_in_place {
    ($doc:literal, $name:ident, $signal_ty:ty, $value_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $signal_ty>,
            value: $value_ty,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    signal.as_mut_ptr(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_destination_update_constant_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Destination elements must also be initialized and ready before this read-modify-write operation.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            value: $ty,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_constant_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            value: $source_ty,
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
                    value,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_constant_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            value: $source_ty,
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
                    value,
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

macro_rules! impl_shift_constant_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            shift: i32,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    shift,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_shift_constant_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            shift: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    shift,
                    signal.as_mut_ptr(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_binary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Destination elements must also be initialized and ready, and remain exclusively
        /// writable and disjoint from the read-only input signal through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;
            stream_context.validate_region(destination)?;

            validate_same_len(signal.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_binary_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_mixed_binary_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_binary_operation_in_place {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Destination elements must also be initialized and ready, and remain exclusively
        /// writable and disjoint from the read-only input signal through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;
            stream_context.validate_region(destination)?;

            validate_same_len(signal.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_heterogeneous_binary_operation {
    ($doc:literal, $name:ident, $left_ty:ty, $right_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $left_ty>,
            right: &SignalView<'_, $right_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_heterogeneous_binary_operation_in_place {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Destination elements must also be initialized and ready, and remain exclusively
        /// writable and disjoint from the read-only input signal through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;
            stream_context.validate_region(destination)?;

            validate_same_len(signal.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_binary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_scaled_binary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_binary_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_operation {
    ($doc:literal, $name:ident, $left_ty:ty, $right_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $left_ty>,
            right: &SignalView<'_, $right_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_operation_in_place {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Destination elements must also be initialized and ready, and remain exclusively
        /// writable and disjoint from the read-only input signal through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;
            stream_context.validate_region(destination)?;

            validate_same_len(signal.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_scaled_heterogeneous_binary_operation {
    ($doc:literal, $name:ident, $left_ty:ty, $right_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $left_ty>,
            right: &SignalView<'_, $right_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_binary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Destination elements must also be initialized and ready, and remain exclusively
        /// writable and disjoint from the read-only input signal through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;
            stream_context.validate_region(destination)?;

            validate_same_len(signal.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_scaled_round_binary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_round_binary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Destination elements must also be initialized and ready, and remain exclusively
        /// writable and disjoint from the read-only input signal through completion.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;
            stream_context.validate_region(destination)?;

            validate_same_len(signal.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_destination_update_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Destination elements must also be initialized and ready before this accumulation.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_destination_update_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, a signal length cannot be represented by the native interface, or NPP fails.
        /// All participating signal lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep inputs read-only and destination elements exclusively writable and disjoint
        /// from both input signals through completion.
        /// Destination elements must also be initialized and ready before this accumulation.
        /// Addresses and capacities must satisfy the selected native routine's alignment and
        /// representation requirements. Retain all signal allocations and the stream context
        /// through queued completion, including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(left)?;
            stream_context.validate_region(right)?;
            stream_context.validate_region(destination)?;

            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
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

macro_rules! impl_unary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_parameterized_unary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $parameter_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            parameter: $parameter_ty,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr(),
                    to_u64(signal.len(), "signal length")?,
                    parameter,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_unary_operation {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
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

macro_rules! impl_scaled_unary_operation_in_place {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the signal exclusively writable through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(signal)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_unary_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
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

macro_rules! impl_mixed_unary_operation {
    ($doc:literal, $name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
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

macro_rules! impl_normalize {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            subtrahend: $ty,
            divisor: $ty,
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
                    subtrahend,
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_complex_normalize {
    ($doc:literal, $name:ident, $ty:ty, $divisor_ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            subtrahend: $ty,
            divisor: $divisor_ty,
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
                    subtrahend.into_npp(),
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_normalize {
    ($doc:literal, $name:ident, $ty:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another CUDA
        /// context, the signal length cannot be represented by the native interface, or NPP fails.
        /// Source and destination lengths must match; differing lengths return an error.
        /// Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches the operation.
        /// Keep the source read-only and destination exclusively writable and disjoint from the
        /// source through queued completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain all signal allocations
        /// and the stream context through queued completion, including submission errors and
        /// every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalOutputView<'_, $ty>,
            subtrahend: $ty,
            divisor: i32,
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
                    subtrahend.into_npp(),
                    divisor,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
