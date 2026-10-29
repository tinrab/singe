macro_rules! impl_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the required scratch capacity, in bytes, for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source length cannot be
        /// represented by the native interface, NPP rejects the query, or the returned
        /// scratch size cannot be represented as `usize`.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Writes the scalar result into the single element of `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another
        /// CUDA context, `destination` does not contain exactly one element, a length or
        /// queried size exceeds its native integer range, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and every captured graph replay; destination writes must not conflict with other access.
        /// `scratch` must belong to this CUDA context, remain live and exclusively writable
        /// through completion, and be disjoint from source and destination storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            destination: &mut SignalOutputView<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another
        /// CUDA context, `destination` does not contain exactly one element, a length or
        /// queried size exceeds its native integer range, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and every captured graph replay; destination writes must not conflict with other access.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            destination: &mut SignalOutputView<'_, $type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(stream_context, source, destination, &mut scratch)
            }
        }
    };
}

macro_rules! impl_mixed_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the required scratch capacity, in bytes, for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source length cannot be
        /// represented by the native interface, NPP rejects the query, or the returned
        /// scratch size cannot be represented as `usize`.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Writes the scalar result into the single element of `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another
        /// CUDA context, `destination` does not contain exactly one element, a length or
        /// queried size exceeds its native integer range, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and every captured graph replay; destination writes must not conflict with other access.
        /// `scratch` must belong to this CUDA context, remain live and exclusively writable
        /// through completion, and be disjoint from source and destination storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another
        /// CUDA context, `destination` does not contain exactly one element, a length or
        /// queried size exceeds its native integer range, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and every captured graph replay; destination writes must not conflict with other access.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(stream_context, source, destination, &mut scratch)
            }
        }
    };
}

macro_rules! impl_binary_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $type>,
            source_2: &SignalView<'_, $type>,
            destination: &mut SignalOutputView<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $type>,
            source_2: &SignalView<'_, $type>,
            destination: &mut SignalOutputView<'_, $type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source_1)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_mixed_binary_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source_1)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_heterogeneous_binary_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_1_type:ty,
        $source_2_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_1_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source_1)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_scaled_binary_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source_1)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    scale_factor,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_1_type:ty,
        $source_2_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_1_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source_1)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    scale_factor,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_scaled_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the required scratch capacity, in bytes, for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the source length cannot be
        /// represented by the native interface, NPP rejects the query, or the returned
        /// scratch size cannot be represented as `usize`.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another
        /// CUDA context, `destination` does not contain exactly one element, a length or
        /// queried size exceeds its native integer range, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and every captured graph replay; destination writes must not conflict with other access.
        /// `scratch` must belong to this CUDA context, remain live and exclusively writable
        /// through completion, and be disjoint from source and destination storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Writes the scalar result into the single element of `destination`.
        /// Allocates temporary scratch storage for this call.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a signal belongs to another
        /// CUDA context, `destination` does not contain exactly one element, a length or
        /// queried size exceeds its native integer range, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Retain the source, destination, and stream context through queued completion
        /// and every captured graph replay; destination writes must not conflict with other access.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source,
                    destination,
                    scale_factor,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_error_metric {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, f64>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// The two sources must have equal lengths.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, f64>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source_1)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source_1,
                    source_2,
                    destination,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_mean_stddev {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `mean`, `standard_deviation`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            mean: &mut SignalOutputView<'_, $type>,
            standard_deviation: &mut SignalOutputView<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;

            validate_scalar_output(mean, "mean")?;
            validate_scalar_output(standard_deviation, "standard_deviation")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `mean`, `standard_deviation`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            mean: &mut SignalOutputView<'_, $type>,
            standard_deviation: &mut SignalOutputView<'_, $type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source,
                    mean,
                    standard_deviation,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_scaled_mean_stddev {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `mean`, `standard_deviation`.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            mean: &mut SignalOutputView<'_, $destination_type>,
            standard_deviation: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;

            validate_scalar_output(mean, "mean")?;
            validate_scalar_output(standard_deviation, "standard_deviation")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `mean`, `standard_deviation`.
        /// Allocates temporary scratch storage for this call.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            mean: &mut SignalOutputView<'_, $destination_type>,
            standard_deviation: &mut SignalOutputView<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mean)?;
            stream_context.validate_region(standard_deviation)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source,
                    mean,
                    standard_deviation,
                    scale_factor,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_indexed_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `value`, `index`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            value: &mut SignalOutputView<'_, $type>,
            index: &mut SignalOutputView<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(value)?;
            stream_context.validate_region(index)?;

            validate_scalar_output(value, "value")?;
            validate_scalar_output(index, "index")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    value.as_mut_ptr().cast(),
                    index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `value`, `index`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            value: &mut SignalOutputView<'_, $type>,
            index: &mut SignalOutputView<'_, i32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(value)?;
            stream_context.validate_region(index)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(stream_context, source, value, index, &mut scratch)
            }
        }
    };
}

macro_rules! impl_pair_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `min`, `max`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalOutputView<'_, $type>,
            max: &mut SignalOutputView<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(min)?;
            stream_context.validate_region(max)?;

            validate_scalar_output(min, "min")?;
            validate_scalar_output(max, "max")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    min.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `min`, `max`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalOutputView<'_, $type>,
            max: &mut SignalOutputView<'_, $type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(min)?;
            stream_context.validate_region(max)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(stream_context, source, min, max, &mut scratch)
            }
        }
    };
}

macro_rules! impl_indexed_pair_statistic {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `min`, `min_index`, `max`, `max_index`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalOutputView<'_, $type>,
            min_index: &mut SignalOutputView<'_, i32>,
            max: &mut SignalOutputView<'_, $type>,
            max_index: &mut SignalOutputView<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(min)?;
            stream_context.validate_region(min_index)?;
            stream_context.validate_region(max)?;
            stream_context.validate_region(max_index)?;

            validate_scalar_output(min, "min")?;
            validate_scalar_output(min_index, "min_index")?;
            validate_scalar_output(max, "max")?;
            validate_scalar_output(max_index, "max_index")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    min.as_mut_ptr().cast(),
                    min_index.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    max_index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the results to `min`, `min_index`, `max`, `max_index`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalOutputView<'_, $type>,
            min_index: &mut SignalOutputView<'_, i32>,
            max: &mut SignalOutputView<'_, $type>,
            max_index: &mut SignalOutputView<'_, i32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(min)?;
            stream_context.validate_region(min_index)?;
            stream_context.validate_region(max)?;
            stream_context.validate_region(max_index)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source,
                    min,
                    min_index,
                    max,
                    max_index,
                    &mut scratch,
                )
            }
        }
    };
}

macro_rules! impl_every_statistic_in_place {
    ($doc:literal, $name:ident, $type:ty, $statistic:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Both signals must contain initialized values ready for the selected stream.
        /// Keep the source read-only and the destination exclusively writable and disjoint from the source.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            destination: &mut SignalViewMut<'_, $type>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_len(source.len(), destination.len(), "destination")?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_zero_crossing {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        /// Returns the scratch capacity, in bytes, required for the source length.
        #[doc = concat!("Queries [`sys::", stringify!($buffer_size), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the size query.
        /// Source lengths and the returned scratch size must fit their required integer types.
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            stream_context.cuda_context().bind()?;
            let mut bytes = 0;
            // SAFETY: the `buffer_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Returns an error if `scratch` is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from all source and result storage.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    zero_crossing_type.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($statistic), "`].")]
        ///
        /// Each result view must contain exactly one element.
        /// Writes the result to `destination`.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Each result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or scratch-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalOutputView<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let required_bytes = $buffer_size_name(stream_context, source)?;
                let mut scratch =
                    ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
                $with_scratch_name(
                    stream_context,
                    source,
                    destination,
                    zero_crossing_type,
                    &mut scratch,
                )
            }
        }
    };
}
