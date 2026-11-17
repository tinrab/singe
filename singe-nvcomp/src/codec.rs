// Generates a codec module's limits, option conversions, and the checked operations that `format::Codec` dispatches to.
// Every path is absolute, so the invoking module imports only what its own code uses.
// Codecs whose decompression limit depends on the options, such as Gzip, declare their constants and `decompression_chunk_limit` by hand and invoke the `@operations` arm.
macro_rules! impl_nvcomp_batched_codec_module {
    (
        @operations
        compress_options = $compress_options:ident,
        decompress_options = $decompress_options:ident,
        compress_get_required_alignments = $compress_get_required_alignments:ident,
        decompress_get_required_alignments = $decompress_get_required_alignments:ident,
        compress_get_temp_size = $compress_get_temp_size:ident,
        compress_get_temp_size_sync = $compress_get_temp_size_sync:ident,
        decompress_get_temp_size = $decompress_get_temp_size:ident,
        decompress_get_temp_size_sync = $decompress_get_temp_size_sync:ident,
        compress_get_max_output_chunk_size = $compress_get_max_output_chunk_size:ident,
        compress_async = $compress_async:ident,
        get_decompress_size_async = $get_decompress_size_async:ident,
        decompress_async = $decompress_async:ident $(,)?
    ) => {
        impl CompressOptions {
            /// Returns the native nvCOMP options.
            #[must_use]
            pub fn as_raw(self) -> $crate::sys::$compress_options {
                self.into()
            }
        }

        impl DecompressOptions {
            /// Returns the native nvCOMP options.
            #[must_use]
            pub fn as_raw(self) -> $crate::sys::$decompress_options {
                self.into()
            }
        }

        pub(crate) fn compression_alignments(
            options: CompressOptions,
        ) -> $crate::error::Result<$crate::types::AlignmentRequirements> {
            let mut requirements = $crate::sys::nvcompAlignmentRequirements_t::default();
            // SAFETY: the output location is a live local valid for writes.
            unsafe {
                $crate::error::Error::check($crate::sys::$compress_get_required_alignments(
                    options.into(),
                    &raw mut requirements,
                ))?;
            }
            requirements.try_into()
        }

        pub(crate) fn decompression_alignments(
            options: DecompressOptions,
        ) -> $crate::error::Result<$crate::types::AlignmentRequirements> {
            let mut requirements = $crate::sys::nvcompAlignmentRequirements_t::default();
            // SAFETY: the output location is a live local valid for writes.
            unsafe {
                $crate::error::Error::check($crate::sys::$decompress_get_required_alignments(
                    options.into(),
                    &raw mut requirements,
                ))?;
            }
            requirements.try_into()
        }

        pub(crate) fn compression_workspace_size(
            num_chunks: usize,
            max_uncompressed_chunk_bytes: usize,
            options: CompressOptions,
            max_total_uncompressed_bytes: usize,
            stream: &::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<usize> {
            $crate::utility::validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                max_uncompressed_chunk_bytes,
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;

            let mut workspace_bytes = 0;
            stream.context().bind()?;
            // SAFETY: the query reads only host values, the stream is live in the bound context, and the output location is a live local.
            unsafe {
                $crate::error::Error::check($crate::sys::$compress_get_temp_size(
                    $crate::utility::to_u64(num_chunks, "num_chunks")?,
                    $crate::utility::to_u64(
                        max_uncompressed_chunk_bytes,
                        "max_uncompressed_chunk_bytes",
                    )?,
                    options.into(),
                    &raw mut workspace_bytes,
                    $crate::utility::to_u64(
                        max_total_uncompressed_bytes,
                        "max_total_uncompressed_bytes",
                    )?,
                    stream.as_raw(),
                ))?;
            }
            $crate::utility::to_usize(workspace_bytes, "workspace bytes")
        }

        pub(crate) fn compression_workspace_size_for_chunks(
            input: &$crate::batch::DeviceChunkTable<'_>,
            options: CompressOptions,
            stream: &::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<usize> {
            $crate::utility::validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                input.max_chunk_bytes(),
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;
            let max_chunk_bytes = input.max_chunk_bytes();
            let total_bytes = input.total_bytes();
            let input = input.view();
            stream.complete(|| {
                input.join_regions(stream)?;
                stream.context().bind()?;
                let mut workspace_bytes = 0;
                // SAFETY: the table's device arrays describe `input.len()` borrowed chunks, which the table guarantees are initialized; `complete` waits for the stream before the borrows end.
                unsafe {
                    $crate::error::Error::check($crate::sys::$compress_get_temp_size_sync(
                        input
                            .device_chunk_ptrs()
                            .cast::<*const ::std::ffi::c_void>()
                            .cast_const(),
                        input.device_chunk_bytes().cast(),
                        $crate::utility::to_u64(input.len(), "num_chunks")?,
                        $crate::utility::to_u64(max_chunk_bytes, "max_uncompressed_chunk_bytes")?,
                        options.into(),
                        &raw mut workspace_bytes,
                        $crate::utility::to_u64(total_bytes, "max_total_uncompressed_bytes")?,
                        stream.as_raw(),
                    ))?;
                }
                $crate::utility::to_usize(workspace_bytes, "workspace bytes")
            })
        }

        pub(crate) fn max_output_chunk_size(
            max_uncompressed_chunk_bytes: usize,
            options: CompressOptions,
        ) -> $crate::error::Result<usize> {
            $crate::utility::validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                max_uncompressed_chunk_bytes,
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;

            let mut max_compressed_chunk_bytes = 0;
            // SAFETY: the query reads only host values, and the output location is a live local.
            unsafe {
                $crate::error::Error::check($crate::sys::$compress_get_max_output_chunk_size(
                    $crate::utility::to_u64(
                        max_uncompressed_chunk_bytes,
                        "max_uncompressed_chunk_bytes",
                    )?,
                    options.into(),
                    &raw mut max_compressed_chunk_bytes,
                ))?;
            }
            $crate::utility::to_usize(max_compressed_chunk_bytes, "maximum compressed chunk bytes")
        }

        pub(crate) fn decompression_workspace_size(
            num_chunks: usize,
            max_uncompressed_chunk_bytes: usize,
            options: DecompressOptions,
            max_total_uncompressed_bytes: usize,
            stream: &::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<usize> {
            $crate::utility::validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                max_uncompressed_chunk_bytes,
                decompression_chunk_limit(options),
            )?;

            let mut workspace_bytes = 0;
            stream.context().bind()?;
            // SAFETY: the query reads only host values, the stream is live in the bound context, and the output location is a live local.
            unsafe {
                $crate::error::Error::check($crate::sys::$decompress_get_temp_size(
                    $crate::utility::to_u64(num_chunks, "num_chunks")?,
                    $crate::utility::to_u64(
                        max_uncompressed_chunk_bytes,
                        "max_uncompressed_chunk_bytes",
                    )?,
                    options.into(),
                    &raw mut workspace_bytes,
                    $crate::utility::to_u64(
                        max_total_uncompressed_bytes,
                        "max_total_uncompressed_bytes",
                    )?,
                    stream.as_raw(),
                ))?;
            }
            $crate::utility::to_usize(workspace_bytes, "workspace bytes")
        }

        pub(crate) unsafe fn decompression_workspace_size_for_chunks(
            input: $crate::batch::DeviceChunkTableView<'_>,
            output: &$crate::batch::DeviceChunkTableMut<'_>,
            options: DecompressOptions,
            statuses: &mut $crate::batch::DeviceStatuses,
            stream: &::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<usize> {
            $crate::utility::validate_same_len(input.len(), output.len(), "output chunks")?;
            $crate::utility::validate_chunk_size_limit(
                "output chunk capacity",
                output.max_buffer_bytes(),
                decompression_chunk_limit(options),
            )?;
            statuses.require(input.len())?;
            input.validate_context(stream)?;
            statuses.validate_context(stream)?;
            let max_chunk_bytes = output.max_buffer_bytes();
            let total_bytes = output.total_buffer_bytes();
            stream.complete(|| {
                $crate::batch::join_operands(stream, input.regions(), &[], None, Some(&*statuses))?;
                stream.context().bind()?;
                let mut workspace_bytes = 0;
                // SAFETY: the table's device arrays describe `input.len()` borrowed chunks, which the caller guarantees are initialized, and `statuses` has one slot per chunk; `complete` waits for the stream before the borrows end.
                unsafe {
                    $crate::error::Error::check($crate::sys::$decompress_get_temp_size_sync(
                        input
                            .device_chunk_ptrs()
                            .cast::<*const ::std::ffi::c_void>()
                            .cast_const(),
                        input.device_chunk_bytes().cast(),
                        $crate::utility::to_u64(input.len(), "num_chunks")?,
                        $crate::utility::to_u64(max_chunk_bytes, "max_uncompressed_chunk_bytes")?,
                        &raw mut workspace_bytes,
                        $crate::utility::to_u64(total_bytes, "max_total_uncompressed_bytes")?,
                        options.into(),
                        statuses.as_device_ptr().cast(),
                        stream.as_raw(),
                    ))?;
                }
                $crate::utility::to_usize(workspace_bytes, "workspace bytes")
            })
        }

        pub(crate) unsafe fn compress_chunks_async<'a>(
            input: &'a $crate::batch::DeviceChunkTable<'_>,
            output: &'a mut $crate::batch::DeviceChunkTableMut<'_>,
            mut workspace: Option<::singe_cuda::view::DeviceOutputView<'_, u8>>,
            statuses: &'a mut $crate::batch::DeviceStatuses,
            options: CompressOptions,
            stream: &'a ::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<$crate::batch::AsyncBatchResult<'a>> {
            let input_view = input.view();
            input_view.validate_context(stream)?;
            output.buffers().validate_context(stream)?;
            statuses.validate_context(stream)?;
            $crate::utility::validate_same_len(input.len(), output.len(), "output chunks")?;
            $crate::utility::validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                input.max_chunk_bytes(),
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;
            statuses.require(input.len())?;

            stream.context().bind()?;
            let required_output_bytes = max_output_chunk_size(input.max_chunk_bytes(), options)?;
            if output.min_buffer_bytes() < required_output_bytes {
                return Err($crate::error::Error::BufferTooSmall {
                    name: "output chunk capacity".into(),
                    required: required_output_bytes,
                    actual: output.min_buffer_bytes(),
                });
            }

            let alignments = compression_alignments(options)?;
            $crate::utility::validate_alignment(
                "input chunks",
                input.is_aligned_to(alignments.input),
                alignments.input,
            )?;
            $crate::utility::validate_alignment(
                "output chunks",
                output.is_aligned_to(alignments.output),
                alignments.output,
            )?;
            let required_workspace_bytes = compression_workspace_size(
                input.len(),
                input.max_chunk_bytes(),
                options,
                input.total_bytes(),
                stream,
            )?;
            $crate::batch::check_workspace(
                workspace.as_ref(),
                required_workspace_bytes,
                alignments.temp,
                stream,
            )?;

            $crate::batch::join_operands(
                stream,
                input_view.regions(),
                output.buffers().regions(),
                workspace.as_ref(),
                Some(&*statuses),
            )?;
            let (workspace_ptr, workspace_bytes) = $crate::batch::workspace_parts(&mut workspace);
            // SAFETY: the checks above establish matching contexts, chunk counts, output capacities, alignments, and a large enough workspace, and the operands are ordered after their producers; the caller keeps every operand alive and unaliased until the work finishes.
            unsafe {
                $crate::error::Error::check($crate::sys::$compress_async(
                    input_view
                        .device_chunk_ptrs()
                        .cast::<*const ::std::ffi::c_void>()
                        .cast_const(),
                    input_view.device_chunk_bytes().cast(),
                    $crate::utility::to_u64(
                        input.max_chunk_bytes(),
                        "max_uncompressed_chunk_bytes",
                    )?,
                    $crate::utility::to_u64(input.len(), "num_chunks")?,
                    workspace_ptr.cast(),
                    $crate::utility::to_u64(workspace_bytes, "workspace bytes")?,
                    output.buffers().device_chunk_ptrs().cast(),
                    output.written_bytes_device_ptr().cast(),
                    options.into(),
                    statuses.as_device_ptr().cast(),
                    stream.as_raw(),
                ))?;
            }

            // SAFETY: the launch above writes every status slot on `stream`, and the caller keeps the operands alive until it finishes.
            Ok(unsafe { $crate::batch::AsyncBatchResult::from_submission(statuses, stream) })
        }

        pub(crate) unsafe fn decompressed_sizes_async<S>(
            input: $crate::batch::DeviceChunkTableView<'_>,
            uncompressed_chunk_bytes: &mut S,
            stream: &::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<()>
        where
            S: ::singe_cuda::view::DeviceOutput<usize> + ?Sized,
        {
            input.validate_context(stream)?;
            if uncompressed_chunk_bytes.context() != stream.context() {
                return Err(::singe_cuda::error::Error::MemoryContextMismatch.into());
            }

            $crate::utility::validate_same_len(input.len(), uncompressed_chunk_bytes.len(), "uncompressed chunk bytes")?;
            $crate::batch::join_operands(
                stream,
                input.regions(),
                &[::singe_cuda::view::DeviceRegionProvider::region(&*uncompressed_chunk_bytes)],
                None,
                None,
            )?;
            stream.context().bind()?;
            // SAFETY: the table and the size array have one entry per chunk in the stream's context, and the operands are ordered after their producers; the caller keeps them alive and unaliased until the work finishes.
            unsafe {
                $crate::error::Error::check($crate::sys::$get_decompress_size_async(
                    input
                        .device_chunk_ptrs()
                        .cast::<*const ::std::ffi::c_void>()
                        .cast_const(),
                    input.device_chunk_bytes().cast(),
                    uncompressed_chunk_bytes.as_mut_ptr().cast(),
                    $crate::utility::to_u64(input.len(), "num_chunks")?,
                    stream.as_raw(),
                ))
            }
        }

        pub(crate) unsafe fn decompress_chunks_async<'a>(
            input: $crate::batch::DeviceChunkTableView<'a>,
            output: &'a mut $crate::batch::DeviceChunkTableMut<'_>,
            mut workspace: Option<::singe_cuda::view::DeviceOutputView<'_, u8>>,
            statuses: &'a mut $crate::batch::DeviceStatuses,
            options: DecompressOptions,
            stream: &'a ::singe_cuda::stream::Stream,
        ) -> $crate::error::Result<$crate::batch::AsyncBatchResult<'a>> {
            input.validate_context(stream)?;
            output.buffers().validate_context(stream)?;
            statuses.validate_context(stream)?;
            $crate::utility::validate_same_len(input.len(), output.len(), "output chunks")?;
            $crate::utility::validate_chunk_size_limit(
                "output chunk capacity",
                output.max_buffer_bytes(),
                decompression_chunk_limit(options),
            )?;
            statuses.require(input.len())?;

            stream.context().bind()?;
            let alignments = decompression_alignments(options)?;
            $crate::utility::validate_alignment(
                "input chunks",
                input.is_aligned_to(alignments.input),
                alignments.input,
            )?;
            $crate::utility::validate_alignment(
                "output chunks",
                output.is_aligned_to(alignments.output),
                alignments.output,
            )?;
            let required_workspace_bytes = decompression_workspace_size(
                input.len(),
                output.max_buffer_bytes(),
                options,
                output.total_buffer_bytes(),
                stream,
            )?;
            $crate::batch::check_workspace(
                workspace.as_ref(),
                required_workspace_bytes,
                alignments.temp,
                stream,
            )?;

            $crate::batch::join_operands(
                stream,
                input.regions(),
                output.buffers().regions(),
                workspace.as_ref(),
                Some(&*statuses),
            )?;
            let (workspace_ptr, workspace_bytes) = $crate::batch::workspace_parts(&mut workspace);
            // SAFETY: the checks above establish matching contexts, chunk counts, output limits, alignments, and a large enough workspace, and the operands are ordered after their producers; the caller guarantees valid compressed input and keeps every operand alive and unaliased until the work finishes.
            unsafe {
                $crate::error::Error::check($crate::sys::$decompress_async(
                    input
                        .device_chunk_ptrs()
                        .cast::<*const ::std::ffi::c_void>()
                        .cast_const(),
                    input.device_chunk_bytes().cast(),
                    output.buffers().device_chunk_bytes().cast(),
                    output.written_bytes_device_ptr().cast(),
                    $crate::utility::to_u64(input.len(), "num_chunks")?,
                    workspace_ptr.cast(),
                    $crate::utility::to_u64(workspace_bytes, "workspace bytes")?,
                    output.buffers().device_chunk_ptrs().cast(),
                    options.into(),
                    statuses.as_device_ptr().cast(),
                    stream.as_raw(),
                ))?;
            }

            // SAFETY: the launch above writes every status slot on `stream`, and the caller keeps the operands alive until it finishes.
            Ok(unsafe { $crate::batch::AsyncBatchResult::from_submission(statuses, stream) })
        }
    };
    (
        $format:literal,
        compress_options = $compress_options:ident,
        decompress_options = $decompress_options:ident,
        compression_max_allowed_chunk_size = $compression_max_allowed_chunk_size:ident,
        decompression_max_allowed_chunk_size = $decompression_max_allowed_chunk_size:ident,
        required_compression_alignment = $required_compression_alignment:ident,
        required_decompression_alignment = $required_decompression_alignment:ident,
        $($operations:tt)*
    ) => {
        /// Largest uncompressed chunk size, in bytes, that compression accepts.
        pub const COMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
            $crate::utility::codec_size($crate::sys::$compression_max_allowed_chunk_size);

        /// Largest compressed or decompressed chunk size, in bytes, that decompression accepts.
        ///
        /// The hardware decompression engine may have a lower limit; see [`DecompressBackend::Hardware`](crate::types::DecompressBackend::Hardware).
        pub const DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
            $crate::utility::codec_size($crate::sys::$decompression_max_allowed_chunk_size);

        /// Alignment, in bytes, that meets every minimum alignment for compression buffers.
        ///
        /// Typed buffers must also be aligned to their element size, such as 4 bytes for an `i32`.
        pub const REQUIRED_COMPRESSION_ALIGNMENT: usize =
            $crate::utility::codec_size($crate::sys::$required_compression_alignment);

        /// Alignment, in bytes, that meets every minimum alignment for decompression buffers.
        ///
        /// Typed buffers must also be aligned to their element size, such as 4 bytes for an `i32`.
        pub const REQUIRED_DECOMPRESSION_ALIGNMENT: usize =
            $crate::utility::codec_size($crate::sys::$required_decompression_alignment);

        const fn decompression_chunk_limit(_options: DecompressOptions) -> usize {
            DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE
        }

        impl_nvcomp_batched_codec_module! {
            @operations
            compress_options = $compress_options,
            decompress_options = $decompress_options,
            $($operations)*
        }
    };
}

pub(crate) use impl_nvcomp_batched_codec_module;
