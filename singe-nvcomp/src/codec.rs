macro_rules! impl_nvcomp_batched_codec_module {
    (
        $format:literal,
        compress_options = $compress_options:ident,
        decompress_options = $decompress_options:ident,
        compression_max_allowed_chunk_size = $compression_max_allowed_chunk_size:ident,
        decompression_max_allowed_chunk_size = $decompression_max_allowed_chunk_size:ident,
        required_compression_alignment = $required_compression_alignment:ident,
        required_decompression_alignment = $required_decompression_alignment:ident,
        compress_get_required_alignments = $compress_get_required_alignments:ident,
        decompress_get_required_alignments = $decompress_get_required_alignments:ident,
        compress_get_temporary_size = $compress_get_temporary_size:ident,
        compress_get_temporary_size_sync = $compress_get_temporary_size_sync:ident,
        decompress_get_temporary_size = $decompress_get_temporary_size:ident,
        decompress_get_temporary_size_sync = $decompress_get_temporary_size_sync:ident,
        compress_get_max_output_chunk_size = $compress_get_max_output_chunk_size:ident,
        compress_raw_async = $compress_raw_async:ident,
        decompress_size_raw_async = $decompress_size_raw_async:ident,
        decompress_raw_async = $decompress_raw_async:ident $(,)?
    ) => {
        use std::ffi::c_void;

        use crate::{
            batch::{
                AsyncBatchResult, DeviceChunkTable, DeviceChunkTableMut, DeviceChunkTableView,
                DeviceStatuses,
            },
            error::{Error, Result},
            try_ffi,
            types::AlignmentRequirements,
            utility::{
                to_u64, to_usize, validate_alignment, validate_chunk_size_limit, validate_same_len,
            },
            workspace::TemporaryBuffer,
        };

        impl CompressOptions {
            /// Converts these options to the native nvCOMP struct.
            pub fn as_raw(self) -> sys::$compress_options {
                self.into()
            }
        }

        impl DecompressOptions {
            /// Converts these options to the native nvCOMP struct.
            pub fn as_raw(self) -> sys::$decompress_options {
                self.into()
            }
        }
        use singe_cuda::{stream::Stream, types::DevicePtr, view::DeviceOutput};

        /// Largest uncompressed chunk size, in bytes, that compression accepts.
        pub const COMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
            $crate::utility::codec_size(sys::$compression_max_allowed_chunk_size);

        /// Largest compressed or decompressed chunk size, in bytes, that decompression accepts.
        ///
        /// The hardware decompression engine may have a lower limit; see [`DecompressBackend::Hardware`](crate::types::DecompressBackend::Hardware).
        pub const DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
            $crate::utility::codec_size(sys::$decompression_max_allowed_chunk_size);

        /// Alignment, in bytes, that meets every minimum alignment for compression buffers.
        ///
        /// Typed buffers must also be aligned to their element size, such as 4 bytes for an `i32`.
        pub const REQUIRED_COMPRESSION_ALIGNMENT: usize = $crate::utility::codec_size(sys::$required_compression_alignment);

        /// Alignment, in bytes, that meets every minimum alignment for decompression buffers.
        ///
        /// Typed buffers must also be aligned to their element size, such as 4 bytes for an `i32`.
        pub const REQUIRED_DECOMPRESSION_ALIGNMENT: usize =
            $crate::utility::codec_size(sys::$required_decompression_alignment);

        pub(crate) fn compression_alignments(options: CompressOptions) -> Result<AlignmentRequirements> {
            let mut requirements = sys::nvcompAlignmentRequirements_t::default();
            // SAFETY: the `compress_get_required_alignments` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$compress_get_required_alignments(
                    options.into(),
                    &raw mut requirements,
                ))?;
            }
            requirements.try_into()
        }

        pub(crate) fn decompression_alignments(
            options: DecompressOptions,
        ) -> Result<AlignmentRequirements> {
            let mut requirements = sys::nvcompAlignmentRequirements_t::default();
            // SAFETY: the `decompress_get_required_alignments` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$decompress_get_required_alignments(
                    options.into(),
                    &raw mut requirements,
                ))?;
            }
            requirements.try_into()
        }

        pub(crate) fn compression_temporary_size(
            num_chunks: usize,
            max_uncompressed_chunk_bytes: usize,
            options: CompressOptions,
            max_total_uncompressed_bytes: usize,
            stream: &Stream,
        ) -> Result<usize> {
            validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                max_uncompressed_chunk_bytes,
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;

            let mut temp_bytes = 0;
            stream.context().bind()?;
            // SAFETY: the `compress_get_temporary_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$compress_get_temporary_size(
                    to_u64(num_chunks, "num_chunks")?,
                    to_u64(max_uncompressed_chunk_bytes, "max_uncompressed_chunk_bytes")?,
                    options.into(),
                    &raw mut temp_bytes,
                    to_u64(max_total_uncompressed_bytes, "max_total_uncompressed_bytes")?,
                    stream.as_raw(),
                ))?;
            }
            to_usize(temp_bytes, "temporary bytes")
        }

        /// Returns the size, in bytes, of the temporary GPU workspace that compressing `input` needs.
        ///
        /// nvCOMP may read the chunk tables on `stream`.
        /// If it does, it synchronizes `stream` before returning, so the result is ready when this returns.
        ///
        /// # Safety
        ///
        /// Every chunk in `input` must be initialized device memory that meets the input alignment from [`BatchedCodec::compression_alignments`](crate::format::BatchedCodec::compression_alignments).
        /// Nothing may write to the chunks or their tables until this returns.
        pub(crate) unsafe fn compression_temporary_size_sync(
            input: &DeviceChunkTable<'_>,
            options: CompressOptions,
            stream: &Stream,
        ) -> Result<usize> {
            let mut temp_bytes = 0;
            stream.sync_scope(|_| {
                Ok((|| -> Result<usize> {
                    input.view().join_regions(stream)?;

                    stream.context().bind()?;
                    validate_chunk_size_limit(
                        "max_uncompressed_chunk_bytes",
                        input.max_chunk_bytes(),
                        COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
                    )?;

                    let input_view = input.view();

                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        try_ffi!(sys::$compress_get_temporary_size_sync(
                            input_view
                                .device_chunk_ptrs()
                                .cast::<*const c_void>()
                                .cast_const(),
                            input_view.device_chunk_bytes().cast(),
                            to_u64(input_view.len(), "num_chunks")?,
                            to_u64(input.max_chunk_bytes(), "max_uncompressed_chunk_bytes")?,
                            options.into(),
                            &raw mut temp_bytes,
                            to_u64(input.total_bytes(), "max_total_uncompressed_bytes")?,
                            stream.as_raw(),
                        ))?;
                    }
                    to_usize(temp_bytes, "temporary bytes")
                })())
            })?
        }

        pub(crate) fn max_output_chunk_size(
            max_uncompressed_chunk_bytes: usize,
            options: CompressOptions,
        ) -> Result<usize> {
            validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                max_uncompressed_chunk_bytes,
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;

            let mut max_compressed_chunk_bytes = 0;
            // SAFETY: the `compress_get_max_output_chunk_size` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$compress_get_max_output_chunk_size(
                    to_u64(max_uncompressed_chunk_bytes, "max_uncompressed_chunk_bytes")?,
                    options.into(),
                    &raw mut max_compressed_chunk_bytes,
                ))?;
            }
            to_usize(max_compressed_chunk_bytes, "maximum compressed chunk bytes")
        }

        pub(crate) fn decompression_temporary_size(
            num_chunks: usize,
            max_uncompressed_chunk_bytes: usize,
            options: DecompressOptions,
            max_total_uncompressed_bytes: usize,
            stream: &Stream,
        ) -> Result<usize> {
            validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                max_uncompressed_chunk_bytes,
                DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;

            let mut temp_bytes = 0;
            stream.context().bind()?;
            // SAFETY: the `decompress_get_temporary_size` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$decompress_get_temporary_size(
                    to_u64(num_chunks, "num_chunks")?,
                    to_u64(max_uncompressed_chunk_bytes, "max_uncompressed_chunk_bytes")?,
                    options.into(),
                    &raw mut temp_bytes,
                    to_u64(max_total_uncompressed_bytes, "max_total_uncompressed_bytes")?,
                    stream.as_raw(),
                ))?;
            }
            to_usize(temp_bytes, "temporary bytes")
        }

        /// Returns the size, in bytes, of the temporary GPU workspace that decompressing `input` needs.
        ///
        /// The chunks decompress to at most `max_uncompressed_chunk_bytes` each and `max_total_uncompressed_bytes` in total.
        /// nvCOMP may read the compressed chunks on `stream`.
        /// If it does, it synchronizes `stream` before returning, so the result is ready when this returns.
        /// For each chunk, nvCOMP writes [`Status::Success`](crate::error::Status::Success) to `statuses` if it can parse the chunk, and an error status otherwise.
        ///
        /// # Safety
        ///
        /// Every chunk in `input` must be initialized device memory that meets the input alignment from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments).
        /// Nothing may write to the chunks, their tables, or `statuses` until this returns.
        pub(crate) unsafe fn decompression_temporary_size_sync(
            input: DeviceChunkTableView<'_>,
            max_uncompressed_chunk_bytes: usize,
            options: DecompressOptions,
            max_total_uncompressed_bytes: usize,
            statuses: &mut DeviceStatuses,
            stream: &Stream,
        ) -> Result<usize> {
            let mut temp_bytes = 0;
            stream.sync_scope(|_| {
                Ok((|| -> Result<usize> {
                    input.join_regions(stream)?;
                    if statuses.context() != stream.context() {
                        return Err(Error::StreamContextMismatch);
                    }

                    stream.context().bind()?;
                    validate_chunk_size_limit(
                        "max_uncompressed_chunk_bytes",
                        max_uncompressed_chunk_bytes,
                        DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
                    )?;
                    statuses.require(input.len())?;

                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        try_ffi!(sys::$decompress_get_temporary_size_sync(
                            input
                                .device_chunk_ptrs()
                                .cast::<*const c_void>()
                                .cast_const(),
                            input.device_chunk_bytes().cast(),
                            to_u64(input.len(), "num_chunks")?,
                            to_u64(max_uncompressed_chunk_bytes, "max_uncompressed_chunk_bytes")?,
                            &raw mut temp_bytes,
                            to_u64(max_total_uncompressed_bytes, "max_total_uncompressed_bytes")?,
                            options.into(),
                            statuses.as_device_ptr().cast(),
                            stream.as_raw(),
                        ))?;
                    }
                    to_usize(temp_bytes, "temporary bytes")
                })())
            })?
        }

        /// Returns the size, in bytes, of the temporary GPU workspace that decompressing `input` into `output` needs.
        ///
        /// This calls [`BatchedCodec::decompression_temporary_size_sync`](crate::format::BatchedCodec::decompression_temporary_size_sync) with the largest and total capacities of `output`'s chunks.
        ///
        /// # Safety
        ///
        /// The requirements of [`BatchedCodec::decompression_temporary_size_sync`](crate::format::BatchedCodec::decompression_temporary_size_sync) apply.
        pub(crate) unsafe fn decompression_temporary_size_for_output_sync(
            input: DeviceChunkTableView<'_>,
            output: &DeviceChunkTableMut<'_>,
            options: DecompressOptions,
            statuses: &mut DeviceStatuses,
            stream: &Stream,
        ) -> Result<usize> {
            validate_same_len("output chunks", input.len(), output.len())?;
            validate_chunk_size_limit(
                "output chunk capacity",
                output.max_buffer_bytes(),
                DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                decompression_temporary_size_sync(
                    input,
                    output.max_buffer_bytes(),
                    options,
                    output.total_buffer_bytes(),
                    statuses,
                    stream,
                )
            }
        }

        /// Arguments for [`compress_raw_async`].
        #[derive(Debug, Clone, Copy)]
        pub struct CompressRawAsync<'a> {
            /// Device array of `num_chunks` pointers to the uncompressed chunks.
            ///
            /// Each chunk must meet the input alignment from [`BatchedCodec::compression_alignments`](crate::format::BatchedCodec::compression_alignments).
            pub device_uncompressed_chunk_ptrs: DevicePtr,
            /// Device array of `num_chunks` `size_t` sizes, in bytes, of the uncompressed chunks.
            pub device_uncompressed_chunk_bytes: DevicePtr,
            /// Size of the largest uncompressed chunk in bytes.
            pub max_uncompressed_chunk_bytes: usize,
            /// Number of chunks, which is also the length of every device array.
            pub num_chunks: usize,
            /// Temporary GPU workspace, aligned to the `temp` alignment from [`BatchedCodec::compression_alignments`](crate::format::BatchedCodec::compression_alignments).
            ///
            /// It may be null when the workspace size is zero.
            pub device_temp: DevicePtr,
            /// Size of `device_temp` in bytes.
            pub temp_bytes: usize,
            /// Device array of `num_chunks` pointers to the compressed output buffers.
            ///
            /// Each buffer must hold [`BatchedCodec::max_output_chunk_size`](crate::format::BatchedCodec::max_output_chunk_size) bytes and meet the output alignment from [`BatchedCodec::compression_alignments`](crate::format::BatchedCodec::compression_alignments).
            pub device_compressed_chunk_ptrs: DevicePtr,
            /// Device array of `num_chunks` `size_t` entries that receive the compressed sizes in bytes.
            pub device_compressed_chunk_bytes: DevicePtr,
            /// Compression options.
            pub options: CompressOptions,
            /// Device array of `num_chunks` statuses that receive each chunk's result, or null to skip status reporting.
            pub device_statuses: DevicePtr,
            /// Stream that runs the work.
            pub stream: &'a Stream,
        }

        /// Enqueues batched compression on `params.stream`.
        ///
        /// nvCOMP does not synchronize the stream.
        /// Synchronize it, or order later work after it on the stream, before using the results.
        ///
        /// # Safety
        ///
        /// Every pointer must reference device-accessible memory that stays valid until the work finishes.
        /// The arrays and buffers must have the sizes and alignments described on [`CompressRawAsync`]'s fields, and the output, size, and status arrays must be writable by the device.
        ///
        /// # Errors
        ///
        /// Returns an error if a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the launch.
        pub unsafe fn compress_raw_async(params: CompressRawAsync<'_>) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$compress_raw_async(
                    params
                        .device_uncompressed_chunk_ptrs
                        .cast::<*const c_void>()
                        .cast_const(),
                    params.device_uncompressed_chunk_bytes.cast(),
                    to_u64(
                        params.max_uncompressed_chunk_bytes,
                        "max_uncompressed_chunk_bytes"
                    )?,
                    to_u64(params.num_chunks, "num_chunks")?,
                    params.device_temp.cast(),
                    to_u64(params.temp_bytes, "temp_bytes")?,
                    params.device_compressed_chunk_ptrs.cast(),
                    params.device_compressed_chunk_bytes.cast(),
                    params.options.into(),
                    params.device_statuses.cast(),
                    params.stream.as_raw(),
                ))
            }
        }

        /// Arguments for [`decompress_size_raw_async`].
        #[derive(Debug, Clone, Copy)]
        pub struct GetDecompressSizeRawAsync<'a> {
            /// Device array of `num_chunks` pointers to the compressed chunks.
            pub device_compressed_chunk_ptrs: DevicePtr,
            /// Device array of `num_chunks` `size_t` sizes, in bytes, of the compressed chunks.
            pub device_compressed_chunk_bytes: DevicePtr,
            /// Device array of `num_chunks` `size_t` entries that receive the uncompressed sizes in bytes.
            pub device_uncompressed_chunk_bytes: DevicePtr,
            /// Number of chunks, which is also the length of every device array.
            pub num_chunks: usize,
            /// Stream that runs the work.
            pub stream: &'a Stream,
        }

        /// Enqueues a query for the uncompressed size of each compressed chunk.
        ///
        /// Use it when the uncompressed sizes are not known in advance.
        /// The query does not decompress, and sizes computed from corrupt input are invalid.
        /// nvCOMP does not synchronize the stream.
        /// Synchronize it, or order later work after it on the stream, before using the results.
        ///
        /// # Safety
        ///
        /// Every pointer must reference device-accessible memory that stays valid until the work finishes.
        /// The compressed chunks must meet the input alignment from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments), and `device_uncompressed_chunk_bytes` must have `num_chunks` entries writable by the device.
        ///
        /// # Errors
        ///
        /// Returns an error if a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the launch.
        pub unsafe fn decompress_size_raw_async(
            params: GetDecompressSizeRawAsync<'_>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$decompress_size_raw_async(
                    params
                        .device_compressed_chunk_ptrs
                        .cast::<*const c_void>()
                        .cast_const(),
                    params.device_compressed_chunk_bytes.cast(),
                    params.device_uncompressed_chunk_bytes.cast(),
                    to_u64(params.num_chunks, "num_chunks")?,
                    params.stream.as_raw(),
                ))
            }
        }

        /// Arguments for [`decompress_raw_async`].
        #[derive(Debug, Clone, Copy)]
        pub struct DecompressRawAsync<'a> {
            /// Device array of `num_chunks` pointers to the compressed chunks.
            ///
            /// Each chunk must meet the input alignment from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments).
            pub device_compressed_chunk_ptrs: DevicePtr,
            /// Device array of `num_chunks` `size_t` sizes, in bytes, of the compressed chunks.
            pub device_compressed_chunk_bytes: DevicePtr,
            /// Device array of `num_chunks` `size_t` capacities, in bytes, of the output buffers.
            ///
            /// A chunk that does not fit its buffer gets [`Status::CannotDecompress`](crate::error::Status::CannotDecompress).
            pub device_uncompressed_buffer_bytes: DevicePtr,
            /// Device array of `num_chunks` `size_t` entries that receive the decompressed sizes in bytes.
            ///
            /// The hardware backend requires it.
            /// With the CUDA backend it may be null if the sizes are not needed.
            pub device_uncompressed_chunk_bytes: DevicePtr,
            /// Number of chunks, which is also the length of every device array.
            pub num_chunks: usize,
            /// Temporary GPU workspace, aligned to the `temp` alignment from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments).
            ///
            /// It may be null when the workspace size is zero.
            pub device_temp: DevicePtr,
            /// Size of `device_temp` in bytes.
            pub temp_bytes: usize,
            /// Device array of `num_chunks` pointers to the output buffers.
            ///
            /// Each buffer must have the capacity given in `device_uncompressed_buffer_bytes` and meet the output alignment from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments).
            pub device_uncompressed_chunk_ptrs: DevicePtr,
            /// Decompression options.
            pub options: DecompressOptions,
            /// Device array of `num_chunks` statuses that receive each chunk's result.
            ///
            /// It may be null only where nvCOMP's documentation for the codec allows it.
            pub device_statuses: DevicePtr,
            /// Stream that runs the work.
            pub stream: &'a Stream,
        }

        /// Enqueues batched decompression on `params.stream`.
        ///
        /// nvCOMP does not synchronize the stream.
        /// Synchronize it, or order later work after it on the stream, before using the results.
        ///
        /// # Safety
        ///
        /// Every compressed chunk must be valid data for this codec and `params.options`.
        /// For most codecs, corrupt input is undefined behavior.
        /// Every pointer must reference device-accessible memory that stays valid until the work finishes.
        /// The arrays and buffers must have the sizes and alignments described on [`DecompressRawAsync`]'s fields, and the output, size, and status arrays must be writable by the device.
        ///
        /// # Errors
        ///
        /// Returns an error if a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the launch.
        pub unsafe fn decompress_raw_async(params: DecompressRawAsync<'_>) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$decompress_raw_async(
                    params
                        .device_compressed_chunk_ptrs
                        .cast::<*const c_void>()
                        .cast_const(),
                    params.device_compressed_chunk_bytes.cast(),
                    params.device_uncompressed_buffer_bytes.cast(),
                    params.device_uncompressed_chunk_bytes.cast(),
                    to_u64(params.num_chunks, "num_chunks")?,
                    params.device_temp.cast(),
                    to_u64(params.temp_bytes, "temp_bytes")?,
                    params.device_uncompressed_chunk_ptrs.cast(),
                    params.options.into(),
                    params.device_statuses.cast(),
                    params.stream.as_raw(),
                ))
            }
        }

        /// Enqueues batched compression of `input` into `output` on `stream`.
        ///
        /// Before launching, this checks that:
        ///
        /// - the tables, `temp`, and `statuses` share the stream's CUDA context, and the tables have the same length;
        /// - no input chunk exceeds [`COMPRESSION_MAX_ALLOWED_CHUNK_SIZE`];
        /// - every output chunk can hold [`BatchedCodec::max_output_chunk_size`](crate::format::BatchedCodec::max_output_chunk_size) bytes;
        /// - the input chunks, output chunks, and `temp` meet the alignments from [`BatchedCodec::compression_alignments`](crate::format::BatchedCodec::compression_alignments);
        /// - `statuses` has one slot per chunk, and `temp` is large enough.
        ///
        /// Call [`AsyncBatchResult::check`](crate::batch::AsyncBatchResult::check) on the result to wait for the work and read the statuses.
        /// Each output chunk then holds compressed data up to its reported size; the rest of its capacity stays uninitialized.
        ///
        /// # Safety
        ///
        /// Every input chunk must be initialized and meet the codec's alignment and data-type requirements.
        /// `input`, `output`, `temp`, `statuses`, and the memory they reference must stay alive until the work finishes, including every replay of a captured graph.
        /// Nothing else may write to them, or read `output`, `temp`, or `statuses`, until then.
        pub(crate) unsafe fn compress_chunks_async<'a>(
            input: &'a DeviceChunkTable<'_>,
            output: &'a mut DeviceChunkTableMut<'_>,
            temp: &'a mut TemporaryBuffer,
            statuses: &'a mut DeviceStatuses,
            options: CompressOptions,
            stream: &'a Stream,
        ) -> Result<AsyncBatchResult<'a>> {
            input.view().validate_context(stream)?;
            output.buffers().validate_context(stream)?;
            if temp.context() != stream.context() || statuses.context() != stream.context() {
                return Err(Error::StreamContextMismatch);
            }

            stream.context().bind()?;
            validate_same_len("output chunks", input.len(), output.len())?;
            validate_chunk_size_limit(
                "max_uncompressed_chunk_bytes",
                input.max_chunk_bytes(),
                COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;
            statuses.require(input.len())?;

            let required_output_bytes = max_output_chunk_size(input.max_chunk_bytes(), options)?;
            if output.min_buffer_bytes() < required_output_bytes {
                return Err(Error::LengthMismatch {
                    name: "output chunk capacity".into(),
                    expected: required_output_bytes,
                    actual: output.min_buffer_bytes(),
                });
            }

            let alignments = compression_alignments(options)?;
            validate_alignment(
                "input chunks",
                input.is_aligned_to(alignments.input),
                alignments.input,
            )?;
            validate_alignment(
                "output chunks",
                output.is_aligned_to(alignments.output),
                alignments.output,
            )?;

            let required_temp_bytes = compression_temporary_size(
                input.len(),
                input.max_chunk_bytes(),
                options,
                input.total_bytes(),
                stream,
            )?;
            temp.require(required_temp_bytes)?;
            validate_alignment(
                "temp buffer",
                temp.is_aligned_to(alignments.temp),
                alignments.temp,
            )?;

            let input_view = input.view();
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                compress_raw_async(CompressRawAsync {
                    device_uncompressed_chunk_ptrs: input_view.device_chunk_ptrs(),
                    device_uncompressed_chunk_bytes: input_view.device_chunk_bytes(),
                    max_uncompressed_chunk_bytes: input.max_chunk_bytes(),
                    num_chunks: input_view.len(),
                    device_temp: temp.as_device_ptr(),
                    temp_bytes: temp.len(),
                    device_compressed_chunk_ptrs: output.buffers().device_chunk_ptrs(),
                    device_compressed_chunk_bytes: output.written_bytes_device_ptr(),
                    options,
                    device_statuses: statuses.as_device_ptr(),
                    stream,
                })?;
            }

            // SAFETY: the caller upholds this function's safety contract.
            Ok(unsafe { AsyncBatchResult::from_submission(statuses, stream) })
        }

        /// Enqueues a query for the uncompressed size of each chunk in `input`, written to `uncompressed_chunk_bytes`.
        ///
        /// The query does not decompress, and sizes computed from corrupt input are invalid.
        /// Synchronize `stream` before reading the sizes.
        ///
        /// # Safety
        ///
        /// Every chunk in `input` must be initialized and meet the input alignment from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments).
        /// `input`, `uncompressed_chunk_bytes`, and the memory they reference must stay alive until the work finishes, including every replay of a captured graph.
        /// Nothing else may write to them, or read `uncompressed_chunk_bytes`, until then.
        pub(crate) unsafe fn decompress_size_chunks_async<S>(
            input: DeviceChunkTableView<'_>,
            uncompressed_chunk_bytes: &mut S,
            stream: &Stream,
        ) -> Result<()>
        where
            S: DeviceOutput<usize> + ?Sized,
        {
            input.validate_context(stream)?;
            if uncompressed_chunk_bytes.context() != stream.context() {
                return Err(Error::StreamContextMismatch);
            }

            stream.context().bind()?;
            validate_same_len(
                "uncompressed chunk bytes",
                input.len(),
                uncompressed_chunk_bytes.len(),
            )?;

            let uncompressed_chunk_bytes =
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { DevicePtr::from_raw(uncompressed_chunk_bytes.as_mut_ptr().cast()) };

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                decompress_size_raw_async(GetDecompressSizeRawAsync {
                    device_compressed_chunk_ptrs: input.device_chunk_ptrs(),
                    device_compressed_chunk_bytes: input.device_chunk_bytes(),
                    device_uncompressed_chunk_bytes: uncompressed_chunk_bytes,
                    num_chunks: input.len(),
                    stream,
                })
            }
        }

        /// Enqueues batched decompression of `input` into `output` on `stream`.
        ///
        /// Before launching, this checks that:
        ///
        /// - the tables, `temp`, and `statuses` share the stream's CUDA context, and the tables have the same length;
        /// - no output chunk exceeds [`DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE`];
        /// - the input chunks, output chunks, and `temp` meet the alignments from [`BatchedCodec::decompression_alignments`](crate::format::BatchedCodec::decompression_alignments);
        /// - `statuses` has one slot per chunk, and `temp` is large enough.
        ///
        /// Call [`AsyncBatchResult::check`](crate::batch::AsyncBatchResult::check) on the result to wait for the work and read the statuses.
        /// Each output chunk then holds decompressed data up to its reported size; the rest of its capacity stays uninitialized.
        ///
        /// # Safety
        ///
        /// Every input chunk must be valid compressed data for this codec and `options`.
        /// For most codecs, corrupt input is undefined behavior.
        /// `input`, `output`, `temp`, `statuses`, and the memory they reference must stay alive until the work finishes, including every replay of a captured graph.
        /// Nothing else may write to them, or read `output`, `temp`, or `statuses`, until then.
        pub(crate) unsafe fn decompress_chunks_async<'a>(
            input: DeviceChunkTableView<'a>,
            output: &'a mut DeviceChunkTableMut<'_>,
            temp: &'a mut TemporaryBuffer,
            statuses: &'a mut DeviceStatuses,
            options: DecompressOptions,
            stream: &'a Stream,
        ) -> Result<AsyncBatchResult<'a>> {
            input.validate_context(stream)?;
            output.buffers().validate_context(stream)?;
            if temp.context() != stream.context() || statuses.context() != stream.context() {
                return Err(Error::StreamContextMismatch);
            }

            stream.context().bind()?;
            validate_same_len("output chunks", input.len(), output.len())?;
            validate_chunk_size_limit(
                "output chunk capacity",
                output.max_buffer_bytes(),
                DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
            )?;
            statuses.require(input.len())?;

            let alignments = decompression_alignments(options)?;
            validate_alignment(
                "input chunks",
                input.is_aligned_to(alignments.input),
                alignments.input,
            )?;
            validate_alignment(
                "output chunks",
                output.is_aligned_to(alignments.output),
                alignments.output,
            )?;

            let required_temp_bytes = decompression_temporary_size(
                input.len(),
                output.max_buffer_bytes(),
                options,
                output.total_buffer_bytes(),
                stream,
            )?;
            temp.require(required_temp_bytes)?;
            validate_alignment(
                "temp buffer",
                temp.is_aligned_to(alignments.temp),
                alignments.temp,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                decompress_raw_async(DecompressRawAsync {
                    device_compressed_chunk_ptrs: input.device_chunk_ptrs(),
                    device_compressed_chunk_bytes: input.device_chunk_bytes(),
                    device_uncompressed_buffer_bytes: output.buffers().device_chunk_bytes(),
                    device_uncompressed_chunk_bytes: output.written_bytes_device_ptr(),
                    num_chunks: input.len(),
                    device_temp: temp.as_device_ptr(),
                    temp_bytes: temp.len(),
                    device_uncompressed_chunk_ptrs: output.buffers().device_chunk_ptrs(),
                    options,
                    device_statuses: statuses.as_device_ptr(),
                    stream,
                })?;
            }

            // SAFETY: the caller upholds this function's safety contract.
            Ok(unsafe { AsyncBatchResult::from_submission(statuses, stream) })
        }
    };
}

pub(crate) use impl_nvcomp_batched_codec_module;
