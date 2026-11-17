//! Codec-independent interface for compressing with any nvCOMP codec.

use std::{marker::PhantomData, mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::{Stream, StreamScope},
    submission::{Access, AccessMode},
    view::{DeviceOutput, DeviceOutputView, DeviceRegionProvider, DeviceSlice},
};

use crate::{
    ans::{CompressOptions as AnsCompressOptions, DecompressOptions as AnsDecompressOptions},
    batch::{
        AsyncBatchResult, DeviceChunkTable, DeviceChunkTableMut, DeviceChunkTableView,
        DeviceStatuses,
    },
    compressed::{CompressedBatch, CompressedBatchParts, CompressedChunk, CompressedChunkParts},
    error::{Error, Result},
    lz4::{CompressOptions as Lz4CompressOptions, DecompressOptions as Lz4DecompressOptions},
    types::AlignmentRequirements,
    utility::validate_same_len,
};

/// nvCOMP codec, implemented by the marker types such as [`Lz4`] and [`Zstd`].
///
/// The trait is sealed; its operations are reached through [`Codec`].
pub trait BatchedCodec: private::Dispatch + Copy + Default + 'static {
    /// Name of the codec, such as `"LZ4"`.
    const NAME: &'static str;
}

/// Compression and decompression options for one codec, with the operations that use them.
///
/// The safe methods, such as [`compress_one`](Self::compress_one) and [`decompress_contiguous`](Self::decompress_contiguous), allocate what they need and wait for the GPU.
/// The unsafe `*_async` methods enqueue batched work over caller-provided chunk tables, workspaces, and status buffers; see the crate's raw execution contract.
/// [`compress_chunks`](Self::compress_chunks) and [`compress_chunks_scoped`](Self::compress_chunks_scoped) compress caller-provided chunk tables safely, waiting or inside a [`StreamScope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Codec<C: BatchedCodec> {
    compress_options: C::CompressOptions,
    decompress_options: C::DecompressOptions,
    _marker: PhantomData<C>,
}

impl<C> Default for Codec<C>
where
    C: BatchedCodec,
{
    fn default() -> Self {
        Self::new(Default::default(), Default::default())
    }
}

impl<C> Codec<C>
where
    C: BatchedCodec,
{
    /// Creates a codec with the given options.
    pub const fn new(
        compress_options: C::CompressOptions,
        decompress_options: C::DecompressOptions,
    ) -> Self {
        Self {
            compress_options,
            decompress_options,
            _marker: PhantomData,
        }
    }

    /// Returns the compression options.
    pub const fn compress_options(self) -> C::CompressOptions {
        self.compress_options
    }

    /// Returns the decompression options.
    pub const fn decompress_options(self) -> C::DecompressOptions {
        self.decompress_options
    }

    /// Returns a copy with `options` as the compression options.
    #[must_use]
    pub const fn with_compress_options(mut self, options: C::CompressOptions) -> Self {
        self.compress_options = options;
        self
    }

    /// Returns a copy with `options` as the decompression options.
    #[must_use]
    pub const fn with_decompress_options(mut self, options: C::DecompressOptions) -> Self {
        self.decompress_options = options;
        self
    }

    /// Returns `decompress_options` updated with the settings from the compression options that decompression needs.
    ///
    /// For example, LZ4 copies the data type and bit shuffle mode.
    #[must_use]
    pub fn decoding_options(self) -> C::DecompressOptions {
        C::decoding_options(self.compress_options, self.decompress_options)
    }

    /// Returns the minimum buffer alignments for compression.
    ///
    /// Buffers aligned beyond these minimums, such as to 16 or 32 bytes, can be faster.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP rejects the options.
    pub fn compression_alignments(self) -> Result<AlignmentRequirements> {
        C::compression_alignments(self.compress_options)
    }

    /// Returns the minimum buffer alignments for decompression.
    ///
    /// Buffers aligned beyond these minimums, such as to 16 or 32 bytes, can be faster.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP rejects the options.
    pub fn decompression_alignments(self) -> Result<AlignmentRequirements> {
        C::decompression_alignments(self.decompress_options)
    }

    /// Returns the largest size, in bytes, that a chunk of at most `max_uncompressed_chunk_bytes` can compress to.
    ///
    /// Each compressed output buffer needs at least this much capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_uncompressed_chunk_bytes` exceeds the codec's limit, it does not fit nvCOMP's integer types, or nvCOMP rejects the query.
    pub fn max_output_chunk_size(self, max_uncompressed_chunk_bytes: usize) -> Result<usize> {
        C::max_output_chunk_size(max_uncompressed_chunk_bytes, self.compress_options)
    }

    /// Returns the workspace size, in bytes, that compressing `num_chunks` chunks needs.
    ///
    /// The chunks hold at most `max_uncompressed_chunk_bytes` each and `max_total_uncompressed_bytes` in total.
    /// nvCOMP enqueues no work on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk size exceeds the codec's limit, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    pub fn compression_workspace_size(
        self,
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        stream: &Stream,
    ) -> Result<usize> {
        C::compression_workspace_size(
            num_chunks,
            max_uncompressed_chunk_bytes,
            self.compress_options,
            max_total_uncompressed_bytes,
            stream,
        )
    }

    /// Returns the workspace size, in bytes, that nvCOMP reports for compressing the chunks of `input`.
    ///
    /// [`compress_chunks_async`](Self::compress_chunks_async) checks its workspace against [`compression_workspace_size`](Self::compression_workspace_size) instead, so size that workspace from that query.
    /// nvCOMP may read the chunks on `stream`; this waits for `stream` before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if the largest chunk exceeds the codec's limit, `input` belongs to another CUDA context, stream capture is active, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
    pub fn compression_workspace_size_for_chunks(
        self,
        input: &DeviceChunkTable<'_>,
        stream: &Stream,
    ) -> Result<usize> {
        C::compression_workspace_size_for_chunks(input, self.compress_options, stream)
    }

    /// Returns the workspace size, in bytes, that decompressing `num_chunks` chunks needs.
    ///
    /// The chunks decompress to at most `max_uncompressed_chunk_bytes` each and `max_total_uncompressed_bytes` in total.
    /// nvCOMP enqueues no work on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk size exceeds the codec's limit, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    pub fn decompression_workspace_size(
        self,
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        stream: &Stream,
    ) -> Result<usize> {
        C::decompression_workspace_size(
            num_chunks,
            max_uncompressed_chunk_bytes,
            self.decompress_options,
            max_total_uncompressed_bytes,
            stream,
        )
    }

    /// Returns the workspace size, in bytes, that nvCOMP reports for decompressing the chunks of `input` into `output`.
    ///
    /// [`decompress_chunks_async`](Self::decompress_chunks_async) checks its workspace against [`decompression_workspace_size`](Self::decompression_workspace_size) instead, so size that workspace from that query.
    /// nvCOMP may read the compressed chunks on `stream`; this waits for `stream` before returning.
    /// For each chunk, nvCOMP writes [`Status::Success`](crate::error::Status::Success) to `statuses` if it can parse the chunk, and an error status otherwise.
    ///
    /// # Safety
    ///
    /// Every chunk of `input` must be initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` and `output` have different lengths, an output chunk is larger than the codec's limit, `statuses` does not have one slot per chunk, an operand belongs to another CUDA context, stream capture is active, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
    pub unsafe fn decompression_workspace_size_for_chunks(
        self,
        input: DeviceChunkTableView<'_>,
        output: &DeviceChunkTableMut<'_>,
        statuses: &mut DeviceStatuses,
        stream: &Stream,
    ) -> Result<usize> {
        // SAFETY: the caller guarantees that every chunk of `input` is initialized, the only requirement of the codec query.
        unsafe {
            C::decompression_workspace_size_for_chunks(
                input,
                output,
                self.decompress_options,
                statuses,
                stream,
            )
        }
    }

    /// Enqueues batched compression of `input` into `output` on `stream`.
    ///
    /// Before launching, this checks that:
    ///
    /// - the tables, `workspace`, and `statuses` share the stream's CUDA context, and the tables have the same length;
    /// - every input chunk size is a multiple of the element size of the compression data type, and no input chunk exceeds the codec's maximum chunk size for compression;
    /// - every output chunk can hold [`max_output_chunk_size`](Self::max_output_chunk_size) bytes;
    /// - the input chunks, output chunks, and `workspace` meet the alignments from [`compression_alignments`](Self::compression_alignments);
    /// - `statuses` has one slot per chunk, and `workspace` holds [`compression_workspace_size`](Self::compression_workspace_size) bytes, where `None` holds none.
    ///
    /// The work is ordered after the operands' producers.
    /// Call [`AsyncBatchResult::check`] on the result to wait for the work and read the statuses.
    /// Each output chunk then holds compressed data up to its reported size; the rest of its capacity stays uninitialized.
    ///
    /// # Safety
    ///
    /// The crate's raw execution contract applies to every operand.
    ///
    /// # Errors
    ///
    /// Returns an error if a check fails, the CUDA context cannot be bound, or nvCOMP rejects the launch.
    pub unsafe fn compress_chunks_async<'a>(
        self,
        input: &'a DeviceChunkTable<'_>,
        output: &'a mut DeviceChunkTableMut<'_>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        statuses: &'a mut DeviceStatuses,
        stream: &'a Stream,
    ) -> Result<AsyncBatchResult<'a>> {
        input.require_element_size(C::compression_element_size(self.compress_options))?;
        // SAFETY: the table holds initialized chunks of whole elements, and the caller keeps every operand alive and unaliased until the work finishes, as the codec function requires.
        unsafe {
            C::compress_chunks_async(
                input,
                output,
                workspace,
                statuses,
                self.compress_options,
                stream,
            )
        }
    }

    /// Enqueues batched compression of `input` into `output` on the scope's stream.
    ///
    /// This makes the checks of [`compress_chunks_async`](Self::compress_chunks_async) and declares the reads of `input` and the writes of `output`, `workspace`, and `statuses`, so later scoped work on the same memory waits for it.
    /// The scope keeps every operand borrowed until its stream drains.
    /// Call [`AsyncBatchResult::check`] on the result inside the scope to wait for the work and read the statuses.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails or [`compress_chunks_async`](Self::compress_chunks_async) fails.
    pub fn compress_chunks_scoped<'scope, 'env>(
        self,
        input: &'env DeviceChunkTable<'_>,
        output: &'env mut DeviceChunkTableMut<'_>,
        workspace: Option<DeviceOutputView<'env, u8>>,
        statuses: &'env mut DeviceStatuses,
        scope: &'scope StreamScope<'scope, 'env>,
    ) -> Result<AsyncBatchResult<'scope>> {
        let mut accesses = input.read_accesses();
        accesses.extend(output.write_accesses());
        accesses.extend(
            workspace
                .as_ref()
                .map(|workspace| Access::new(workspace.region(), AccessMode::Write)),
        );
        accesses.push(Access::new(statuses.region(), AccessMode::Write));
        // SAFETY: the compression touches only the declared operands on the scope's stream, and every operand is borrowed for `'env`, which the scope keeps alive and unaliased until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.compress_chunks_async(input, output, workspace, statuses, scope.stream()))
            })
        }?
    }

    /// Compresses the chunks of `input` into the chunks of `output` in one batch and waits for the result.
    ///
    /// This allocates the workspace and status buffers, enqueues the work on `stream` with the checks of [`compress_chunks_async`](Self::compress_chunks_async), waits for it, and checks the statuses.
    /// Every output chunk must hold [`max_output_chunk_size`](Self::max_output_chunk_size) bytes for the largest input chunk.
    /// Returns the compressed size of each chunk; each output chunk holds compressed data up to its size, and the rest of its capacity stays uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if a check of [`compress_chunks_async`](Self::compress_chunks_async) fails, stream capture is active, an allocation fails, nvCOMP rejects a query or the launch, or a chunk reports an error status.
    pub fn compress_chunks(
        self,
        input: &DeviceChunkTable<'_>,
        output: &mut DeviceChunkTableMut<'_>,
        stream: &Stream,
    ) -> Result<Vec<usize>> {
        stream.ensure_not_capturing()?;
        input.view().validate_context(stream)?;
        output.buffers().validate_context(stream)?;
        validate_same_len(input.len(), output.len(), "output chunks")?;
        if input.is_empty() {
            return Ok(Vec::new());
        }

        stream.context().bind()?;
        let workspace_bytes = self.compression_workspace_size(
            input.len(),
            input.max_chunk_bytes(),
            input.total_bytes(),
            stream,
        )?;
        let mut workspace = allocate_workspace(stream.context(), workspace_bytes)?;
        let mut statuses = DeviceStatuses::create(stream.context(), input.len())?;
        stream.complete(|| -> Result<()> {
            // SAFETY: `complete` waits for the stream before the tables, workspace, and statuses are released, on error too.
            unsafe {
                self.compress_chunks_async(
                    input,
                    output,
                    workspace.as_mut().map(|memory| memory.output()),
                    &mut statuses,
                    stream,
                )
            }?
            .check()?;
            Ok(())
        })?;

        // SAFETY: the completed work above wrote every chunk size.
        unsafe { output.copy_written_bytes(stream) }
    }

    /// Enqueues a query for the uncompressed size of each chunk in `input`, written to `uncompressed_chunk_bytes`.
    ///
    /// The query does not decompress, and sizes computed from corrupt input are invalid.
    /// Wait for `stream` before reading the sizes.
    ///
    /// # Safety
    ///
    /// Every chunk of `input` must be initialized, and the crate's raw execution contract applies to every operand.
    ///
    /// # Errors
    ///
    /// Returns an error if `uncompressed_chunk_bytes` does not have one entry per chunk, an operand belongs to another CUDA context, the CUDA context cannot be bound, or nvCOMP rejects the launch.
    pub unsafe fn decompressed_sizes_async<S>(
        self,
        input: DeviceChunkTableView<'_>,
        uncompressed_chunk_bytes: &mut S,
        stream: &Stream,
    ) -> Result<()>
    where
        S: DeviceOutput<usize> + ?Sized,
    {
        // SAFETY: the caller guarantees valid input chunks and keeps every operand alive and unaliased until the work finishes, as the codec function requires.
        unsafe { C::decompressed_sizes_async(input, uncompressed_chunk_bytes, stream) }
    }

    /// Enqueues batched decompression of `input` into `output` on `stream`.
    ///
    /// Before launching, this checks that:
    ///
    /// - the tables, `workspace`, and `statuses` share the stream's CUDA context, and the tables have the same length;
    /// - no output chunk exceeds the codec's maximum chunk size for decompression;
    /// - the input chunks, output chunks, and `workspace` meet the alignments from [`decompression_alignments`](Self::decompression_alignments);
    /// - `statuses` has one slot per chunk, and `workspace` holds [`decompression_workspace_size`](Self::decompression_workspace_size) bytes, where `None` holds none.
    ///
    /// The work is ordered after the operands' producers.
    /// Call [`AsyncBatchResult::check`] on the result to wait for the work and read the statuses.
    /// Each output chunk then holds decompressed data up to its reported size; the rest of its capacity stays uninitialized.
    ///
    /// # Safety
    ///
    /// Every input chunk must be initialized, valid compressed data for this codec and its decompression options.
    /// nvCOMP documents corrupt, truncated, or mismatched input to batched decompression as undefined behavior, which can include writes past the output capacities, and validates only part of it, so no check here can make decompressing unknown bytes safe.
    /// The crate's raw execution contract applies to every operand.
    ///
    /// # Errors
    ///
    /// Returns an error if a check fails, the CUDA context cannot be bound, or nvCOMP rejects the launch.
    pub unsafe fn decompress_chunks_async<'a>(
        self,
        input: DeviceChunkTableView<'a>,
        output: &'a mut DeviceChunkTableMut<'_>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        statuses: &'a mut DeviceStatuses,
        stream: &'a Stream,
    ) -> Result<AsyncBatchResult<'a>> {
        // SAFETY: the caller guarantees valid input chunks and keeps every operand alive and unaliased until the work finishes, as the codec function requires.
        unsafe {
            C::decompress_chunks_async(
                input,
                output,
                workspace,
                statuses,
                self.decompress_options,
                stream,
            )
        }
    }

    /// Compresses all of `input` as one chunk and waits for the result.
    ///
    /// This allocates the output, workspace, and status buffers, enqueues the work on `stream`, waits for it, and checks the status.
    /// The options from [`decoding_options`](Self::decoding_options) are stored with the result for decompression.
    /// An empty `input` gives an empty chunk without calling nvCOMP.
    ///
    /// # Errors
    ///
    /// Returns an error if the input size is not a multiple of the codec's element size, stream capture is active, the buffers and `stream` belong to different CUDA contexts, an allocation fails, nvCOMP rejects a query or the launch, or a chunk reports an error status.
    pub fn compress_one<B>(self, input: &B, stream: &Stream) -> Result<CompressedChunk<C>>
    where
        B: DeviceSlice<u8> + ?Sized,
    {
        stream.ensure_not_capturing()?;
        if input.context() != stream.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        stream.context().bind()?;
        let decoding_options = self.decoding_options();
        if input.is_empty() {
            // SAFETY: an empty allocation has no bytes to initialize.
            let memory =
                unsafe { DeviceMemory::<u8>::create_uninit(stream.context(), 0)?.assume_init() };
            let parts = CompressedChunkParts {
                memory,
                uncompressed_bytes: 0,
                decoding_options,
            };
            // SAFETY: the decompressing calls return an empty chunk with an uncompressed size of 0 without passing it to nvCOMP.
            return Ok(unsafe { CompressedChunk::from_parts(parts) });
        }

        let input_chunks = DeviceChunkTable::from_contiguous(stream.context(), input)?;
        let max_compressed_bytes = self.max_output_chunk_size(input.len())?;
        let mut memory = DeviceMemory::<u8>::create_uninit(stream.context(), max_compressed_bytes)?;
        let mut output_chunks =
            DeviceChunkTableMut::from_contiguous(stream.context(), &mut memory)?;
        let compressed_sizes = self.compress_chunks(&input_chunks, &mut output_chunks, stream)?;
        drop(output_chunks);

        // SAFETY: `compress_chunks` waited for the work, and its checked status shows that the first `compressed_sizes[0]` bytes hold compressed data for this codec.
        let memory = unsafe { memory.assume_init_prefix(compressed_sizes[0]) }?;
        let parts = CompressedChunkParts {
            memory,
            uncompressed_bytes: input.len(),
            decoding_options,
        };
        // SAFETY: the bytes above hold compressed data for this codec and its decoding options, which decompresses to `input`.
        Ok(unsafe { CompressedChunk::from_parts(parts) })
    }

    /// Decompresses `compressed` into `output`, waits for the result, and returns the number of bytes written.
    ///
    /// This allocates the workspace and status buffers, enqueues the work on `stream`, waits for it, and checks the status.
    /// The chunk's stored decoding options are used, not this codec's decompression options.
    /// An empty chunk with an uncompressed size of 0 writes nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is smaller than the uncompressed size, the number of bytes written differs from it, stream capture is active, the buffers and `stream` belong to different CUDA contexts, an allocation fails, nvCOMP rejects a query or the launch, or a chunk reports an error status.
    pub fn decompress_one<B>(
        self,
        compressed: &CompressedChunk<C>,
        output: &mut B,
        stream: &Stream,
    ) -> Result<usize>
    where
        B: DeviceOutput<u8> + ?Sized,
    {
        stream.ensure_not_capturing()?;
        if compressed.memory().context() != stream.context() || output.context() != stream.context()
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        stream.context().bind()?;
        let codec = self.with_decompress_options(compressed.decoding_options);
        if output.len() < compressed.uncompressed_len() {
            return Err(Error::BufferTooSmall {
                name: "output buffer".into(),
                required: compressed.uncompressed_len(),
                actual: output.len(),
            });
        }

        if compressed.is_empty() {
            if compressed.uncompressed_len() == 0 {
                return Ok(0);
            }

            return Err(Error::Empty {
                name: "compressed chunk bytes".into(),
            });
        }

        let input_chunks = DeviceChunkTable::from_contiguous_chunk_sizes(
            stream.context(),
            compressed.memory(),
            &[compressed.len()],
        )?;
        // The output chunk covers only the uncompressed size, so a larger `output` stays within the codec's chunk limit.
        let mut output_chunks = DeviceChunkTableMut::from_contiguous_chunk_sizes(
            stream.context(),
            output,
            &[compressed.uncompressed_len()],
        )?;
        let mut statuses = DeviceStatuses::create(stream.context(), input_chunks.len())?;
        let workspace_bytes = codec.decompression_workspace_size(
            input_chunks.len(),
            output_chunks.max_buffer_bytes(),
            output_chunks.total_buffer_bytes(),
            stream,
        )?;
        let mut workspace = allocate_workspace(stream.context(), workspace_bytes)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: a `CompressedChunk` holds completed compressed data for this codec, and `complete` waits for the stream before the tables, workspace, and statuses are released, on error too.
            unsafe {
                codec.decompress_chunks_async(
                    input_chunks.view(),
                    &mut output_chunks,
                    workspace.as_mut().map(|memory| memory.output()),
                    &mut statuses,
                    stream,
                )
            }?
            .check()?;
            Ok(())
        })?;

        // SAFETY: the completed work above wrote every chunk size.
        let output_bytes = unsafe { output_chunks.copy_written_bytes(stream) }?[0];
        if output_bytes != compressed.uncompressed_len() {
            return Err(Error::LengthMismatch {
                name: "decompressed chunk bytes".into(),
                expected: compressed.uncompressed_len(),
                actual: output_bytes,
            });
        }

        Ok(output_bytes)
    }

    /// Compresses `input` in chunks of `chunk_bytes` bytes and waits for the result.
    ///
    /// The chunks are compressed in one batch into a [`CompressedBatch`] whose slots each hold [`max_output_chunk_size`](Self::max_output_chunk_size) bytes.
    /// The options from [`decoding_options`](Self::decoding_options) are stored with the result for decompression.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunk_bytes` is zero, a chunk size is not a multiple of the codec's element size, stream capture is active, the buffers and `stream` belong to different CUDA contexts, an allocation fails, nvCOMP rejects a query or the launch, or a chunk reports an error status.
    pub fn compress_contiguous<B>(
        self,
        input: &B,
        chunk_bytes: usize,
        stream: &Stream,
    ) -> Result<CompressedBatch<C>>
    where
        B: DeviceSlice<u8> + ?Sized,
    {
        stream.ensure_not_capturing()?;
        if input.context() != stream.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        stream.context().bind()?;
        let decoding_options = self.decoding_options();
        let input_chunks =
            DeviceChunkTable::from_contiguous_chunks(stream.context(), input, chunk_bytes)?;
        if input_chunks.is_empty() {
            return CompressedBatch::create_trusted(CompressedBatchParts {
                memory: DeviceMemory::<u8>::create_uninit(stream.context(), 0)?,
                stride_bytes: 0,
                chunk_bytes: vec![],
                uncompressed_chunk_bytes: vec![],
                decoding_options,
            });
        }

        let stride_bytes = self.max_output_chunk_size(input_chunks.max_chunk_bytes())?;
        let allocation_bytes = stride_bytes
            .checked_mul(input_chunks.len())
            .ok_or_else(|| Error::out_of_range("compressed batch bytes"))?;
        let mut memory = DeviceMemory::<u8>::create_uninit(stream.context(), allocation_bytes)?;
        let mut output_chunks = DeviceChunkTableMut::from_contiguous_chunks(
            stream.context(),
            &mut memory,
            stride_bytes,
        )?;
        let compressed_chunk_bytes =
            self.compress_chunks(&input_chunks, &mut output_chunks, stream)?;
        drop(output_chunks);
        let uncompressed_chunk_bytes = contiguous_chunk_sizes(input.len(), chunk_bytes)?;

        CompressedBatch::create_trusted(CompressedBatchParts {
            memory,
            stride_bytes,
            chunk_bytes: compressed_chunk_bytes,
            uncompressed_chunk_bytes,
            decoding_options,
        })
    }

    /// Decompresses every chunk of `compressed` into consecutive ranges of `output`, waits for the result, and returns the number of bytes written.
    ///
    /// The batch's stored decoding options are used, not this codec's decompression options.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is smaller than the total uncompressed size, a chunk decompresses to a different size than recorded, stream capture is active, the buffers and `stream` belong to different CUDA contexts, an allocation fails, nvCOMP rejects a query or the launch, or a chunk reports an error status.
    pub fn decompress_contiguous<B>(
        self,
        compressed: &CompressedBatch<C>,
        output: &mut B,
        stream: &Stream,
    ) -> Result<usize>
    where
        B: DeviceOutput<u8> + ?Sized,
    {
        stream.ensure_not_capturing()?;
        if compressed.memory().context() != stream.context() || output.context() != stream.context()
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        stream.context().bind()?;
        let codec = self.with_decompress_options(compressed.decoding_options);
        if output.len() < compressed.total_uncompressed_len() {
            return Err(Error::BufferTooSmall {
                name: "output buffer".into(),
                required: compressed.total_uncompressed_len(),
                actual: output.len(),
            });
        }

        if compressed.is_empty() {
            return Ok(0);
        }

        let input_chunks = compressed.chunks()?;
        let mut output_chunks = DeviceChunkTableMut::from_contiguous_chunk_sizes(
            stream.context(),
            output,
            compressed.uncompressed_chunk_bytes(),
        )?;
        let mut statuses = DeviceStatuses::create(stream.context(), input_chunks.len())?;
        let workspace_bytes = codec.decompression_workspace_size(
            input_chunks.len(),
            output_chunks.max_buffer_bytes(),
            output_chunks.total_buffer_bytes(),
            stream,
        )?;
        let mut workspace = allocate_workspace(stream.context(), workspace_bytes)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: a `CompressedBatch` holds completed compressed data for this codec, and `complete` waits for the stream before the tables, workspace, and statuses are released, on error too.
            unsafe {
                codec.decompress_chunks_async(
                    input_chunks.view(),
                    &mut output_chunks,
                    workspace.as_mut().map(|memory| memory.output()),
                    &mut statuses,
                    stream,
                )
            }?
            .check()?;
            Ok(())
        })?;

        // SAFETY: the completed work above wrote every chunk size.
        let output_bytes = unsafe { output_chunks.copy_written_bytes(stream) }?;
        for (index, (actual, expected)) in output_bytes
            .iter()
            .copied()
            .zip(compressed.uncompressed_chunk_bytes().iter().copied())
            .enumerate()
        {
            if actual != expected {
                return Err(Error::ChunkLengthMismatch {
                    index,
                    name: "decompressed chunk bytes".into(),
                    expected,
                    actual,
                });
            }
        }

        Ok(compressed.total_uncompressed_len())
    }
}

fn allocate_workspace(
    context: &Arc<CudaContext>,
    bytes: usize,
) -> Result<Option<DeviceMemory<MaybeUninit<u8>>>> {
    if bytes == 0 {
        return Ok(None);
    }

    Ok(Some(DeviceMemory::<u8>::create_uninit(context, bytes)?))
}

fn contiguous_chunk_sizes(bytes: usize, chunk_bytes: usize) -> Result<Vec<usize>> {
    if chunk_bytes == 0 {
        return Err(Error::NotPositive {
            name: "chunk bytes".into(),
        });
    }

    let chunks = bytes.div_ceil(chunk_bytes);
    let mut sizes = Vec::with_capacity(chunks);
    for index in 0..chunks {
        let offset = index * chunk_bytes;
        sizes.push(chunk_bytes.min(bytes - offset));
    }

    Ok(sizes)
}

macro_rules! impl_batched_codec {
    ($marker:ident, $module:ident, $name:literal $(, $decoding_options:expr)?) => {
        #[doc = concat!("Marker type for the ", $name, " codec, used as the type parameter of [`Codec`].")]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $marker;

        impl BatchedCodec for $marker {
            const NAME: &'static str = $name;
        }

        impl private::Dispatch for $marker {
            type CompressOptions = $crate::$module::CompressOptions;
            type DecompressOptions = $crate::$module::DecompressOptions;

            fn compression_element_size(options: Self::CompressOptions) -> usize {
                impl_batched_codec!(@element_size $module, options)
            }

            fn decoding_options(
                compress_options: Self::CompressOptions,
                decompress_options: Self::DecompressOptions,
            ) -> Self::DecompressOptions {
                let _ = compress_options;
                impl_batched_codec!(@decoding compress_options, decompress_options $(, $decoding_options)?)
            }

            fn compression_alignments(
                options: Self::CompressOptions,
            ) -> Result<AlignmentRequirements> {
                $crate::$module::compression_alignments(options)
            }

            fn decompression_alignments(
                options: Self::DecompressOptions,
            ) -> Result<AlignmentRequirements> {
                $crate::$module::decompression_alignments(options)
            }

            fn compression_workspace_size(
                num_chunks: usize,
                max_uncompressed_chunk_bytes: usize,
                options: Self::CompressOptions,
                max_total_uncompressed_bytes: usize,
                stream: &Stream,
            ) -> Result<usize> {
                $crate::$module::compression_workspace_size(
                    num_chunks,
                    max_uncompressed_chunk_bytes,
                    options,
                    max_total_uncompressed_bytes,
                    stream,
                )
            }

            fn compression_workspace_size_for_chunks(
                input: &DeviceChunkTable<'_>,
                options: Self::CompressOptions,
                stream: &Stream,
            ) -> Result<usize> {
                $crate::$module::compression_workspace_size_for_chunks(input, options, stream)
            }

            fn max_output_chunk_size(
                max_uncompressed_chunk_bytes: usize,
                options: Self::CompressOptions,
            ) -> Result<usize> {
                $crate::$module::max_output_chunk_size(max_uncompressed_chunk_bytes, options)
            }

            fn decompression_workspace_size(
                num_chunks: usize,
                max_uncompressed_chunk_bytes: usize,
                options: Self::DecompressOptions,
                max_total_uncompressed_bytes: usize,
                stream: &Stream,
            ) -> Result<usize> {
                $crate::$module::decompression_workspace_size(
                    num_chunks,
                    max_uncompressed_chunk_bytes,
                    options,
                    max_total_uncompressed_bytes,
                    stream,
                )
            }

            unsafe fn decompression_workspace_size_for_chunks(
                input: DeviceChunkTableView<'_>,
                output: &DeviceChunkTableMut<'_>,
                options: Self::DecompressOptions,
                statuses: &mut DeviceStatuses,
                stream: &Stream,
            ) -> Result<usize> {
                // SAFETY: the caller guarantees that every chunk of `input` is initialized, the only requirement of the codec query.
                unsafe {
                    $crate::$module::decompression_workspace_size_for_chunks(
                        input, output, options, statuses, stream,
                    )
                }
            }

            unsafe fn compress_chunks_async<'a>(
                input: &'a DeviceChunkTable<'_>,
                output: &'a mut DeviceChunkTableMut<'_>,
                workspace: Option<DeviceOutputView<'_, u8>>,
                statuses: &'a mut DeviceStatuses,
                options: Self::CompressOptions,
                stream: &'a Stream,
            ) -> Result<AsyncBatchResult<'a>> {
                // SAFETY: the caller guarantees valid input chunks and keeps every operand alive and unaliased until the work finishes, as the codec function requires.
                unsafe {
                    $crate::$module::compress_chunks_async(
                        input, output, workspace, statuses, options, stream,
                    )
                }
            }

            unsafe fn decompressed_sizes_async<S>(
                input: DeviceChunkTableView<'_>,
                uncompressed_chunk_bytes: &mut S,
                stream: &Stream,
            ) -> Result<()>
            where
                S: DeviceOutput<usize> + ?Sized,
            {
                // SAFETY: the caller guarantees valid input chunks and keeps every operand alive and unaliased until the work finishes, as the codec function requires.
                unsafe {
                    $crate::$module::decompressed_sizes_async(
                        input,
                        uncompressed_chunk_bytes,
                        stream,
                    )
                }
            }

            unsafe fn decompress_chunks_async<'a>(
                input: DeviceChunkTableView<'a>,
                output: &'a mut DeviceChunkTableMut<'_>,
                workspace: Option<DeviceOutputView<'_, u8>>,
                statuses: &'a mut DeviceStatuses,
                options: Self::DecompressOptions,
                stream: &'a Stream,
            ) -> Result<AsyncBatchResult<'a>> {
                // SAFETY: the caller guarantees valid input chunks and keeps every operand alive and unaliased until the work finishes, as the codec function requires.
                unsafe {
                    $crate::$module::decompress_chunks_async(
                        input, output, workspace, statuses, options, stream,
                    )
                }
            }
        }
    };
    (@element_size ans, $options:ident) => { $options.data_type.size_bytes() };
    (@element_size bitcomp, $options:ident) => { $options.data_type.size_bytes() };
    (@element_size cascaded, $options:ident) => { $options.common.data_type.size_bytes() };
    (@element_size lz4, $options:ident) => { $options.data_type.size_bytes() };
    (@element_size $other:ident, $options:ident) => { { let _ = $options; 1 } };
    (@decoding $compress:ident, $decompress:ident) => {
        $decompress
    };
    (@decoding $compress:ident, $decompress:ident, $conversion:expr) => {
        $conversion($compress, $decompress)
    };
}

impl_batched_codec!(
    Ans,
    ans,
    "ANS",
    |compress: AnsCompressOptions, mut decompress: AnsDecompressOptions| {
        decompress.data_type = compress.data_type;
        decompress.max_sub_chunk_count = compress.max_sub_chunk_count;
        decompress.states_per_lane = compress.states_per_lane;
        decompress
    }
);

impl_batched_codec!(Bitcomp, bitcomp, "Bitcomp");

impl_batched_codec!(Cascaded, cascaded, "Cascaded");

impl_batched_codec!(Deflate, deflate, "Deflate");

impl_batched_codec!(Gdeflate, gdeflate, "GDeflate");

impl_batched_codec!(Gzip, gzip, "Gzip");

impl_batched_codec!(
    Lz4,
    lz4,
    "LZ4",
    |compress: Lz4CompressOptions, mut decompress: Lz4DecompressOptions| {
        decompress.data_type = compress.data_type;
        decompress.bitshuffle_mode = compress.bitshuffle_mode;
        decompress
    }
);

impl_batched_codec!(Snappy, snappy, "Snappy");

impl_batched_codec!(Zstd, zstd, "Zstd");

mod private {
    use singe_cuda::{
        stream::Stream,
        view::{DeviceOutput, DeviceOutputView},
    };

    use crate::{
        batch::{
            AsyncBatchResult, DeviceChunkTable, DeviceChunkTableMut, DeviceChunkTableView,
            DeviceStatuses,
        },
        error::Result,
        types::AlignmentRequirements,
    };

    // Seals `BatchedCodec` and dispatches `Codec` to the codec modules.
    // The unsafe functions carry the contracts of the `Codec` methods of the same names.
    #[expect(
        unnameable_types,
        reason = "sealed supertrait of BatchedCodec; users cannot name or implement it"
    )]
    pub trait Dispatch {
        type CompressOptions: Copy + Default;

        type DecompressOptions: Copy + Default;

        fn compression_element_size(options: Self::CompressOptions) -> usize;

        fn decoding_options(
            compress_options: Self::CompressOptions,
            decompress_options: Self::DecompressOptions,
        ) -> Self::DecompressOptions;

        fn compression_alignments(options: Self::CompressOptions) -> Result<AlignmentRequirements>;

        fn decompression_alignments(
            options: Self::DecompressOptions,
        ) -> Result<AlignmentRequirements>;

        fn compression_workspace_size(
            num_chunks: usize,
            max_uncompressed_chunk_bytes: usize,
            options: Self::CompressOptions,
            max_total_uncompressed_bytes: usize,
            stream: &Stream,
        ) -> Result<usize>;

        fn compression_workspace_size_for_chunks(
            input: &DeviceChunkTable<'_>,
            options: Self::CompressOptions,
            stream: &Stream,
        ) -> Result<usize>;

        fn max_output_chunk_size(
            max_uncompressed_chunk_bytes: usize,
            options: Self::CompressOptions,
        ) -> Result<usize>;

        fn decompression_workspace_size(
            num_chunks: usize,
            max_uncompressed_chunk_bytes: usize,
            options: Self::DecompressOptions,
            max_total_uncompressed_bytes: usize,
            stream: &Stream,
        ) -> Result<usize>;

        unsafe fn decompression_workspace_size_for_chunks(
            input: DeviceChunkTableView<'_>,
            output: &DeviceChunkTableMut<'_>,
            options: Self::DecompressOptions,
            statuses: &mut DeviceStatuses,
            stream: &Stream,
        ) -> Result<usize>;

        unsafe fn compress_chunks_async<'a>(
            input: &'a DeviceChunkTable<'_>,
            output: &'a mut DeviceChunkTableMut<'_>,
            workspace: Option<DeviceOutputView<'_, u8>>,
            statuses: &'a mut DeviceStatuses,
            options: Self::CompressOptions,
            stream: &'a Stream,
        ) -> Result<AsyncBatchResult<'a>>;

        unsafe fn decompressed_sizes_async<S>(
            input: DeviceChunkTableView<'_>,
            uncompressed_chunk_bytes: &mut S,
            stream: &Stream,
        ) -> Result<()>
        where
            S: DeviceOutput<usize> + ?Sized;

        unsafe fn decompress_chunks_async<'a>(
            input: DeviceChunkTableView<'a>,
            output: &'a mut DeviceChunkTableMut<'_>,
            workspace: Option<DeviceOutputView<'_, u8>>,
            statuses: &'a mut DeviceStatuses,
            options: Self::DecompressOptions,
            stream: &'a Stream,
        ) -> Result<AsyncBatchResult<'a>>;
    }
}
