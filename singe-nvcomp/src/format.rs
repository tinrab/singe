//! Codec-independent interface for compressing with any nvCOMP codec.

use std::marker::PhantomData;

use singe_cuda::{
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceOutput, DeviceSlice},
};

use crate::{
    ans::{CompressOptions as AnsCompressOptions, DecompressOptions as AnsDecompressOptions},
    batch::{
        AsyncBatchResult, DeviceChunkTable, DeviceChunkTableMut, DeviceChunkTableView,
        DeviceStatuses, OwnedAsyncBatchResult,
    },
    compressed::{CompressedBatch, CompressedBatchParts, CompressedChunk},
    error::{Error, Result},
    lz4::{CompressOptions as Lz4CompressOptions, DecompressOptions as Lz4DecompressOptions},
    types::AlignmentRequirements,
    workspace::TemporaryBuffer,
};

/// nvCOMP codec, implemented by the marker types such as [`Lz4`] and [`Zstd`].
///
/// The trait is sealed.
pub trait BatchedCodec: private::Sealed + Copy + Default + 'static {
    /// Returns the element size, in bytes, that chunk sizes must be a multiple of for compression with `options`.
    fn compression_element_size(options: Self::CompressOptions) -> usize;

    /// Compression options.
    type CompressOptions: Copy + Default;

    /// Decompression options.
    type DecompressOptions: Copy + Default;

    /// Name of the codec, such as `"LZ4"`.
    const NAME: &'static str;

    /// Returns `decompress_options` updated with the settings from `compress_options` that decompression needs.
    ///
    /// For example, LZ4 copies the data type and bit shuffle mode.
    fn decoding_options(
        _compress_options: Self::CompressOptions,
        decompress_options: Self::DecompressOptions,
    ) -> Self::DecompressOptions {
        decompress_options
    }

    /// Returns the minimum buffer alignments for compression with `options`.
    ///
    /// Buffers aligned beyond these minimums, such as to 16 or 32 bytes, can be faster.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP rejects the options.
    fn compression_alignments(options: Self::CompressOptions) -> Result<AlignmentRequirements>;

    /// Returns the minimum buffer alignments for decompression with `options`.
    ///
    /// Buffers aligned beyond these minimums, such as to 16 or 32 bytes, can be faster.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP rejects the options.
    fn decompression_alignments(options: Self::DecompressOptions) -> Result<AlignmentRequirements>;

    /// Returns the size, in bytes, of the temporary GPU workspace that compression needs.
    ///
    /// The batch has `num_chunks` chunks of at most `max_uncompressed_chunk_bytes` each and at most `max_total_uncompressed_bytes` in total.
    /// nvCOMP enqueues no work on `stream`, so the result is ready when this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_uncompressed_chunk_bytes` exceeds the codec's maximum chunk size for compression, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    fn compression_temporary_size(
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        options: Self::CompressOptions,
        max_total_uncompressed_bytes: usize,
        stream: &Stream,
    ) -> Result<usize>;

    /// Returns the size, in bytes, of the temporary GPU workspace that compressing `input` needs.
    ///
    /// nvCOMP may read the chunk tables on `stream`.
    /// If it does, it synchronizes `stream` before returning, so the result is ready when this returns.
    ///
    /// # Safety
    ///
    /// Every chunk in `input` must be initialized device memory that meets the input alignment from [`compression_alignments`](Self::compression_alignments).
    /// Nothing may write to the chunks or their tables until this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if the largest chunk exceeds the codec's maximum chunk size for compression, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    unsafe fn compression_temporary_size_sync(
        input: &DeviceChunkTable<'_>,
        options: Self::CompressOptions,
        stream: &Stream,
    ) -> Result<usize>;

    /// Returns the largest size, in bytes, that a chunk of at most `max_uncompressed_chunk_bytes` can compress to.
    ///
    /// Each compressed output buffer needs at least this much capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_uncompressed_chunk_bytes` exceeds the codec's maximum chunk size for compression, it does not fit nvCOMP's integer types, or nvCOMP rejects the query.
    fn max_output_chunk_size(
        max_uncompressed_chunk_bytes: usize,
        options: Self::CompressOptions,
    ) -> Result<usize>;

    /// Returns the size, in bytes, of the temporary GPU workspace that decompression needs.
    ///
    /// The batch has `num_chunks` chunks that decompress to at most `max_uncompressed_chunk_bytes` each and `max_total_uncompressed_bytes` in total.
    /// nvCOMP enqueues no work on `stream`, so the result is ready when this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_uncompressed_chunk_bytes` exceeds the codec's maximum chunk size for decompression, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    fn decompression_temporary_size(
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        options: Self::DecompressOptions,
        max_total_uncompressed_bytes: usize,
        stream: &Stream,
    ) -> Result<usize>;

    /// Returns the size, in bytes, of the temporary GPU workspace that decompressing `input` needs.
    ///
    /// The chunks decompress to at most `max_uncompressed_chunk_bytes` each and `max_total_uncompressed_bytes` in total.
    /// nvCOMP may read the compressed chunks on `stream`.
    /// If it does, it synchronizes `stream` before returning, so the result is ready when this returns.
    /// For each chunk, nvCOMP writes [`Status::Success`](crate::error::Status::Success) to `statuses` if it can parse the chunk, and an error status otherwise.
    ///
    /// # Safety
    ///
    /// Every chunk in `input` must be initialized device memory that meets the input alignment from [`decompression_alignments`](Self::decompression_alignments).
    /// Nothing may write to the chunks, their tables, or `statuses` until this returns.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_uncompressed_chunk_bytes` exceeds the codec's maximum chunk size for decompression, `statuses` does not have one slot per chunk or belongs to another CUDA context, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    unsafe fn decompression_temporary_size_sync(
        input: DeviceChunkTableView<'_>,
        max_uncompressed_chunk_bytes: usize,
        options: Self::DecompressOptions,
        max_total_uncompressed_bytes: usize,
        statuses: &mut DeviceStatuses,
        stream: &Stream,
    ) -> Result<usize>;

    /// Returns the size, in bytes, of the temporary GPU workspace that decompressing `input` into `output` needs.
    ///
    /// This calls [`decompression_temporary_size_sync`](Self::decompression_temporary_size_sync) with the largest and total capacities of `output`'s chunks.
    ///
    /// # Safety
    ///
    /// The requirements of [`decompression_temporary_size_sync`](Self::decompression_temporary_size_sync) apply.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` and `output` have different lengths, an output chunk is larger than the codec's maximum chunk size for decompression, or [`decompression_temporary_size_sync`](Self::decompression_temporary_size_sync) fails.
    unsafe fn decompression_temporary_size_for_output_sync(
        input: DeviceChunkTableView<'_>,
        output: &DeviceChunkTableMut<'_>,
        options: Self::DecompressOptions,
        statuses: &mut DeviceStatuses,
        stream: &Stream,
    ) -> Result<usize>;

    /// Enqueues batched compression of `input` into `output` on `stream`.
    ///
    /// Before launching, this checks that:
    ///
    /// - the tables, `temp`, and `statuses` share the stream's CUDA context, and the tables have the same length;
    /// - no input chunk exceeds the codec's maximum chunk size for compression;
    /// - every output chunk can hold [`max_output_chunk_size`](Self::max_output_chunk_size) bytes;
    /// - the input chunks, output chunks, and `temp` meet the alignments from [`compression_alignments`](Self::compression_alignments);
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
    ///
    /// # Errors
    ///
    /// Returns an error if a check fails, the CUDA context cannot be bound, or nvCOMP rejects the launch.
    unsafe fn compress_chunks_async<'a>(
        input: &'a DeviceChunkTable<'_>,
        output: &'a mut DeviceChunkTableMut<'_>,
        temp: &'a mut TemporaryBuffer,
        statuses: &'a mut DeviceStatuses,
        options: Self::CompressOptions,
        stream: &'a Stream,
    ) -> Result<AsyncBatchResult<'a>>;

    /// Enqueues a query for the uncompressed size of each chunk in `input`, written to `uncompressed_chunk_bytes`.
    ///
    /// The query does not decompress, and sizes computed from corrupt input are invalid.
    /// Synchronize `stream` before reading the sizes.
    ///
    /// # Safety
    ///
    /// Every chunk in `input` must be initialized and meet the input alignment from [`decompression_alignments`](Self::decompression_alignments).
    /// `input`, `uncompressed_chunk_bytes`, and the memory they reference must stay alive until the work finishes, including every replay of a captured graph.
    /// Nothing else may write to them, or read `uncompressed_chunk_bytes`, until then.
    ///
    /// # Errors
    ///
    /// Returns an error if `uncompressed_chunk_bytes` does not have one entry per chunk, the CUDA context cannot be bound, or nvCOMP rejects the launch.
    unsafe fn decompress_size_chunks_async<S>(
        input: DeviceChunkTableView<'_>,
        uncompressed_chunk_bytes: &mut S,
        stream: &Stream,
    ) -> Result<()>
    where
        S: DeviceOutput<usize> + ?Sized;

    /// Enqueues batched decompression of `input` into `output` on `stream`.
    ///
    /// Before launching, this checks that:
    ///
    /// - the tables, `temp`, and `statuses` share the stream's CUDA context, and the tables have the same length;
    /// - no output chunk exceeds the codec's maximum chunk size for decompression;
    /// - the input chunks, output chunks, and `temp` meet the alignments from [`decompression_alignments`](Self::decompression_alignments);
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
    ///
    /// # Errors
    ///
    /// Returns an error if a check fails, the CUDA context cannot be bound, or nvCOMP rejects the launch.
    unsafe fn decompress_chunks_async<'a>(
        input: DeviceChunkTableView<'a>,
        output: &'a mut DeviceChunkTableMut<'_>,
        temp: &'a mut TemporaryBuffer,
        statuses: &'a mut DeviceStatuses,
        options: Self::DecompressOptions,
        stream: &'a Stream,
    ) -> Result<AsyncBatchResult<'a>>;
}

/// Compression and decompression options for one codec, with methods that use them.
///
/// Use it when compressing repeatedly with the same settings.
/// For other options, call the [`BatchedCodec`] methods on the codec type, such as `Lz4::max_output_chunk_size`.
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

    /// Returns the minimum buffer alignments for compression.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP rejects the options.
    pub fn compression_alignments(self) -> Result<AlignmentRequirements> {
        C::compression_alignments(self.compress_options)
    }

    /// Returns the minimum buffer alignments for decompression.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP rejects the options.
    pub fn decompression_alignments(self) -> Result<AlignmentRequirements> {
        C::decompression_alignments(self.decompress_options)
    }

    /// Returns the size, in bytes, of the temporary GPU workspace that compression needs.
    ///
    /// See [`BatchedCodec::compression_temporary_size`].
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk size exceeds the codec's limit, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    pub fn compression_temporary_size(
        self,
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        stream: &Stream,
    ) -> Result<usize> {
        C::compression_temporary_size(
            num_chunks,
            max_uncompressed_chunk_bytes,
            self.compress_options,
            max_total_uncompressed_bytes,
            stream,
        )
    }

    /// Returns the size, in bytes, of the temporary GPU workspace that compressing `input` needs.
    ///
    /// See [`BatchedCodec::compression_temporary_size_sync`].
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::compression_temporary_size_sync`] apply.
    ///
    /// # Errors
    ///
    /// Returns an error if the largest chunk exceeds the codec's limit, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    pub unsafe fn compression_temporary_size_sync(
        self,
        input: &DeviceChunkTable<'_>,
        stream: &Stream,
    ) -> Result<usize> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { C::compression_temporary_size_sync(input, self.compress_options, stream) }
    }

    /// Returns the largest size, in bytes, that a chunk of at most `max_uncompressed_chunk_bytes` can compress to.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_uncompressed_chunk_bytes` exceeds the codec's limit, it does not fit nvCOMP's integer types, or nvCOMP rejects the query.
    pub fn max_output_chunk_size(self, max_uncompressed_chunk_bytes: usize) -> Result<usize> {
        C::max_output_chunk_size(max_uncompressed_chunk_bytes, self.compress_options)
    }

    /// Returns the size, in bytes, of the temporary GPU workspace that decompression needs.
    ///
    /// See [`BatchedCodec::decompression_temporary_size`].
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk size exceeds the codec's limit, a size does not fit nvCOMP's integer types, the CUDA context cannot be bound, or nvCOMP rejects the query.
    pub fn decompression_temporary_size(
        self,
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        stream: &Stream,
    ) -> Result<usize> {
        C::decompression_temporary_size(
            num_chunks,
            max_uncompressed_chunk_bytes,
            self.decompress_options,
            max_total_uncompressed_bytes,
            stream,
        )
    }

    /// Returns the size, in bytes, of the temporary GPU workspace that decompressing `input` needs.
    ///
    /// See [`BatchedCodec::decompression_temporary_size_sync`].
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::decompression_temporary_size_sync`] apply.
    ///
    /// # Errors
    ///
    /// Returns an error if [`BatchedCodec::decompression_temporary_size_sync`] fails.
    pub unsafe fn decompression_temporary_size_sync(
        self,
        input: DeviceChunkTableView<'_>,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        statuses: &mut DeviceStatuses,
        stream: &Stream,
    ) -> Result<usize> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            C::decompression_temporary_size_sync(
                input,
                max_uncompressed_chunk_bytes,
                self.decompress_options,
                max_total_uncompressed_bytes,
                statuses,
                stream,
            )
        }
    }

    /// Returns the size, in bytes, of the temporary GPU workspace that decompressing `input` into `output` needs.
    ///
    /// See [`BatchedCodec::decompression_temporary_size_for_output_sync`].
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::decompression_temporary_size_for_output_sync`] apply.
    ///
    /// # Errors
    ///
    /// Returns an error if [`BatchedCodec::decompression_temporary_size_for_output_sync`] fails.
    pub unsafe fn decompression_temporary_size_for_output_sync(
        self,
        input: DeviceChunkTableView<'_>,
        output: &DeviceChunkTableMut<'_>,
        statuses: &mut DeviceStatuses,
        stream: &Stream,
    ) -> Result<usize> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            C::decompression_temporary_size_for_output_sync(
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
    /// See [`BatchedCodec::compress_chunks_async`] for the checks it makes and how to wait for the result.
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::compress_chunks_async`] apply.
    ///
    /// # Errors
    ///
    /// Returns an error if [`BatchedCodec::compress_chunks_async`] fails.
    pub unsafe fn compress_chunks_async<'a>(
        self,
        input: &'a DeviceChunkTable<'_>,
        output: &'a mut DeviceChunkTableMut<'_>,
        temp: &'a mut TemporaryBuffer,
        statuses: &'a mut DeviceStatuses,
        stream: &'a Stream,
    ) -> Result<AsyncBatchResult<'a>> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            C::compress_chunks_async(input, output, temp, statuses, self.compress_options, stream)
        }
    }

    /// Allocates temporary and status buffers, then enqueues batched compression of `input` into `output` on `stream`.
    ///
    /// The returned handle owns those buffers.
    /// Keep it alive until you call [`OwnedAsyncBatchResult::check`].
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::compress_chunks_async`] apply to `input` and `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, an allocation fails, or [`BatchedCodec::compress_chunks_async`] fails.
    pub unsafe fn compress_chunks_alloc_async<'a>(
        self,
        input: &'a DeviceChunkTable<'_>,
        output: &'a mut DeviceChunkTableMut<'_>,
        stream: &'a Stream,
    ) -> Result<OwnedAsyncBatchResult<'a>> {
        stream.ensure_not_capturing()?;

        stream.context().bind()?;
        let temp_bytes = self.compression_temporary_size(
            input.len(),
            input.max_chunk_bytes(),
            input.total_bytes(),
            stream,
        )?;
        let temp = TemporaryBuffer::create(stream.context(), temp_bytes)?;
        let statuses = DeviceStatuses::create(stream.context(), input.len())?;

        // SAFETY: the caller upholds this function's safety contract.
        let mut submission = unsafe { OwnedAsyncBatchResult::create(statuses, temp, stream)? };
        // SAFETY: the caller upholds this function's safety contract.
        let _ = unsafe {
            C::compress_chunks_async(
                input,
                output,
                &mut submission.temp,
                &mut submission.statuses,
                self.compress_options,
                stream,
            )
        }?;

        Ok(submission)
    }

    /// Enqueues a query for the uncompressed size of each chunk in `input`, written to `uncompressed_chunk_bytes`.
    ///
    /// See [`BatchedCodec::decompress_size_chunks_async`].
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::decompress_size_chunks_async`] apply.
    ///
    /// # Errors
    ///
    /// Returns an error if [`BatchedCodec::decompress_size_chunks_async`] fails.
    pub unsafe fn decompress_size_chunks_async<S>(
        self,
        input: DeviceChunkTableView<'_>,
        uncompressed_chunk_bytes: &mut S,
        stream: &Stream,
    ) -> Result<()>
    where
        S: DeviceOutput<usize> + ?Sized,
    {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { C::decompress_size_chunks_async(input, uncompressed_chunk_bytes, stream) }
    }

    /// Enqueues batched decompression of `input` into `output` on `stream`.
    ///
    /// See [`BatchedCodec::decompress_chunks_async`] for the checks it makes and how to wait for the result.
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::decompress_chunks_async`] apply.
    ///
    /// # Errors
    ///
    /// Returns an error if [`BatchedCodec::decompress_chunks_async`] fails.
    pub unsafe fn decompress_chunks_async<'a>(
        self,
        input: DeviceChunkTableView<'a>,
        output: &'a mut DeviceChunkTableMut<'_>,
        temp: &'a mut TemporaryBuffer,
        statuses: &'a mut DeviceStatuses,
        stream: &'a Stream,
    ) -> Result<AsyncBatchResult<'a>> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            C::decompress_chunks_async(
                input,
                output,
                temp,
                statuses,
                self.decompress_options,
                stream,
            )
        }
    }

    /// Allocates temporary and status buffers, then enqueues batched decompression of `input` into `output` on `stream`.
    ///
    /// The returned handle owns those buffers.
    /// Keep it alive until you call [`OwnedAsyncBatchResult::check`].
    ///
    /// # Safety
    ///
    /// The requirements of [`BatchedCodec::decompress_chunks_async`] apply to `input` and `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, an allocation fails, or [`BatchedCodec::decompress_chunks_async`] fails.
    pub unsafe fn decompress_chunks_alloc_async<'a>(
        self,
        input: DeviceChunkTableView<'a>,
        output: &'a mut DeviceChunkTableMut<'_>,
        stream: &'a Stream,
    ) -> Result<OwnedAsyncBatchResult<'a>> {
        stream.ensure_not_capturing()?;

        let statuses = DeviceStatuses::create(stream.context(), input.len())?;
        let temp_bytes = C::decompression_temporary_size(
            input.len(),
            output.max_buffer_bytes(),
            self.decompress_options,
            output.total_buffer_bytes(),
            stream,
        )?;
        let temp = TemporaryBuffer::create(stream.context(), temp_bytes)?;

        // SAFETY: the caller upholds this function's safety contract.
        let mut submission = unsafe { OwnedAsyncBatchResult::create(statuses, temp, stream)? };
        // SAFETY: the caller upholds this function's safety contract.
        let _ = unsafe {
            C::decompress_chunks_async(
                input,
                output,
                &mut submission.temp,
                &mut submission.statuses,
                self.decompress_options,
                stream,
            )?
        };

        Ok(submission)
    }

    /// Compresses all of `input` as one chunk and waits for the result.
    ///
    /// This allocates the output and temporary buffers, enqueues the work on `stream`, waits for it, and checks the status.
    /// The decoding options from [`BatchedCodec::decoding_options`] are stored with the result for decompression.
    /// An empty `input` gives an empty chunk without calling nvCOMP.
    ///
    /// # Errors
    ///
    /// Returns an error if the input size is not a multiple of the codec's element size, stream capture is active, the buffers and `stream` belong to different CUDA contexts, an allocation fails, nvCOMP rejects a query or the launch, or a chunk reports an error status.
    pub fn compress_one<B>(self, input: &B, stream: &Stream) -> Result<CompressedChunk<C>>
    where
        B: DeviceSlice<u8> + ?Sized,
    {
        compress_one::<C, B>(
            input,
            self.compress_options,
            C::decoding_options(self.compress_options, self.decompress_options),
            stream,
        )
    }

    /// Decompresses `compressed` into `output`, waits for the result, and returns the number of bytes written.
    ///
    /// This allocates the temporary and status buffers, enqueues the work on `stream`, waits for it, and checks the status.
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
        decompress_one::<C, B>(compressed, output, stream)
    }

    /// Compresses `input` in chunks of `chunk_bytes` bytes and waits for the result.
    ///
    /// The chunks are compressed in one batch into a [`CompressedBatch`] whose slots each hold [`max_output_chunk_size`](BatchedCodec::max_output_chunk_size) bytes.
    /// The decoding options from [`BatchedCodec::decoding_options`] are stored with the result for decompression.
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
        compress_contiguous::<C, B>(
            input,
            chunk_bytes,
            self.compress_options,
            C::decoding_options(self.compress_options, self.decompress_options),
            stream,
        )
    }

    /// Decompresses every chunk of `compressed` into consecutive ranges of `output`, waits for the result, and returns the number of bytes written.
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
        decompress_contiguous::<C, B>(compressed, output, stream)
    }
}

fn compress_one<C, B>(
    input: &B,
    options: C::CompressOptions,
    decoding_options: C::DecompressOptions,
    stream: &Stream,
) -> Result<CompressedChunk<C>>
where
    C: BatchedCodec,
    B: DeviceSlice<u8> + ?Sized,
{
    stream.ensure_not_capturing()?;
    if input.context() != stream.context() {
        return Err(Error::StreamContextMismatch);
    }

    stream.context().bind()?;
    if input.is_empty() {
        // SAFETY: an empty chunk has no bytes to initialize.
        return unsafe {
            CompressedChunk::create(
                DeviceMemory::<u8>::create_uninit(stream.context(), 0)?,
                0,
                0,
                decoding_options,
            )
        };
    }

    let input_chunks = DeviceChunkTable::from_contiguous(stream.context(), input)?;
    let temp_bytes = C::compression_temporary_size(
        input_chunks.len(),
        input_chunks.max_chunk_bytes(),
        options,
        input_chunks.total_bytes(),
        stream,
    )?;
    input_chunks.require_element_size(C::compression_element_size(options))?;
    let max_compressed_bytes = C::max_output_chunk_size(input.len(), options)?;

    let mut memory = DeviceMemory::<u8>::create_uninit(stream.context(), max_compressed_bytes)?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(stream.context(), &mut memory)?;
    let mut statuses = DeviceStatuses::create(stream.context(), input_chunks.len())?;
    let mut temp = TemporaryBuffer::create(stream.context(), temp_bytes)?;

    stream.sync_scope(|_| {
        Ok((|| -> Result<()> {
            input_chunks.view().join_regions(stream)?;
            // SAFETY: the draining scope keeps every chunk buffer borrowed until the queued work completes.
            unsafe {
                C::compress_chunks_async(
                    &input_chunks,
                    &mut output_chunks,
                    &mut temp,
                    &mut statuses,
                    options,
                    stream,
                )
            }?
            .check()?;
            Ok(())
        })())
    })??;

    // SAFETY: the draining scope above completed every chunk write before this readback.
    let compressed_sizes = unsafe { output_chunks.copy_written_bytes(stream) }?;
    drop(output_chunks);

    // SAFETY: checked status and written bounds establish a completed valid prefix.
    unsafe { CompressedChunk::create(memory, compressed_sizes[0], input.len(), decoding_options) }
}

fn decompress_one<C, B>(
    compressed: &CompressedChunk<C>,
    output: &mut B,
    stream: &Stream,
) -> Result<usize>
where
    C: BatchedCodec,
    B: DeviceOutput<u8> + ?Sized,
{
    stream.ensure_not_capturing()?;
    if compressed.memory().context() != stream.context() || output.context() != stream.context() {
        return Err(Error::StreamContextMismatch);
    }

    stream.context().bind()?;
    let options = compressed.decoding_options;
    if output.len() < compressed.uncompressed_len() {
        return Err(Error::LengthMismatch {
            name: "output buffer".into(),
            expected: compressed.uncompressed_len(),
            actual: output.len(),
        });
    }

    if compressed.is_empty() {
        if compressed.uncompressed_len() == 0 {
            return Ok(0);
        }

        return Err(Error::InvalidValue {
            name: "compressed chunk bytes".into(),
        });
    }

    let input_chunks = DeviceChunkTable::from_contiguous_prefix(
        stream.context(),
        compressed.memory(),
        compressed.len(),
    )?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(stream.context(), output)?;
    let mut statuses = DeviceStatuses::create(stream.context(), input_chunks.len())?;
    let temp_bytes = C::decompression_temporary_size(
        input_chunks.len(),
        output_chunks.max_buffer_bytes(),
        options,
        output_chunks.total_buffer_bytes(),
        stream,
    )?;
    let mut temp = TemporaryBuffer::create(stream.context(), temp_bytes)?;

    stream.sync_scope(|_| {
        Ok((|| -> Result<()> {
            input_chunks.view().join_regions(stream)?;
            // SAFETY: the draining scope keeps every chunk buffer borrowed until the queued work completes.
            unsafe {
                C::decompress_chunks_async(
                    input_chunks.view(),
                    &mut output_chunks,
                    &mut temp,
                    &mut statuses,
                    options,
                    stream,
                )?
            }
            .check()?;
            Ok(())
        })())
    })??;

    // SAFETY: the draining scope above completed every chunk write before this readback.
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

fn compress_contiguous<C, B>(
    input: &B,
    chunk_bytes: usize,
    options: C::CompressOptions,
    decoding_options: C::DecompressOptions,
    stream: &Stream,
) -> Result<CompressedBatch<C>>
where
    C: BatchedCodec,
    B: DeviceSlice<u8> + ?Sized,
{
    stream.ensure_not_capturing()?;
    if input.context() != stream.context() {
        return Err(Error::StreamContextMismatch);
    }

    stream.context().bind()?;
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

    input_chunks.require_element_size(C::compression_element_size(options))?;
    let stride_bytes = C::max_output_chunk_size(input_chunks.max_chunk_bytes(), options)?;
    let allocation_bytes = stride_bytes
        .checked_mul(input_chunks.len())
        .ok_or_else(|| Error::OutOfRange {
            name: "compressed batch bytes".into(),
        })?;
    let mut memory = DeviceMemory::<u8>::create_uninit(stream.context(), allocation_bytes)?;
    let mut output_chunks =
        DeviceChunkTableMut::from_contiguous_chunks(stream.context(), &mut memory, stride_bytes)?;
    let temp_bytes = C::compression_temporary_size(
        input_chunks.len(),
        input_chunks.max_chunk_bytes(),
        options,
        input_chunks.total_bytes(),
        stream,
    )?;
    let mut temp = TemporaryBuffer::create(stream.context(), temp_bytes)?;
    let mut statuses = DeviceStatuses::create(stream.context(), input_chunks.len())?;

    stream.sync_scope(|_| {
        Ok((|| -> Result<()> {
            input_chunks.view().join_regions(stream)?;
            // SAFETY: the draining scope keeps every chunk buffer borrowed until the queued work completes.
            unsafe {
                C::compress_chunks_async(
                    &input_chunks,
                    &mut output_chunks,
                    &mut temp,
                    &mut statuses,
                    options,
                    stream,
                )
            }?
            .check()?;
            Ok(())
        })())
    })??;

    // SAFETY: the draining scope above completed every chunk write before this readback.
    let compressed_chunk_bytes = unsafe { output_chunks.copy_written_bytes(stream) }?;
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

fn decompress_contiguous<C, B>(
    compressed: &CompressedBatch<C>,
    output: &mut B,
    stream: &Stream,
) -> Result<usize>
where
    C: BatchedCodec,
    B: DeviceOutput<u8> + ?Sized,
{
    stream.ensure_not_capturing()?;
    if compressed.memory().context() != stream.context() || output.context() != stream.context() {
        return Err(Error::StreamContextMismatch);
    }

    stream.context().bind()?;
    let options = compressed.decoding_options;
    if output.len() < compressed.total_uncompressed_len() {
        return Err(Error::LengthMismatch {
            name: "output buffer".into(),
            expected: compressed.total_uncompressed_len(),
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
    let temp_bytes = C::decompression_temporary_size(
        input_chunks.len(),
        output_chunks.max_buffer_bytes(),
        options,
        output_chunks.total_buffer_bytes(),
        stream,
    )?;
    let mut temp = TemporaryBuffer::create(stream.context(), temp_bytes)?;

    stream.sync_scope(|_| {
        Ok((|| -> Result<()> {
            input_chunks.view().join_regions(stream)?;
            // SAFETY: the draining scope keeps every chunk buffer borrowed until the queued work completes.
            unsafe {
                C::decompress_chunks_async(
                    input_chunks.view(),
                    &mut output_chunks,
                    &mut temp,
                    &mut statuses,
                    options,
                    stream,
                )?
            }
            .check()?;
            Ok(())
        })())
    })??;

    // SAFETY: the draining scope above completed every chunk write before this readback.
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

fn contiguous_chunk_sizes(bytes: usize, chunk_bytes: usize) -> Result<Vec<usize>> {
    if chunk_bytes == 0 {
        return Err(Error::InvalidValue {
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

        impl private::Sealed for $marker {}

        impl BatchedCodec for $marker {
            type CompressOptions = $crate::$module::CompressOptions;
            type DecompressOptions = $crate::$module::DecompressOptions;

            const NAME: &'static str = $name;
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

            fn compression_temporary_size(
                num_chunks: usize,
                max_uncompressed_chunk_bytes: usize,
                options: Self::CompressOptions,
                max_total_uncompressed_bytes: usize,
                stream: &Stream,
            ) -> Result<usize> {
                $crate::$module::compression_temporary_size(
                    num_chunks,
                    max_uncompressed_chunk_bytes,
                    options,
                    max_total_uncompressed_bytes,
                    stream,
                )
            }

            unsafe fn compression_temporary_size_sync(
                input: &DeviceChunkTable<'_>,
                options: Self::CompressOptions,
                stream: &Stream,
            ) -> Result<usize> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $crate::$module::compression_temporary_size_sync(input, options, stream) }
            }

            fn max_output_chunk_size(
                max_uncompressed_chunk_bytes: usize,
                options: Self::CompressOptions,
            ) -> Result<usize> {
                $crate::$module::max_output_chunk_size(max_uncompressed_chunk_bytes, options)
            }

            fn decompression_temporary_size(
                num_chunks: usize,
                max_uncompressed_chunk_bytes: usize,
                options: Self::DecompressOptions,
                max_total_uncompressed_bytes: usize,
                stream: &Stream,
            ) -> Result<usize> {
                $crate::$module::decompression_temporary_size(
                    num_chunks,
                    max_uncompressed_chunk_bytes,
                    options,
                    max_total_uncompressed_bytes,
                    stream,
                )
            }

            unsafe fn decompression_temporary_size_sync(
                input: DeviceChunkTableView<'_>,
                max_uncompressed_chunk_bytes: usize,
                options: Self::DecompressOptions,
                max_total_uncompressed_bytes: usize,
                statuses: &mut DeviceStatuses,
                stream: &Stream,
            ) -> Result<usize> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $crate::$module::decompression_temporary_size_sync(
                    input,
                    max_uncompressed_chunk_bytes,
                    options,
                    max_total_uncompressed_bytes,
                    statuses,
                    stream,
                ) }
            }

            unsafe fn decompression_temporary_size_for_output_sync(
                input: DeviceChunkTableView<'_>,
                output: &DeviceChunkTableMut<'_>,
                options: Self::DecompressOptions,
                statuses: &mut DeviceStatuses,
                stream: &Stream,
            ) -> Result<usize> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $crate::$module::decompression_temporary_size_for_output_sync(
                    input, output, options, statuses, stream,
                ) }
            }

            unsafe fn compress_chunks_async<'a>(
                input: &'a DeviceChunkTable<'_>,
                output: &'a mut DeviceChunkTableMut<'_>,
                temp: &'a mut TemporaryBuffer,
                statuses: &'a mut DeviceStatuses,
                options: Self::CompressOptions,
                stream: &'a Stream,
            ) -> Result<AsyncBatchResult<'a>> {

                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $crate::$module::compress_chunks_async(
                    input, output, temp, statuses, options, stream,
                ) }
            }

            unsafe fn decompress_size_chunks_async<S>(
                input: DeviceChunkTableView<'_>,
                uncompressed_chunk_bytes: &mut S,
                stream: &Stream,
            ) -> Result<()>
            where
                S: DeviceOutput<usize> + ?Sized,
            {

                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $crate::$module::decompress_size_chunks_async(
                    input,
                    uncompressed_chunk_bytes,
                    stream,
                ) }
            }

            unsafe fn decompress_chunks_async<'a>(
                input: DeviceChunkTableView<'a>,
                output: &'a mut DeviceChunkTableMut<'_>,
                temp: &'a mut TemporaryBuffer,
                statuses: &'a mut DeviceStatuses,
                options: Self::DecompressOptions,
                stream: &'a Stream,
            ) -> Result<AsyncBatchResult<'a>> {

                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $crate::$module::decompress_chunks_async(
                    input, output, temp, statuses, options, stream,
                ) }
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
    pub trait Sealed {}
}
