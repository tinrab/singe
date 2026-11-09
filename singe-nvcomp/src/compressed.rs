use std::mem::MaybeUninit;

use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceRegionProvider, DeviceView},
};

use crate::{
    batch::DeviceChunkTable,
    error::{Error, Result},
    format::BatchedCodec,
};

/// One compressed chunk in device memory, with the information needed to decompress it.
///
/// The allocation may be larger than the compressed data; only the compressed bytes are exposed.
#[derive(Debug)]
pub struct CompressedChunk<C: BatchedCodec> {
    memory: DeviceMemory<u8>,
    uncompressed_bytes: usize,
    pub(crate) decoding_options: C::DecompressOptions,
}

impl<C: BatchedCodec> CompressedChunk<C> {
    /// Creates a chunk from compressed data in the first `bytes` bytes of `memory`.
    ///
    /// # Safety
    ///
    /// The first `bytes` bytes of `memory` must be valid compressed data for codec `C` and `decoding_options` that decompresses to `uncompressed_bytes` bytes.
    /// The work that wrote them must have finished, and nothing else may write to the allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `bytes` exceeds the size of `memory`.
    pub unsafe fn create(
        memory: DeviceMemory<MaybeUninit<u8>>,
        bytes: usize,
        uncompressed_bytes: usize,
        decoding_options: C::DecompressOptions,
    ) -> Result<Self> {
        if bytes > memory.len() {
            return Err(Error::LengthMismatch {
                name: "compressed chunk bytes".into(),
                expected: bytes,
                actual: memory.len(),
            });
        }

        // SAFETY: the caller establishes completed prefix validity.
        let memory = unsafe { memory.assume_init_prefix(bytes)? };
        Ok(Self {
            memory,
            uncompressed_bytes,
            decoding_options,
        })
    }

    /// Returns the compressed bytes.
    pub const fn memory(&self) -> &DeviceMemory<u8> {
        &self.memory
    }

    /// Returns the size of the whole allocation in bytes, including unused space after the compressed bytes.
    pub fn capacity(&self) -> usize {
        self.memory.allocation().byte_len()
    }

    /// Returns the size of the compressed data in bytes.
    pub const fn len(&self) -> usize {
        self.memory.len()
    }

    /// Returns whether the compressed data is empty.
    pub const fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Returns the size of the uncompressed data in bytes.
    pub const fn uncompressed_len(&self) -> usize {
        self.uncompressed_bytes
    }

    /// Consumes the chunk and returns its compressed bytes, uncompressed size, and decompression options.
    pub fn into_parts(self) -> CompressedChunkParts<C::DecompressOptions> {
        CompressedChunkParts {
            memory: self.memory,
            uncompressed_bytes: self.uncompressed_bytes,
            decoding_options: self.decoding_options,
        }
    }
}

/// Compressed chunks of one input, stored in equal-size slots of one device allocation.
///
/// Each slot holds one compressed chunk at its start; the rest of the slot is uninitialized.
#[derive(Debug)]
pub struct CompressedBatch<C: BatchedCodec> {
    memory: DeviceMemory<MaybeUninit<u8>>,
    stride_bytes: usize,
    chunk_bytes: Vec<usize>,
    uncompressed_chunk_bytes: Vec<usize>,
    total_compressed_bytes: usize,
    total_uncompressed_bytes: usize,
    pub(crate) decoding_options: C::DecompressOptions,
}

impl<C: BatchedCodec> CompressedBatch<C> {
    /// Creates a batch from an allocation of equal-size slots and the size of each chunk.
    ///
    /// # Safety
    ///
    /// Each slot must start with valid compressed data for codec `C` and `parts.decoding_options`, of the size in `parts.chunk_bytes`, that decompresses to the size in `parts.uncompressed_chunk_bytes`.
    /// The work that wrote it must have finished, and nothing else may write to the allocation.
    /// The rest of each slot may be uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the two size lists have different lengths, a compressed chunk is larger than its slot, or the allocation cannot hold every slot.
    pub unsafe fn create(parts: CompressedBatchParts<C::DecompressOptions>) -> Result<Self> {
        Self::create_trusted(parts)
    }

    pub(crate) fn create_trusted(
        parts: CompressedBatchParts<C::DecompressOptions>,
    ) -> Result<Self> {
        let CompressedBatchParts {
            memory,
            stride_bytes,
            chunk_bytes,
            uncompressed_chunk_bytes,
            decoding_options,
        } = parts;
        if chunk_bytes.len() != uncompressed_chunk_bytes.len() {
            return Err(Error::LengthMismatch {
                name: "uncompressed chunk bytes".into(),
                expected: chunk_bytes.len(),
                actual: uncompressed_chunk_bytes.len(),
            });
        }

        if stride_bytes == 0 && !chunk_bytes.is_empty() {
            return Err(Error::InvalidValue {
                name: "stride bytes".into(),
            });
        }

        let required_bytes =
            stride_bytes
                .checked_mul(chunk_bytes.len())
                .ok_or_else(|| Error::OutOfRange {
                    name: "compressed batch bytes".into(),
                })?;
        if required_bytes > memory.len() {
            return Err(Error::LengthMismatch {
                name: "compressed batch bytes".into(),
                expected: required_bytes,
                actual: memory.len(),
            });
        }

        for &bytes in &chunk_bytes {
            if bytes > stride_bytes {
                return Err(Error::LengthMismatch {
                    name: "compressed chunk bytes".into(),
                    expected: stride_bytes,
                    actual: bytes,
                });
            }
        }

        let total_compressed_bytes = chunk_bytes.iter().try_fold(0usize, |total, bytes| {
            total.checked_add(*bytes).ok_or_else(|| Error::OutOfRange {
                name: "total compressed bytes".into(),
            })
        })?;
        let total_uncompressed_bytes =
            uncompressed_chunk_bytes
                .iter()
                .try_fold(0usize, |total, bytes| {
                    total.checked_add(*bytes).ok_or_else(|| Error::OutOfRange {
                        name: "total uncompressed bytes".into(),
                    })
                })?;

        Ok(Self {
            memory,
            stride_bytes,
            chunk_bytes,
            uncompressed_chunk_bytes,
            total_compressed_bytes,
            total_uncompressed_bytes,
            decoding_options,
        })
    }

    /// Returns the whole allocation.
    ///
    /// The bytes after each chunk's compressed data are uninitialized.
    pub const fn memory(&self) -> &DeviceMemory<MaybeUninit<u8>> {
        &self.memory
    }

    /// Returns the size of each slot in bytes.
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Returns the compressed size of each chunk in bytes.
    pub fn chunk_bytes(&self) -> &[usize] {
        &self.chunk_bytes
    }

    /// Returns the uncompressed size of each chunk in bytes.
    pub fn uncompressed_chunk_bytes(&self) -> &[usize] {
        &self.uncompressed_chunk_bytes
    }

    /// Returns the number of chunks.
    pub fn len(&self) -> usize {
        self.chunk_bytes.len()
    }

    /// Returns whether the batch has no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunk_bytes.is_empty()
    }

    /// Returns the sum of the compressed chunk sizes in bytes.
    pub const fn total_compressed_len(&self) -> usize {
        self.total_compressed_bytes
    }

    /// Returns the sum of the uncompressed chunk sizes in bytes.
    pub const fn total_uncompressed_len(&self) -> usize {
        self.total_uncompressed_bytes
    }

    /// Returns the compressed bytes of chunk `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds or the chunk does not fit the allocation.
    pub fn chunk(&self, index: usize) -> Result<DeviceView<'_, u8>> {
        let size = *self
            .chunk_bytes
            .get(index)
            .ok_or_else(|| Error::OutOfRange {
                name: "compressed chunk index".into(),
            })?;
        let start = index
            .checked_mul(self.stride_bytes)
            .ok_or_else(|| Error::OutOfRange {
                name: "compressed chunk offset".into(),
            })?;
        let end = start.checked_add(size).ok_or_else(|| Error::OutOfRange {
            name: "compressed chunk end".into(),
        })?;
        let region = self.memory.region().slice(start..end)?;
        // SAFETY: construction establishes that each completed compressed prefix
        // is initialized; the raw tail is never exposed through this view.
        Ok(unsafe { DeviceView::from_raw_parts(region, size)? })
    }

    /// Returns a chunk table over the compressed chunks, for batched decompression.
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk does not fit the allocation, which can happen only if the batch was created from inconsistent parts, or the table cannot be allocated.
    pub fn chunks(&self) -> Result<DeviceChunkTable<'_>> {
        let chunks = (0..self.len())
            .map(|index| self.chunk(index))
            .collect::<Result<Vec<_>>>()?;
        DeviceChunkTable::from_views(self.memory.context(), chunks)
    }

    /// Returns the allocation, slot size, compressed and uncompressed chunk sizes, and decompression options.
    pub fn into_parts(self) -> CompressedBatchParts<C::DecompressOptions> {
        CompressedBatchParts {
            memory: self.memory,
            stride_bytes: self.stride_bytes,
            chunk_bytes: self.chunk_bytes,
            uncompressed_chunk_bytes: self.uncompressed_chunk_bytes,
            decoding_options: self.decoding_options,
        }
    }
}

/// Parts of a [`CompressedChunk`], from [`CompressedChunk::into_parts`].
#[derive(Debug)]
pub struct CompressedChunkParts<O> {
    /// Compressed bytes.
    pub memory: DeviceMemory<u8>,
    /// Size of the uncompressed data in bytes.
    pub uncompressed_bytes: usize,
    /// Options for decompressing the chunk.
    pub decoding_options: O,
}

/// Parts of a [`CompressedBatch`], from [`CompressedBatch::into_parts`] or for [`CompressedBatch::create`].
#[derive(Debug)]
pub struct CompressedBatchParts<O> {
    /// Allocation that holds one slot per chunk.
    pub memory: DeviceMemory<MaybeUninit<u8>>,
    /// Size of each slot in bytes.
    pub stride_bytes: usize,
    /// Compressed size of each chunk in bytes.
    pub chunk_bytes: Vec<usize>,
    /// Uncompressed size of each chunk in bytes.
    pub uncompressed_chunk_bytes: Vec<usize>,
    /// Options for decompressing the chunks.
    pub decoding_options: O,
}
