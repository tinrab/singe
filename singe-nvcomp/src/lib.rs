//! Safe Rust wrappers for nvCOMP, which compresses and decompresses data in GPU memory with the LZ4, Snappy, Zstd, Deflate, Gzip, `GDeflate`, ANS, Cascaded, and Bitcomp codecs.
//!
//! This crate provides:
//!
//! - **One codec type:** [`Codec<C>`](format::Codec) holds one codec's options, selected by a marker type such as [`Lz4`](format::Lz4) or [`Zstd`](format::Zstd), and compresses and decompresses device buffers.
//! - **Whole buffers:** [`compress_one`](format::Codec::compress_one) and [`decompress_one`](format::Codec::decompress_one) treat a buffer as one chunk, and [`compress_contiguous`](format::Codec::compress_contiguous) and [`decompress_contiguous`](format::Codec::decompress_contiguous) split a large buffer into chunks and process them in one batch.
//! - **Chunk tables:** [`compress_chunks`](format::Codec::compress_chunks) and [`compress_chunks_scoped`](format::Codec::compress_chunks_scoped) compress caller-provided chunks, and the `unsafe` [`compress_chunks_async`](format::Codec::compress_chunks_async) and [`decompress_chunks_async`](format::Codec::decompress_chunks_async) enqueue batched work without waiting.
//! - **Compressed data:** [`CompressedChunk`](compressed::CompressedChunk) and [`CompressedBatch`](compressed::CompressedBatch) keep compressed device bytes together with the options needed to decompress them.
//! - **Checksums:** CRC32 computed on the GPU.
//!
//! # Example
//!
//! Compress device bytes with LZ4 and check that decompressing them gives the input back:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions};
//! use singe_nvcomp::format::{Codec, Lz4};
//!
//! fn main() -> anyhow::Result<()> {
//!     let context = Context::create()?;
//!     let stream = context.create_stream(StreamOptions::default())?;
//!
//!     let input = b"Hello, nvCOMP!".repeat(1024);
//!     let device_input = DeviceMemory::from_slice(&context, &input)?;
//!     let codec = Codec::<Lz4>::default();
//!     let compressed = codec.compress_one(&device_input, &stream)?;
//!
//!     let mut output = DeviceMemory::<u8>::create_uninit(&context, input.len())?;
//!     let output_bytes = codec.decompress_one(&compressed, &mut output, &stream)?;
//!     // SAFETY: `decompress_one` returned after writing the first `output_bytes` bytes.
//!     let output = unsafe { output.assume_init_prefix(output_bytes)? };
//!     assert_eq!(output.copy_to_host_vec()?, input);
//!
//!     println!("{} bytes compressed to {}", input.len(), compressed.len());
//!     Ok(())
//! }
//! ```
//!
//! [`compress_one`](format::Codec::compress_one) and [`decompress_one`](format::Codec::decompress_one) wait for the GPU before returning, and [`decompress_one`](format::Codec::decompress_one) returns the number of bytes it wrote.
//! Decompressing a raw chunk table stays `unsafe`, because nvCOMP documents corrupt compressed input as undefined behavior.
//!
//! # Features
//!
//! - `manager`: adds the `manager` module, for nvCOMP's managers, whose output starts with a header that records the codec and sizes, and the `streaming` module, for gzip between Rust readers and writers.
//!   It builds a C++ bridge that links nvCOMP at build time, found through `NVCOMP_INCLUDE_PATH` and `NVCOMP_PATH`.
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` functions check contexts, chunk counts, capacities, alignments, and workspace sizes, order the work on the given stream after the producers of every operand, enqueue it, and return without waiting.
//! The caller must:
//!
//! - keep every chunk, chunk table, workspace, status buffer, and output alive, and let nothing else write them or read the outputs, until the work finishes, including through every replay of a captured graph;
//! - initialize the input chunks, which a [`DeviceChunkTable`](batch::DeviceChunkTable) guarantees and a [`DeviceChunkTableView`](batch::DeviceChunkTableView) does not;
//! - pass decompression input that is valid compressed data for the codec and options, since nvCOMP documents corrupt, truncated, or mismatched input as undefined behavior that can write past the output capacities.
//!
//! Outputs, sizes, and statuses hold results only after the work finishes; [`AsyncBatchResult::check`](batch::AsyncBatchResult::check) waits and checks the statuses.
//! The safe [`decompress_one`](format::Codec::decompress_one) and [`decompress_contiguous`](format::Codec::decompress_contiguous) take a [`CompressedChunk`](compressed::CompressedChunk) or [`CompressedBatch`](compressed::CompressedBatch), which holds data from a completed compression.

#[cfg(not(feature = "nvcomp_6_0"))]
compile_error!("singe-nvcomp requires a version feature, such as `nvcomp_6_0`");

/// ANS options and chunk limits for [`format::Codec`].
pub mod ans;

pub mod batch;
pub mod bitcomp;

/// Cascaded options and chunk limits for [`format::Codec`].
pub mod cascaded;

/// Compressed chunks and batches in device memory.
pub mod compressed;

/// CRC32 checksums computed on the GPU.
pub mod crc32;

/// Deflate options and chunk limits for [`format::Codec`].
pub mod deflate;

/// Error and status types.
pub mod error;

pub mod format;

/// `GDeflate` options and chunk limits for [`format::Codec`].
pub mod gdeflate;

/// Gzip options and chunk limits for [`format::Codec`].
pub mod gzip;

/// LZ4 options and chunk limits for [`format::Codec`].
pub mod lz4;

#[cfg(feature = "manager")]
pub mod manager;

/// Snappy options and chunk limits for [`format::Codec`].
pub mod snappy;

#[cfg(feature = "manager")]
pub mod streaming;

#[cfg(feature = "testing")]
pub mod testing;

/// Data types, alignments, library properties, and option enums shared by the codecs.
pub mod types;

/// Zstd options and chunk limits for [`format::Codec`].
pub mod zstd;

pub(crate) mod utility;

mod codec;

pub use singe_core::{CudaVersion, LibraryVersion};
pub use singe_nvcomp_sys as sys;

use crate::{
    error::{Result, try_ffi},
    types::Properties,
};

/// Returns the properties of the loaded nvCOMP library.
///
/// # Errors
///
/// Returns an error if nvCOMP cannot report its properties or reports a version that does not fit its encoding.
pub fn properties() -> Result<Properties> {
    let mut properties = sys::nvcompProperties_t::default();
    // SAFETY: `properties` is a live local of the native properties type, which nvCOMP fills.
    unsafe {
        try_ffi!(sys::nvcompGetProperties(&raw mut properties))?;
    }
    Properties::try_from(properties)
}

/// Returns the version of the loaded nvCOMP library.
///
/// # Errors
///
/// Returns an error if nvCOMP cannot report its properties.
pub fn version() -> Result<LibraryVersion> {
    Ok(properties()?.version)
}

/// Returns the CUDA runtime version that nvCOMP was built against.
///
/// # Errors
///
/// Returns an error if nvCOMP cannot report its properties.
pub fn cudart_version() -> Result<CudaVersion> {
    Ok(properties()?.cudart_version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?, LibraryVersion::new(0, 0, 0));
        assert_ne!(cudart_version()?.major(), 0);
        Ok(())
    }
}
