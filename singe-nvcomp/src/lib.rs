//! Safe Rust wrappers for the NVIDIA nvCOMP compression library.
//!
//! [`format::Codec`] is the main entry point.
//! It holds one codec's options and compresses or decompresses device buffers, waiting for the result.
//! [`Codec::compress_contiguous`](format::Codec::compress_contiguous) splits a large buffer into chunks and compresses them in one batch.
//!
//! Each codec module, such as [`lz4`] and [`zstd`], offers the same operations as free functions, plus lower-level ones:
//!
//! - `*_chunks_async` functions check their arguments, enqueue batched work over the chunk tables in [`batch`], and return a handle to wait on.
//! - `*_raw_async` functions pass device pointer arrays straight to nvCOMP.
//!
//! Both are `unsafe`, because every buffer must stay valid until the GPU work finishes.
//!
//! [`manager`] wraps nvCOMP's managers, which compress whole buffers in nvCOMP's own format.
//! [`streaming`] compresses gzip data between Rust readers and writers, and [`crc32`] computes checksums on the GPU.
//!
//! # Example
//!
//! ```no_run
//! use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
//! use singe_nvcomp::{error::Result, format::{Codec, Lz4}};
//!
//! fn main() -> Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let stream = cuda.create_stream()?;
//!
//!     let input = b"Hello, nvCOMP!".repeat(1024);
//!     let device_input = DeviceMemory::from_slice(&cuda, &input)?;
//!     let codec = Codec::<Lz4>::default();
//!     let compressed = codec.compress_one(&device_input, &stream)?;
//!
//!     let mut output = DeviceMemory::<u8>::create_uninit(&cuda, input.len())?;
//!     let output_bytes = codec.decompress_one(&compressed, &mut output, &stream)?;
//!     assert_eq!(output_bytes, input.len());
//!     // SAFETY: `decompress_one` returned after writing the first `output_bytes` bytes.
//!     let output = unsafe { output.assume_init_prefix(output_bytes)? };
//!     assert_eq!(output.copy_to_host_vec()?, input);
//!
//!     Ok(())
//! }
//! ```

/// Batched ANS compression and decompression.
pub mod ans;

pub mod batch;
pub mod bitcomp;

/// Batched Cascaded compression and decompression.
pub mod cascaded;

/// Compressed chunks and batches in device memory.
pub mod compressed;

/// CRC32 checksums computed on the GPU.
pub mod crc32;

/// Batched Deflate compression and decompression.
pub mod deflate;

/// Error and status types.
pub mod error;

pub mod format;

/// Batched `GDeflate` compression and decompression.
pub mod gdeflate;

/// Batched Gzip compression and decompression.
pub mod gzip;

/// Batched LZ4 compression and decompression.
pub mod lz4;

pub mod manager;

/// Batched Snappy compression and decompression.
pub mod snappy;

pub mod streaming;

/// Data types, alignments, library properties, and option enums shared by the codecs.
pub mod types;

pub mod workspace;

/// Batched Zstd compression and decompression.
pub mod zstd;

#[cfg(feature = "testing")]
/// CPU reference implementations for checking GPU results.
pub mod testing;

pub(crate) mod utility;

mod codec;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_nvcomp_sys as sys;

use crate::{error::Result, types::Properties};

/// Returns the properties of the loaded nvCOMP library.
///
/// # Errors
///
/// Returns an error if nvCOMP cannot report its properties.
pub fn properties() -> Result<Properties> {
    let mut properties = sys::nvcompProperties_t::default();
    // SAFETY: `nvcompGetProperties` receives valid output locations.
    unsafe {
        try_ffi!(sys::nvcompGetProperties(&raw mut properties))?;
    }
    Ok(properties.into())
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
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    Ok(properties()?.cudart_version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Status;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?, 0);
        assert_ne!(cudart_version()?, 0);
        assert!(!Status::Success.description().is_empty());
        Ok(())
    }
}
