//! Safe Rust wrappers for nvTIFF, which decodes and encodes TIFF images on the GPU.
//!
//! This crate provides:
//!
//! - **Parsing:** [`TiffStream`](stream::TiffStream) opens a TIFF file from a path or from bytes and reads image information, strip and tile geometry, tag values, and `GeoTIFF` keys.
//! - **Decoding:** [`Decoder`](decode::Decoder) decodes selected images, or regions of them, into device memory.
//! - **Encoding:** [`Encoder`](encode::Encoder) encodes device images into an [`EncodedTiff`](encode::EncodedTiff) in host memory, with image geometry and `GeoTIFF` keys.
//! - **Allocators:** custom device and pinned host allocators for the decoder and encoder.
//!
//! # Example
//!
//! Decode the first image of a TIFF file and encode it into a new file:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context, stream::StreamOptions};
//! use singe_nvtiff::{
//!     allocator::AllocatorSet,
//!     decode::Decoder,
//!     encode::{EncodeParams, Encoder},
//!     image::IfdOffset,
//!     stream::TiffStream,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let context = Context::create()?;
//!     let stream = context.create_stream(StreamOptions::default())?;
//!
//!     let source = TiffStream::open_file("input.tif")?;
//!     let info = source.image_info(IfdOffset::new(0))?;
//!     let mut decoder = Decoder::create(AllocatorSet::new(), &stream)?;
//!     let pixels = decoder.decode_to_device(&source)?;
//!
//!     let mut encoder = Encoder::create(AllocatorSet::new(), &stream)?;
//!     let mut params = EncodeParams::create(&stream)?;
//!     params.set_image_info(&info.to_encode_info()?)?;
//!     encoder.encode(&mut params, [&pixels])?.write_file("output.tif")?;
//!     Ok(())
//! }
//! ```
//!
//! A decoder or encoder runs its work on the stream it was created with.
//! [`decode_to_device`](decode::Decoder::decode_to_device) allocates tightly packed device memory, decodes into it, and waits for the GPU before returning.
//! [`encode`](encode::Encoder::encode) also waits, and the [`EncodedTiff`](encode::EncodedTiff) it returns holds the file in host memory.
//! The `unsafe` [`decode_async`](decode::Decoder::decode_async), [`encode_async`](encode::Encoder::encode_async), and [`finalize_async`](encode::Encoder::finalize_async) functions enqueue the work without waiting.
//!
//! # Safe calls and outputs
//!
//! The safe decode and encode calls wait for their GPU work before returning, even when they fail or unwind.
//! They reject stream capture; capturing needs the `unsafe` `*_async` functions.
//! [`ImagePlane`](image::ImagePlane) accepts uninitialized memory to decode into, and decoding writes only the image rows, not the padding between them.
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` functions check contexts and sizes, order the work on the decoder's or encoder's stream after the queued writers of their device operands, enqueue it, and return without waiting.
//! The caller must:
//!
//! - pass initialized inputs, and keep the outputs from being used by anything else until the work finishes;
//! - keep the TIFF data, memory, settings, and decoder or encoder alive until the work finishes, even on failure or unwinding, and through every replay of a captured graph;
//! - let each request finish before reusing its decoder or encoder.
//!
//! These calls do not mark output memory as initialized.

#[cfg(not(feature = "nvtiff_0_8"))]
compile_error!("singe-nvtiff requires a version feature, such as `nvtiff_0_8`");

pub mod allocator;
pub mod decode;
pub mod encode;
pub mod error;
pub mod image;
pub mod stream;
pub mod types;
pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

pub use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
pub use singe_nvtiff_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns a version number of the loaded nvTIFF library.
///
/// # Errors
///
/// Returns an error if nvTIFF rejects the property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a writable `i32`, and `property` converts to a valid native property.
    unsafe {
        try_ffi!(sys::nvtiffGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the version of the loaded nvTIFF library.
///
/// # Errors
///
/// Returns an error if nvTIFF cannot report a version number or reports a negative version component.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("nvtiff version"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        assert_ne!(
            (version.major(), version.minor(), version.patch()),
            (0, 0, 0),
        );
        Ok(())
    }
}
