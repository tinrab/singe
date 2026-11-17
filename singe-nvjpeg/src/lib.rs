//! Safe Rust wrappers for nvJPEG, which decodes and encodes JPEG images on the GPU, and for nvJPEG 2000 behind the `k2` feature.
//! The crate works on bytes in memory; read and write files with [`std::fs`].
//!
//! This crate provides:
//!
//! - **Decoding:** [`Decoder`](decode::Decoder) for one image at a time, with output formats such as interleaved or planar RGB, planar YUV, and luma only, and options for a region, scaling, and EXIF orientation.
//! - **Batched decoding:** [`BatchedDecoder`](decode::BatchedDecoder) decodes several JPEGs in one call.
//! - **Encoding:** [`Encoder`](encode::Encoder) writes device images as JPEG, with a quality setting, chroma subsampling, and interleaved or planar input.
//! - **Buffers and allocators:** reusable pinned and device buffers, and custom allocators.
//! - **JPEG 2000:** decoding and encoding through nvJPEG 2000, in the `k2` module.
//!
//! # Example
//!
//! Decode a JPEG file into interleaved RGB pixels in device memory and copy them to the host:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context as CudaContext, stream::StreamOptions};
//! use singe_nvjpeg::{
//!     context::{Context, ContextOptions},
//!     decode::{DecodeOptions, Decoder},
//!     types::OutputFormat,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda_context = CudaContext::create()?;
//!     let stream = cuda_context.create_stream(StreamOptions::default())?;
//!     let context = Context::create(&cuda_context, &ContextOptions::default())?;
//!
//!     let options = DecodeOptions::new().with_output_format(OutputFormat::RgbInterleaved);
//!     let mut decoder = Decoder::create(&context, options)?;
//!     let data = std::fs::read("input.jpg")?;
//!     let decoded = decoder.decode_to_device(&data, &stream)?;
//!
//!     let plane = decoded.single_plane()?;
//!     let size = plane.spec().size();
//!     let pixels = plane.data().copy_to_host_vec()?;
//!     println!("decoded {}x{} into {} RGB bytes", size.width(), size.height(), pixels.len());
//!     Ok(())
//! }
//! ```
//!
//! [`decode_to_device`](decode::Decoder::decode_to_device) allocates the output planes, decodes into them, and waits for the GPU before returning.
//! The `unsafe` `*_async` functions enqueue the same work without waiting, for stream capture and for callers that manage their own buffers.
//!
//! # Features
//!
//! - `k2`: adds the `k2` module for JPEG 2000, which loads nvJPEG 2000 at run time.
//!
//! # Safe calls and outputs
//!
//! The safe decode and encode calls wait for their GPU work before returning, even when they fail or unwind.
//! They reject stream capture; capturing needs the `unsafe` `*_async` functions.
//! [`EncodeInput`](encode::EncodeInput) reads initialized device images, and [`PlaneMut`](image::PlaneMut) accepts uninitialized memory to decode into.
//! Decoding writes only the image's pixels, not row padding or unused memory, and encoding to host memory returns only the encoded bytes.
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` functions check that the stream and their handles belong to the context, enqueue the work, and return without waiting.
//! Functions that take checked views or [`EncodeInput`](encode::EncodeInput) order the work after the producers of their device operands; for a native [`Image`](image::Image) description, which owns no memory, the caller orders them before the call.
//! The caller must:
//!
//! - pass valid inputs and outputs large enough for the result, and keep the outputs from being used by anything else until the work finishes;
//! - keep the input data, memory, state, settings, attached buffers, and streams alive until the work finishes, even on failure or unwinding, and through every replay of a captured graph; contexts and states do not retain the streams their work ran on.
//!
//! These calls do not mark output memory as initialized.

#[cfg(not(feature = "nvjpeg_13_2"))]
compile_error!("singe-nvjpeg requires a version feature, such as `nvjpeg_13_2`");

pub mod allocator;
pub mod buffer;
pub mod context;
pub mod decode;
pub mod decoder;
pub mod encode;
pub mod error;
pub mod image;
pub mod stream;
pub mod types;

#[cfg(feature = "k2")]
pub mod k2;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

pub use singe_core::{CudaVersion, LibraryVersion};
use singe_cuda::types::LibraryProperty;
pub use singe_nvjpeg_sys as sys;

use crate::error::{Error, Result, try_ffi};
#[cfg(feature = "k2")]
use crate::k2::error::try_ffi_k2;

/// Returns the major version, minor version, or patch level of the loaded nvJPEG library.
///
/// # Errors
///
/// Returns an error if nvJPEG fails the query.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a live local that nvJPEG writes the requested property into.
    unsafe {
        try_ffi!(sys::nvjpegGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the version of the loaded nvJPEG library.
///
/// # Errors
///
/// Returns an error if nvJPEG fails the query or reports a negative version component.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("nvjpeg version"))
}

/// Returns the major version, minor version, or patch level of the CUDA toolkit that nvJPEG was built with.
///
/// For nvJPEG's own version, use [`library_property`].
///
/// # Errors
///
/// Returns an error if nvJPEG fails the query.
pub fn cudart_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a live local that nvJPEG writes the requested property into.
    unsafe {
        try_ffi!(sys::nvjpegGetCudartProperty(
            property.into(),
            &raw mut value
        ))?;
    }
    Ok(value)
}

/// Returns the CUDA runtime version that nvJPEG was built with.
///
/// # Errors
///
/// Returns an error if nvJPEG fails the query or reports a version that does not fit the encoding.
pub fn cudart_version() -> Result<CudaVersion> {
    cuda_version(
        cudart_property(LibraryProperty::Major)?,
        cudart_property(LibraryProperty::Minor)?,
        "nvjpeg cudart version",
    )
}

// CUDA encodes a version as `1000 * major + 10 * minor`, without the patch level.
pub(crate) fn cuda_version(major: i32, minor: i32, name: &'static str) -> Result<CudaVersion> {
    let major = u32::try_from(major).ok();
    let minor = u32::try_from(minor).ok().filter(|&minor| minor < 100);
    major
        .zip(minor)
        .and_then(|(major, minor)| major.checked_mul(1000)?.checked_add(minor * 10))
        .and_then(CudaVersion::from_raw)
        .ok_or_else(|| Error::out_of_range(name))
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
