//! Safe Rust wrappers for the NVIDIA nvJPEG and nvJPEG 2000 libraries.
//!
//! [`Decoder`](decode::Decoder) and [`Encoder`](encode::Encoder) handle one image at a time, and [`BatchedDecoder`](decode::BatchedDecoder) decodes batches.
//! The `k2` module, enabled by the `k2` feature, covers JPEG 2000. For finer control, [`Context`](context::Context) exposes nvJPEG's staged and batched functions directly.
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
//! A native [`Image`](image::Image) description holds pointers and pitches but owns no memory.
//! Callers of the `unsafe` `*_async` functions must pass valid inputs and large enough outputs, order the work that writes the inputs before the call, and keep outputs from being used by anything else.
//! The input data, memory, state, settings, attached buffers, and streams must stay alive until the work finishes, even on failure or unwinding, and for every replay of a captured graph.
//! These calls do not mark output memory as initialized.

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

/// Common imports for decoding and encoding.
pub mod prelude {
    pub use crate::{
        allocator::{AllocatorSet, DeviceAllocator, PinnedAllocator},
        decode::{
            BatchConfig, BatchPrealloc, BatchedDecoder, BatchedDecoderBuilder, DecodeOptions,
            DecodeParams, DecodedImage, DecodedOutput, DecodedPlane, Decoder, DecoderBuilder,
            Rectangle,
        },
        encode::{
            EncodeInput, EncodeOptions, EncodedDeviceBitstream, Encoder, EncoderBuilder, Quality,
        },
        image::{ImageView, ImageViewMut, OutputPlaneSpec, OutputSpec, Plane, PlaneMut, Size},
        stream::{JpegComponent, JpegStream, JpegStreamInfo},
        types::{
            Backend, ChromaSubsampling, CreateFlags, EncoderBackend, ExifOrientation,
            HardwareDecoderInfo, HardwareEncoderInfo, InputFormat, JpegComponentInfo, JpegEncoding,
            JpegInfo, OutputFormat, ScaleFactor,
        },
    };
}

#[cfg(feature = "k2")]
pub mod k2;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cuda::types::LibraryProperty;
use singe_nvjpeg_sys as sys;

use crate::error::Result;

/// Returns the major version, minor version, or patch level of the loaded nvJPEG library.
///
/// # Errors
///
/// Returns an error if nvJPEG fails the query.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `nvjpegGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::nvjpegGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the version of the loaded nvJPEG library.
///
/// # Errors
///
/// Returns an error if nvJPEG fails the query.
pub fn version() -> Result<LibraryVersion> {
    Ok(LibraryVersion::new(
        library_property(LibraryProperty::Major)? as _,
        library_property(LibraryProperty::Minor)? as _,
        library_property(LibraryProperty::Patch)? as _,
    ))
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
    // SAFETY: `nvjpegGetCudartProperty` receives valid output locations.
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
/// Returns an error if nvJPEG fails the query.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    let raw = cudart_property(LibraryProperty::Major)? as u64 * 1000
        + cudart_property(LibraryProperty::Minor)? as u64 * 10
        + cudart_property(LibraryProperty::Patch)? as u64;
    Ok(CudaRuntimeVersion::from_raw(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "testing")]
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?, 0);
        assert_ne!(cudart_version()?.raw(), 0);
        Ok(())
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_setup_context_creates_context() -> Result<()> {
        let _context = setup_context()?;
        Ok(())
    }
}
