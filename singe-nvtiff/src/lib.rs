//! Safe Rust wrappers for the NVIDIA nvTIFF library.
//!
//! [`TiffStream`](stream::TiffStream) parses a TIFF file and reads its images, tags, and `GeoTIFF` keys.
//! [`Decoder`](decode::Decoder) decodes images or regions of them into device memory, and [`Encoder`](encode::Encoder) encodes device images into a TIFF file in host memory.
//!
//! # Safe calls and outputs
//!
//! The safe decode and encode calls wait for their GPU work before returning, even when they fail or unwind.
//! They reject stream capture; capturing needs the `unsafe` `*_async` functions.
//! [`ImagePlane`](image::ImagePlane) accepts uninitialized memory to decode into, and decoding writes only the image rows, not the padding between them.
//!
//! # Raw execution
//!
//! A native image description holds pointers and pitches but owns no memory.
//! Callers of the `unsafe` `*_async` functions must pass valid inputs and large enough outputs, order the work that writes the inputs before the call, and keep outputs from being used by anything else.
//! The TIFF data, memory, settings, and decoder or encoder must stay alive until the work finishes, even on failure or unwinding, and for every replay of a captured graph.
//! These calls do not mark output memory as initialized.

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

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_nvtiff_sys as sys;

use crate::error::Result;

/// Returns a version number of the loaded nvTIFF library.
///
/// # Errors
///
/// Returns an error if nvTIFF rejects the property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `nvtiffGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::nvtiffGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the version of the loaded nvTIFF library.
///
/// # Errors
///
/// Returns an error if nvTIFF cannot report a version number.
pub fn version() -> Result<LibraryVersion> {
    Ok(LibraryVersion::new(
        library_property(LibraryProperty::Major)? as _,
        library_property(LibraryProperty::Minor)? as _,
        library_property(LibraryProperty::Patch)? as _,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "testing")]
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        assert_ne!(
            (version.major(), version.minor(), version.patch()),
            (0, 0, 0),
        );
        Ok(())
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_setup_context_creates_decoder() -> Result<()> {
        let _context = setup_context()?;
        Ok(())
    }
}
