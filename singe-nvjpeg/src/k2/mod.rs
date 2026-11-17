//! JPEG 2000 decoding and encoding with nvJPEG 2000, enabled by the `k2` feature.
//!
//! Handles and the resources created from them keep their CUDA context alive.
//! A parsed [`CodeStream`] keeps the data it parsed, and a failed parse marks it as unparsed.
//! Safe encoding reads an initialized [`EncodeInput`](image::EncodeInput), and safe decoding writes to checked output planes and waits for the GPU work even if it fails.
//! Native [`Image`] descriptions and the `unsafe` `*_async` functions follow the [raw execution rules](crate#raw-execution).

pub mod context;
pub mod decode;
pub mod encode;
pub mod error;
pub mod image;
pub mod types;

use context::Context;
use decode::{CodeStream, DecodeParams, DecodeState};
use image::{Image, validate_image_component_count};
use singe_core::{CudaVersion, LibraryVersion};
use singe_cuda::types::LibraryProperty;
use singe_nvjpeg_sys::k2 as sys;
use types::{
    Backend, BitstreamType, ColorSpace, ImageComponentInfo, ImageFormat, ImageInfo, ImagePixel,
    ImageType, ProgressionOrder, QualityType, TileDecode, decode_pixel_type,
    validate_code_block_size, validate_component_precision, validate_dimensions,
    validate_resolutions,
};

use crate::{
    cuda_version,
    error::{Error, Result},
    try_ffi_k2,
};

pub(crate) const MAX_IMAGE_COMPONENTS: usize = 4;

/// Returns the major version, minor version, or patch level of the loaded nvJPEG 2000 library.
///
/// # Errors
///
/// Returns an error if nvJPEG 2000 fails the query.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a live local that nvJPEG 2000 writes the requested property into.
    unsafe {
        try_ffi_k2!(sys::nvjpeg2kGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the major version, minor version, or patch level of the CUDA toolkit that nvJPEG 2000 was built with.
///
/// # Errors
///
/// Returns an error if nvJPEG 2000 fails the query.
pub fn cudart_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a live local that nvJPEG 2000 writes the requested property into.
    unsafe {
        try_ffi_k2!(sys::nvjpeg2kGetCudartProperty(
            property.into(),
            &raw mut value
        ))?;
    }
    Ok(value)
}

/// Returns the version of the loaded nvJPEG 2000 library.
///
/// # Errors
///
/// Returns an error if nvJPEG 2000 fails the query or reports a negative version component.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("nvjpeg2k version"))
}

/// Returns the CUDA runtime version that nvJPEG 2000 was built with.
///
/// # Errors
///
/// Returns an error if nvJPEG 2000 fails the query or reports a version that does not fit the encoding.
pub fn cudart_version() -> Result<CudaVersion> {
    cuda_version(
        cudart_property(LibraryProperty::Major)?,
        cudart_property(LibraryProperty::Minor)?,
        "nvjpeg2k cudart version",
    )
}
