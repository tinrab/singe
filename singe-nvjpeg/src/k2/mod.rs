//! JPEG 2000 decoding and encoding with nvJPEG 2000, enabled by the `k2` feature.
//!
//! Handles and the resources created from them keep their CUDA context and streams alive.
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
use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cuda::types::LibraryProperty;
use singe_nvjpeg_sys::k2 as sys;
use types::{
    Backend, BitstreamType, ColorSpace, ImageComponentInfo, ImageFormat, ImageInfo, ImagePixel,
    ImageType, ProgressionOrder, QualityType, TileDecode, decode_pixel_type,
    validate_code_block_size, validate_component_precision, validate_dimensions,
    validate_resolutions,
};

use crate::{error::Result, try_ffi_k2};

pub(crate) const MAX_IMAGE_COMPONENTS: usize = 4;

/// Common imports for JPEG 2000 decoding and encoding.
pub mod prelude {
    pub use super::{
        decode::{
            CodeStream, DecodeArea, DecodeOptions, DecodeParams, DecodedComponent,
            DecodedComponentSpec, DecodedImage, Decoder, DecoderBuilder,
        },
        encode::{
            EncodeConfig, EncodeConfigBuilder, EncodeOptions, EncodeParams, EncodeQuality, Encoder,
            EncoderBuilder,
        },
        error::Status,
        image::{EncodeInput, Image},
        types::{
            Backend, BitstreamType, ColorSpace, ImageComponentInfo, ImageFormat, ImageInfo,
            ImageType, ProgressionOrder, QualityType, TileDecode,
        },
    };
    pub use crate::image::Size;
}

/// Returns the major version, minor version, or patch level of the loaded nvJPEG 2000 library.
///
/// # Errors
///
/// Returns an error if nvJPEG 2000 fails the query.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `nvjpeg2kGetProperty` receives valid output locations.
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
    // SAFETY: `nvjpeg2kGetCudartProperty` receives valid output locations.
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
/// Returns an error if nvJPEG 2000 fails the query.
pub fn version() -> Result<LibraryVersion> {
    Ok(LibraryVersion::new(
        library_property(LibraryProperty::Major)? as _,
        library_property(LibraryProperty::Minor)? as _,
        library_property(LibraryProperty::Patch)? as _,
    ))
}

/// Returns the CUDA runtime version that nvJPEG 2000 was built with.
///
/// # Errors
///
/// Returns an error if nvJPEG 2000 fails the query.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    let raw = cudart_property(LibraryProperty::Major)? as u64 * 1000
        + cudart_property(LibraryProperty::Minor)? as u64 * 10
        + cudart_property(LibraryProperty::Patch)? as u64;
    Ok(CudaRuntimeVersion::from_raw(raw))
}
