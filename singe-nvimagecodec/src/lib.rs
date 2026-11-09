//! Safe Rust wrappers for the NVIDIA nvImageCodec library.
//!
//! Create an [`Instance`](instance::Instance) first; code streams, images, and codecs are created from it.
//! A [`CodeStream`](code_stream::CodeStream) is compressed data to decode or a destination to encode into, and an [`Image`](image::Image) describes the pixels and the host or device memory that holds them.
//! [`Decoder`](decode::Decoder) and [`Encoder`](encode::Encoder) process batches of them, on the CPU or on the GPU as [`ExecutionParams`](codec::ExecutionParams) selects.
//!
//! The safe `decode` and `encode` methods wait for all CPU and GPU work before returning, so their outputs are ready to use.
//! The `unsafe` `*_raw_async` methods take native handles and return a [`Future`](future::Future); the caller keeps everything they use alive until the work finishes.
//!
//! Writing codec plugins, executors, and allocators is not covered by the safe API; use `singe-nvimagecodec-sys` for those.
//!
//! # Example
//!
//! Decode a JPEG file into a grayscale image in host memory:
//!
//! ```no_run
//! use std::mem::MaybeUninit;
//! use singe_cuda::{context::Context, device::Device};
//! use singe_nvimagecodec::{
//!     code_stream::CodeStream,
//!     codec::ExecutionParams,
//!     decode::{DecodeParams, Decoder},
//!     error::Result,
//!     image::{Image, ImageInfo, ImagePlaneInfo},
//!     instance::Instance,
//!     types::{
//!         ChromaSubsampling, ColorSpec, SampleDataType, SampleFormat,
//!     },
//! };
//!
//! fn main() -> Result<()> {
//!     let instance = Instance::create_default()?;
//!     let input = CodeStream::from_file(&instance, "image.jpg", None)?;
//!     let input_info = input.image_info()?;
//!     let plane = input_info.plane(0).expect("image has a plane");
//!
//!     let width = plane.width();
//!     let height = plane.height();
//!     let row_stride = width as usize;
//!     let mut output = vec![MaybeUninit::<u8>::uninit(); row_stride * height as usize];
//!     let planes = [ImagePlaneInfo::create(
//!         width,
//!         height,
//!         row_stride,
//!         1,
//!         SampleDataType::U8,
//!     )?];
//!     let mut output_info =
//!         ImageInfo::from_host_output(&mut output, SampleFormat::PlanarY, &planes)?;
//!     output_info
//!         .set_color_spec(ColorSpec::Gray)
//!         .set_chroma_subsampling(ChromaSubsampling::Gray);
//!     let mut output_image = Image::create(&instance, output_info)?;
//!
//!     let context = Context::retain_primary_for_device(Device::new(0))?;
//!     let mut decoder = Decoder::create(&instance, &ExecutionParams::new(&context), None)?;
//!     decoder.decode_one(&input, &mut output_image, &DecodeParams::new())?;
//!     let first_row = output_image.host_row(0, 0)?;
//!     assert_eq!(first_row.len(), row_stride);
//!     Ok(())
//! }
//! ```

/// Compressed image data to decode from or encode into.
pub mod code_stream;

/// Execution settings shared by decoders and encoders.
pub mod codec;

/// Debug messages from nvImageCodec.
pub mod debug;

/// Decoding code streams into images.
pub mod decode;

/// Encoding images into code streams.
pub mod encode;

/// Error and status types.
pub mod error;

/// Futures returned by raw decode and encode submissions.
pub mod future;

/// Images, their pixel layout, and the memory that holds their pixels.
pub mod image;

/// The library instance and its settings.
pub mod instance;

/// Options for creating decoders and encoders.
pub mod options;

/// Image formats, processing statuses, library properties, and other shared types.
pub mod types;

pub(crate) mod utility;

#[cfg(feature = "testing")]
/// Shared setup for tests that need a CUDA device and an nvImageCodec instance.
pub mod testing;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    types::Properties,
    utility::NativeStruct,
};

/// Returns the properties of the loaded nvImageCodec library.
///
/// # Errors
///
/// Returns an error if nvImageCodec cannot report its properties.
pub fn properties() -> Result<Properties> {
    let mut properties = sys::nvimgcodecProperties_t::init();

    // SAFETY: `nvimgcodecGetProperties` receives valid output locations.
    unsafe {
        try_ffi!(sys::nvimgcodecGetProperties(&raw mut properties))?;
    }

    Ok(Properties::from_raw(properties))
}

/// Returns the version of the loaded nvImageCodec library.
///
/// # Errors
///
/// Returns an error if nvImageCodec cannot report its properties.
pub fn version() -> Result<LibraryVersion> {
    let version = properties()?.version();
    if version == 0 {
        return Err(Error::InvalidVersion);
    }

    Ok(LibraryVersion::new(
        (version / 10000).into(),
        ((version % 10000) / 100).into(),
        (version % 100).into(),
    ))
}

/// Returns the CUDA runtime version that nvImageCodec was built with.
///
/// # Errors
///
/// Returns an error if nvImageCodec cannot report its properties.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    let version = properties()?.cudart_version();
    if version == 0 {
        return Err(Error::InvalidVersion);
    }

    Ok(CudaRuntimeVersion::from(version))
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
    fn test_setup_context_creates_instance() -> Result<()> {
        let _context = setup_context()?;
        Ok(())
    }
}
