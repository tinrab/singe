//! Safe Rust wrappers for NVIDIA nvImageCodec, for decoding and encoding JPEG, JPEG 2000, TIFF, PNG, BMP, and other image formats on the GPU or CPU.
//!
//! This crate provides:
//!
//! - **Code streams:** [`CodeStream`](code_stream::CodeStream) holds compressed data in a file or in host memory, reports its image information, and selects one image or a region of it.
//! - **Images:** [`Image`](image::Image) and [`ImageInfo`](image::ImageInfo) describe host or device pixel buffers, with plane layouts, sample types, color specs, and chroma subsampling.
//! - **Decoding and encoding:** [`Decoder`](decode::Decoder) and [`Encoder`](encode::Encoder) process one image or a batch, report a status per image, and answer [`can_decode`](decode::Decoder::can_decode) and [`can_encode`](encode::Encoder::can_encode) queries.
//! - **Execution:** [`ExecutionParams`](codec::ExecutionParams) picks the CPU or a CUDA context, and the `unsafe` [`decode_async`](decode::Decoder::decode_async) and [`encode_async`](encode::Encoder::encode_async) methods return a [`Future`](future::Future) without waiting.
//!
//! Codec plugins, executors, and allocators have no safe API; write them against [`singe-nvimagecodec-sys`](sys).
//!
//! # Example
//!
//! Decode a JPEG file into a grayscale image in host memory:
//!
//! ```rust,no_run
//! use std::mem::MaybeUninit;
//!
//! use singe_cuda::context::Context;
//! use singe_nvimagecodec::{
//!     code_stream::CodeStream,
//!     codec::ExecutionParams,
//!     decode::{DecodeParams, Decoder},
//!     image::{Image, ImageInfo, ImagePlaneInfo},
//!     instance::{Instance, InstanceOptions},
//!     types::{ChromaSubsampling, ColorSpec, SampleDataType, SampleFormat},
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let instance = Instance::create(&InstanceOptions::default())?;
//!     let input = CodeStream::from_file(&instance, "image.jpg", None)?;
//!     let input_info = input.image_info()?;
//!     let plane = input_info
//!         .plane(0)
//!         .ok_or_else(|| anyhow::anyhow!("image has no planes"))?;
//!
//!     let (width, height) = (plane.width(), plane.height());
//!     let row_stride = usize::try_from(width)?;
//!     let mut output = vec![MaybeUninit::<u8>::uninit(); row_stride * usize::try_from(height)?];
//!     let planes = [ImagePlaneInfo::create(width, height, row_stride, 1, SampleDataType::U8)?];
//!     let mut output_info = ImageInfo::from_host_output(&mut output, SampleFormat::PlanarY, &planes)?;
//!     output_info
//!         .set_color_spec(ColorSpec::Gray)
//!         .set_chroma_subsampling(ChromaSubsampling::Gray);
//!     let mut output_image = Image::create(&instance, output_info)?;
//!
//!     let context = Context::create()?;
//!     let decoder = Decoder::create(&instance, &ExecutionParams::new(&context), None)?;
//!     decoder.decode_one(&input, &mut output_image, &DecodeParams::new())?;
//!     println!("first row: {:?}", output_image.host_row(0, 0)?);
//!     Ok(())
//! }
//! ```
//!
//! The safe [`decode`](decode::Decoder::decode) and [`encode`](encode::Encoder::encode) methods, and their `_one` forms, wait for all CPU and GPU work and check each image's status before returning.
//! [`ExecutionParams::cpu_only()`](codec::ExecutionParams::cpu_only) runs the codecs that support it on the CPU.
//! The default [`InstanceOptions`](instance::InstanceOptions) load nvImageCodec's codec extension modules from its default install location.
//!
//! # Raw execution
//!
//! [`Decoder::decode_async`](decode::Decoder::decode_async) and [`Encoder::encode_async`](encode::Encoder::encode_async) check and prepare their items as the safe calls do, wait for the work that produces device inputs, submit, and return a [`Future`](future::Future) without waiting.
//! The caller must:
//!
//! - keep every code stream, image, pixel buffer, and parameter value valid until the future and the GPU work on the images' streams finish, including after an error or unwind;
//! - keep anything else from reading or writing the outputs until then.
//!
//! [`Future::wait`](future::Future::wait) waits for both and checks every image's status; dropping a future waits only for the CPU work.
//! The safe [`decode`](decode::Decoder::decode) and [`encode`](encode::Encoder::encode) wait for both before returning.

#[cfg(not(feature = "nvimagecodec_0_9"))]
compile_error!("singe-nvimagecodec requires a version feature, such as `nvimagecodec_0_9`");

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

/// Futures returned by `decode_async` and `encode_async`.
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

use singe_core::PackedVersion;
pub use singe_core::{CudaVersion, LibraryVersion};
pub use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result, try_ffi},
    types::{NativeStruct, Properties},
};

/// Returns the properties of the loaded nvImageCodec library.
///
/// # Errors
///
/// Returns an error if nvImageCodec cannot report its properties.
pub fn properties() -> Result<Properties> {
    let mut properties = sys::nvimgcodecProperties_t::init();

    // SAFETY: `properties` is a writable local whose structure header is initialized, as nvImageCodec requires.
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
        return Err(Error::out_of_range("nvimagecodec version"));
    }

    LibraryVersion::from_packed(version, PackedVersion::Major1000Minor100)
        .ok_or_else(|| Error::out_of_range("nvimagecodec version"))
}

/// Returns the CUDA runtime version that nvImageCodec was built with.
///
/// # Errors
///
/// Returns an error if nvImageCodec cannot report its properties.
pub fn cudart_version() -> Result<CudaVersion> {
    let version = properties()?.cudart_version();
    if version == 0 {
        return Err(Error::out_of_range("nvimagecodec cudart version"));
    }

    CudaVersion::from_raw(version).ok_or_else(|| Error::out_of_range("nvimagecodec cudart version"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        assert_eq!(
            (version.major(), version.minor()),
            (sys::NVIMGCODEC_VER_MAJOR, sys::NVIMGCODEC_VER_MINOR),
            "the loaded nvImageCodec {version} differs from the bindings' version"
        );
        assert_ne!(cudart_version()?.major(), 0);
        Ok(())
    }
}
