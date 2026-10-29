//! Safe NVIDIA Performance Primitives wrappers for image and signal APIs.
//!
//! This crate wraps NPP version queries, stream contexts, image and signal
//! device memory, image processing pipelines, signal processing pipelines, typed
//! image views, channel layouts, sizes, rectangles, and temporary workspaces.
//!
//! # Example
//!
//! ```no_run
//! use singe_cuda::{context::Context, memory::DeviceMemory};
//! use singe_npp::{
//!     C3, Image, ImagePipeline, ImageView, StreamContext, Workspace,
//!     types::{ComparisonOperation, InterpolationMode, Size},
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let context = Context::create()?;
//! let stream = context.create_stream()?;
//! let stream_context = StreamContext::create(&stream)?;
//! let mut workspace = Workspace::create();
//!
//! let size = Size::new(640, 480);
//! let device_memory = DeviceMemory::<u8>::zeroed(&context, 640 * 480 * 3)?;
//! let source = ImageView::<u8, C3>::from_buffer(&device_memory, size)?;
//!
//! let output = ImagePipeline::from_view(&stream_context, &mut workspace, source)?
//!     .resize(
//!         InterpolationMode::Lanczos,
//!         Image::<u8, C3>::filled(&stream_context, Size::new(256, 256), 0)?,
//!     )?
//!     .rgb_to_gray()?
//!     .threshold(128, ComparisonOperation::Less)?
//!     .filter_sharpen()?
//!     .finish()?;
//! let image = output.copy_to_host_vec()?;
//!
//! println!("NPP version: {:?}", singe_npp::version()?);
//! # Ok(())
//! # }
//! ```

/// Application-managed NPP stream contexts and synchronization.
pub mod context;
/// Typed NPP errors, statuses, and result aliases.
pub mod error;
/// Image memory, typed views, and processing operations.
pub mod image;
pub mod memory;
pub mod pipeline;
/// Signal memory, typed views, and processing operations.
pub mod signal;
/// Native-compatible enums, descriptors, geometry, and numeric types.
pub mod types;
pub(crate) mod utility;
pub mod workspace;

#[cfg(feature = "testing")]
/// CUDA stream-context setup for tests.
pub mod testing;

// TODO: refactor into mods
use std::ptr;

pub use context::StreamContext;
pub use error::{Error, Result};
pub use image::{
    AC4, AllChannels, AlphaIgnoredRgba, C1, C2, C3, C4, ChannelLayout, Channels2, Gray, Image,
    ImageOutputView, ImageView, ImageViewMut, ImageWindow, MaskOutputView, MaskView, MaskViewMut,
    Rgb, Rgba, SupportedImage,
};
pub use pipeline::{
    ImageBacking, ImagePipeline, SignalBacking, SignalPipeline, UninitImagePipeline,
    UninitSignalPipeline, Workspace,
};
pub use signal::{Signal, SignalOutputView, SignalView, SignalViewMut, SupportedSignal};
use singe_core::LibraryVersion;
use singe_npp_sys as sys;

use crate::utility::to_u64;

/// Returns the loaded NPP library version.
///
/// # Errors
///
/// Returns an error if NPP returns a null version pointer.
pub fn version() -> Result<LibraryVersion> {
    // SAFETY: `nppGetLibVersion` takes no arguments.
    let raw = unsafe { sys::nppGetLibVersion() };
    if raw.is_null() {
        return Err(Error::NullHandle);
    }

    // SAFETY: `nppGetLibVersion` returned a non-null pointer to a static version struct.
    let raw = unsafe { ptr::read(raw) };
    Ok(LibraryVersion::new(
        to_u64(raw.major, "npp major version")?,
        to_u64(raw.minor, "npp minor version")?,
        to_u64(raw.build, "npp build version")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        assert_ne!(version.major(), 0);
        Ok(())
    }
}

mod optional;
